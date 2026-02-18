"""
Check 46 - Process Frequency Check
==================================
Validates that process checks (sampling, monitoring) follow the defined frequency
intervals with allowed tolerances (e.g., "every 30 min ± 5 min").

This check:
1. Identifies pages with frequency definitions (e.g., "every 30 min ± 5 min")
2. Tracks frequency context across pages within the same section
3. Validates that timestamps between consecutive records fall within allowed gaps
4. Reports anomalies where frequency violations occur
"""

import json
import re
import html
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# 1. FREQUENCY PATTERN EXTRACTION
# ============================================================================

FREQ_REGEX = re.compile(
    r"every\s+(\d+)\s*"
    r"(min|mins|minute|minutes|hr|hrs|hour|hours)\.?\s*"
    r"(?:±|\+/-|\*|\+-|\\pm|\s)+\s*"  # Tolerance separators
    r"(\d+)\s*"
    r"(min|mins|minute|minutes|hr|hrs|hour|hours)\.?",
    re.IGNORECASE,
)

# Section detection keywords
SECTION_HINT_WORDS = (
    "process", "checks", "check", "verification", "quality", "sampling",
    "in process", "in-process", "procedure", "record", "inspection",
    "filling", "stoppering", "clarity", "printing", "weight"
)


def parse_frequency_from_markdown(markdown: str) -> List[Dict[str, Any]]:
    """
    Extract all frequency mentions from markdown text.
    Returns a list of dicts with target_minutes, tolerance_minutes, role.
    """
    results = []
    for match in FREQ_REGEX.finditer(markdown):
        target_val = int(match.group(1))
        target_unit = match.group(2).lower()
        tolerance_val = int(match.group(3))
        tolerance_unit = match.group(4).lower()

        # Convert to minutes
        if "hr" in target_unit or "hour" in target_unit:
            target_minutes = target_val * 60
        else:
            target_minutes = target_val

        if "hr" in tolerance_unit or "hour" in tolerance_unit:
            tolerance_minutes = tolerance_val * 60
        else:
            tolerance_minutes = tolerance_val

        # Detect role from surrounding context (look ±50 chars around match)
        start = max(0, match.start() - 50)
        end = min(len(markdown), match.end() + 50)
        context = markdown[start:end].lower()

        role = "GENERIC"
        if "for sm" in context or "(sm)" in context:
            role = "SM"
        elif "for qa" in context or "(qa)" in context:
            role = "QA"

        results.append({
            "target_minutes": target_minutes,
            "tolerance_minutes": tolerance_minutes,
            "role": role,
            "raw_text": match.group(0)
        })

    return results


# ============================================================================
# 2. HTML/MARKDOWN CLEANING
# ============================================================================

def clean_html_for_regex(raw_html: str) -> str:
    """Converts HTML/MathML into clean plain text for regex matching."""
    if not raw_html:
        return ""
    
    # Handle <math> tags: preserve content
    text = re.sub(r'<math[^>]*>(.*?)</math>', r' \1 ', raw_html, flags=re.IGNORECASE)
    
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================================================
# 3. SECTION-BASED STATE MANAGEMENT
# ============================================================================

@dataclass
class SectionState:
    """Tracks the current section and its frequency status."""
    marker: Optional[str] = None
    has_frequency: bool = False
    frequency_info: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.frequency_info is None:
            self.frequency_info = []


def _as_text(x: Any) -> str:
    """Convert various types to text."""
    if not x:
        return ""
    if isinstance(x, list):
        return "\n".join([str(i) for i in x if i is not None])
    return str(x)


def extract_section_marker(page_obj: dict) -> Optional[str]:
    """
    Find a 'process step' / section header-like line.
    Returns a normalized marker string (lowercased) or None.
    """
    # Check rules_or_instructions first (often contains clean step titles)
    roi = page_obj.get("rules_or_instructions")
    if isinstance(roi, list):
        candidates = roi
    else:
        candidates = _as_text(roi).splitlines()

    for raw in candidates:
        line = str(raw).strip()
        if not line:
            continue

        # Skip obvious non-headers
        if len(line) < 6 or len(line) > 120:
            continue
        if re.fullmatch(r"[\d\W]+", line):  # only numbers/punct
            continue
        if line.lower().startswith(("note", "*note", "checked by", "balance id")):
            continue

        # Header-like: mostly uppercase OR ends with ':' OR contains hint words
        letters = re.sub(r"[^A-Za-z]+", "", line)
        if not letters:
            continue
        upper_ratio = sum(c.isupper() for c in letters) / max(1, len(letters))

        looks_like_title = (
            upper_ratio >= 0.70
            or line.endswith(":")
            or any(w in line.lower() for w in SECTION_HINT_WORDS)
        )

        if looks_like_title:
            return re.sub(r"\s+", " ", line).strip().lower().rstrip(":")

    # Fallback: look into markdown_page (after HTML normalization)
    md = page_obj.get("markdown_page") or ""
    cleaned = clean_html_for_regex(md)
    for raw in cleaned.splitlines():
        line = raw.strip()
        if len(line) < 6 or len(line) > 120:
            continue
        letters = re.sub(r"[^A-Za-z]+", "", line)
        if not letters:
            continue
        upper_ratio = sum(c.isupper() for c in letters) / max(1, len(letters))
        if upper_ratio >= 0.75:
            return re.sub(r"\s+", " ", line).strip().lower().rstrip(":")

    return None


def looks_like_frequency_log(page_obj: dict) -> bool:
    """Check if page looks like a frequency log/table page."""
    text = clean_html_for_regex(page_obj.get("markdown_page") or "").lower()
    has_date_time = ("date" in text and "time" in text)
    has_check_columns = ("checked by" in text or "done by" in text or "recorded by" in text)
    return has_date_time or has_check_columns


# ============================================================================
# 4. CLASSIFY PAGES: Section-based state with frequency inheritance
# ============================================================================

def compute_frequency_pages(bmr_pages: Dict[str, Any]) -> Tuple[Dict[str, bool], Dict[str, List[Dict[str, Any]]]]:
    """
    Returns:
      - is_freq_page: {page_no: bool} indicating if page is freq/continuation
      - freq_info: {page_no: [freq_dicts]} where freq_dicts contain target/tolerance/role
    """
    keys = list(bmr_pages.keys())
    try:
        keys.sort(key=lambda k: int(k))
    except Exception:
        keys.sort()

    is_freq_page: Dict[str, bool] = {}
    freq_info: Dict[str, List[Dict[str, Any]]] = {}
    state = SectionState()

    for k in keys:
        page = bmr_pages[k]
        
        # 1) Detect section boundary
        marker = extract_section_marker(page)
        if marker is not None and marker != state.marker:
            # New section starts here - reset state
            state.marker = marker
            state.has_frequency = False
            state.frequency_info = []

        # 2) Check if this page explicitly defines frequency
        raw_markdown = page.get("markdown_page") or ""
        markdown = clean_html_for_regex(raw_markdown)
        
        # Also check rules_or_instructions as a secondary source
        if not FREQ_REGEX.search(markdown):
            rules = " ".join(page.get("rules_or_instructions", []))
            markdown = clean_html_for_regex(rules)

        freqs = parse_frequency_from_markdown(markdown)
        
        if freqs:
            # This page explicitly defines frequency - lock it for the section
            state.has_frequency = True
            state.frequency_info = freqs
            is_freq_page[k] = True
            freq_info[k] = freqs
            continue

        # 3) Section-based carry-forward
        if state.has_frequency and looks_like_frequency_log(page):
            is_freq_page[k] = True
            freq_info[k] = state.frequency_info
        else:
            is_freq_page[k] = False

    return is_freq_page, freq_info


# ============================================================================
# 5. TIMESTAMP PARSING
# ============================================================================

def parse_timestamp(date_str: str, time_str: str) -> Optional[datetime]:
    """
    Parse date and time into a datetime object.
    Date formats: 10-Apr-25, 10-Apr-2025, 6-Apr-25
    Time format: HH:MM (24-hour, leading zeros optional)
    """
    if not date_str or not time_str:
        return None

    # Normalize 'HH.MM' -> 'HH:MM'
    t = time_str.strip()
    if re.match(r"^\d{1,2}\.\d{2}$", t):
        t = t.replace(".", ":")

    # Parse time (strict HH:MM)
    time_pattern = re.compile(r"^(\d{1,2}):(\d{2})$")
    time_match = time_pattern.match(t)
    if not time_match:
        return None

    hour = int(time_match.group(1))
    minute = int(time_match.group(2))
    if hour > 23 or minute > 59:
        return None

    # Parse date (D-Mon-YY or D-Mon-YYYY)
    date_pattern = re.compile(r"^(\d{1,2})-([A-Za-z]{3})-(\d{2,4})$")
    date_match = date_pattern.match(date_str.strip())
    if not date_match:
        return None

    day = int(date_match.group(1))
    month_str = date_match.group(2).capitalize()
    year_str = date_match.group(3)

    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    month = month_map.get(month_str)
    if not month:
        return None

    year = int(year_str)
    if year < 100:
        year += 2000

    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


def detect_role_from_record(rec: Dict[str, Any]) -> str:
    """Detect role (SM/QA/GENERIC) from record keys or values."""
    # Check keys
    for k in rec.keys():
        k_lower = k.lower()
        if "done by (sm)" in k_lower or "recorded by (sm)" in k_lower or "done by sm" in k_lower:
            return "SM"
        if "done by (qa)" in k_lower or "recorded by (qa)" in k_lower or "done by qa" in k_lower or "sampling done by (qa)" in k_lower:
            return "QA"

    # Check values
    for v in rec.values():
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower == "sm":
                return "SM"
            if v_lower == "qa":
                return "QA"

    return "GENERIC"


def flatten_records(records: Any) -> List[Dict[str, Any]]:
    """Flatten nested records structure."""
    flat = []
    if isinstance(records, dict):
        flat.append(records)
    elif isinstance(records, list):
        for r in records:
            if isinstance(r, dict):
                flat.append(r)
            elif isinstance(r, list):
                flat.extend(x for x in r if isinstance(x, dict))
    return flat


# ============================================================================
# 6. FREQUENCY VALIDATION
# ============================================================================

def validate_page_frequency(page_no: str,
                            page: Dict[str, Any],
                            freq_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate frequency for a single page using ONLY adjacent records.
    For each pair of consecutive records with same role, check if the
    time gap is within the allowed frequency window.
    """
    records = flatten_records(page.get("records", []))

    # Build frequency map by role
    freq_map: Dict[str, Dict[str, int]] = {}
    for freq in freq_list:
        role = freq["role"]
        freq_map[role] = {
            "target": freq["target_minutes"],
            "tolerance": freq["tolerance_minutes"]
        }

    violations = []

    # Walk records in original order, check only adjacent pairs
    for i in range(len(records) - 1):
        rec1 = records[i]
        rec2 = records[i + 1]

        # Detect role of each record
        role1 = detect_role_from_record(rec1)
        role2 = detect_role_from_record(rec2)

        # Only compare within the same role
        if role1 != role2:
            continue

        role = role1

        # Find applicable frequency for this role
        applicable_freq = None
        if role in freq_map:
            applicable_freq = freq_map[role]
        elif "GENERIC" in freq_map:
            applicable_freq = freq_map["GENERIC"]

        if not applicable_freq:
            continue

        # Parse timestamps for both records
        date1 = rec1.get("Date") or rec1.get("Sample Date") or rec1.get("date") or ""
        time1 = rec1.get("Time") or rec1.get("time") or ""
        date2 = rec2.get("Date") or rec2.get("Sample Date") or rec2.get("date") or ""
        time2 = rec2.get("Time") or rec2.get("time") or ""

        ts1 = parse_timestamp(date1, time1)
        ts2 = parse_timestamp(date2, time2)

        # If either timestamp invalid, skip this pair
        if not ts1 or not ts2:
            continue

        # Compute gap in minutes
        gap_minutes = (ts2 - ts1).total_seconds() / 60

        target = applicable_freq["target"]
        tolerance = applicable_freq["tolerance"]
        low = target - tolerance
        high = target + tolerance

        if gap_minutes < low or gap_minutes > high:
            violations.append({
                "role": role,
                "pair_index": i,
                "timestamps": [
                    ts1.strftime("%Y-%m-%d %H:%M"),
                    ts2.strftime("%Y-%m-%d %H:%M")
                ],
                "gap_minutes": round(gap_minutes, 2),
                "allowed_window_minutes": [low, high]
            })

    if violations:
        return {
            "page_no": page_no,
            "section_name": "Process Parameters",
            "check_name": "process check frequency discrepancy",
            "anomaly_status": 1
        }
    else:
        return {
            "page_no": page_no,
            "section_name": "Process Parameters",
            "check_name": "process check frequency discrepancy",
            "anomaly_status": 0
        }


# ============================================================================
# 7. MAIN FUNCTION (Check 46)
# ============================================================================

def process_frequency_check(bmr_pages: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pure Python frequency validation:
    1. Identify explicit frequency pages and continuation pages using section-based state
    2. Extract frequency info (target, tolerance, role)
    3. Parse timestamps, group by role, validate gaps
    4. Return anomaly results per page
    """
    # Classify pages using section-based state
    is_freq_page, freq_info = compute_frequency_pages(bmr_pages)
    print(f"Frequency pages detected: {is_freq_page}")
    print(f"Frequency info: {freq_info}")

    # Sort pages
    keys = list(bmr_pages.keys())
    try:
        keys.sort(key=lambda k: int(k))
    except Exception:
        keys.sort()

    results = []

    for page_no in keys:
        if not is_freq_page.get(page_no, False):
            # Not a frequency page
            results.append({
                "page_no": page_no,
                "section_name": "Process Parameters",
                "check_name": "process check frequency discrepancy",
                "anomaly_status": 0
            })
            continue

        # Validate frequency
        page = bmr_pages[page_no]
        freq_list = freq_info.get(page_no, [])

        if not freq_list:
            results.append({
                "page_no": page_no,
                "section_name": "Process Parameters",
                "check_name": "process check frequency discrepancy",
                "anomaly_status": 0
            })
            continue

        result = validate_page_frequency(page_no, page, freq_list)
        results.append(result)

    return results


# ============================================================================
# 8. ENTRY POINT
# ============================================================================

def load_json_file(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



if __name__ == "__main__":
    import sys
    import os
    
    # Add yog_checks to path so we can import extract_data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    yog_checks_dir = os.path.join(parent_dir, 'yog_checks')
    if yog_checks_dir not in sys.path:
        sys.path.append(yog_checks_dir)

    from extract_data import get_filled_master_json
    
    # Load OCR data from the complete_data JSON file
    complete_data_path = "/home/softsensor/Desktop/Amneal/challenge_bmr/05jan_AH250076_50Checks 1.json"
    
    print("Running Check 46 - Process Frequency Check...")
    
    # Extract the filled_master_json from the complete data (returns a list)
    filled_master_json = get_filled_master_json(complete_data_path)
    print(f"Loaded {len(filled_master_json)} pages from filled_master_json")
    
    # Convert list to dict format expected by process_frequency_check
    # Format: {"1": {page_data}, "2": {page_data}, ...}
    bmr_pages = {}
    for page_obj in filled_master_json:
        page_no = str(page_obj.get("page", len(bmr_pages) + 1))
        
        # Flatten page_content into the page object for easier access
        page_data = {
            "page": page_obj.get("page"),
            "markdown_page": page_obj.get("markdown_page", ""),
            "additional_keys": page_obj.get("additional_keys", []),
            "rules_or_instructions": [],
            "records": []
        }
        
        # Extract data from page_content
        for content in page_obj.get("page_content", []):
            content_type = content.get("type", "")
            
            if content_type == "table":
                table_json = content.get("table_json", {})
                # Check if it has records
                if "records" in table_json:
                    page_data["records"].extend(table_json["records"])
                else:
                    # Store other table data
                    if not page_data.get("table_data"):
                        page_data["table_data"] = []
                    page_data["table_data"].append(table_json)
            
            elif content_type == "kv_text_block":
                kv_data = content.get("extracted_kv_text_block", {})
                # Handle both dict and list formats
                if isinstance(kv_data, list):
                    for kv in kv_data:
                        if isinstance(kv, dict):
                            rules = kv.get("rules_or_instructions", [])
                            if rules:
                                page_data["rules_or_instructions"].extend(rules)
                elif isinstance(kv_data, dict):
                    rules = kv_data.get("rules_or_instructions", [])
                    if rules:
                        page_data["rules_or_instructions"].extend(rules)
        
        bmr_pages[page_no] = page_data
    
    # Run the process frequency check
    results = process_frequency_check(bmr_pages)
    print(json.dumps(results, indent=2))