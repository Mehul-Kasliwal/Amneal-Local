"""
Check 46 - Process Frequency Check (OOP Version)
================================================
Validates that process checks (sampling, monitoring) follow the defined frequency
intervals with allowed tolerances (e.g., "every 30 min ± 5 min").
"""

import json
import re
import html
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


class ProcessFrequencyValidator:
    """
    Check 46: AI tool should verify process check frequency compliance.
    ---
    Description:
    1. Identify pages with frequency definitions (e.g., "every 30 min ± 5 min")
    2. Track frequency context across pages within the same section
    3. Validate that timestamps between consecutive records fall within allowed gaps
    4. Report anomalies where frequency violations occur
    ---
    Type: Python-based
    ---
    Attachment Required: NO
    ---
    Author: Mehul Kasliwal (30 Jan 2026)
    """

    # ===================== CONSTANTS =====================

    SECTION_NAME = "Process Parameters"
    CHECK_NAME = "process check frequency discrepancy"

    # Frequency extraction regex
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

    # ===================== DATA CLASSES =====================

    @dataclass
    class SectionState:
        """Tracks the current section and its frequency status."""
        marker: Optional[str] = None
        has_frequency: bool = False
        frequency_info: List[Dict[str, Any]] = None
        table_keys: set = None
        last_freq_page_no: Optional[int] = None

        def __post_init__(self):
            if self.frequency_info is None:
                self.frequency_info = []
            if self.table_keys is None:
                self.table_keys = set()

    # ===================== HELPER METHODS =====================

    def _as_text(self, x: Any) -> str:
        """Convert various types to text."""
        if not x:
            return ""
        if isinstance(x, list):
            return "\n".join([str(i) for i in x if i is not None])
        return str(x)

    def _clean_html_for_regex(self, raw_html: str) -> str:
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

    def _parse_frequency_from_markdown(self, markdown: str) -> List[Dict[str, Any]]:
        """
        Extract all frequency mentions from markdown text.
        Returns a list of dicts with target_minutes, tolerance_minutes, role.
        """
        results = []
        seen = set()  # deduplicate identical (target, tolerance, role) tuples
        for match in self.FREQ_REGEX.finditer(markdown):
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

            # Detect role from surrounding context (look up to 150 chars forward)
            # as roles (e.g., PAP-SM-022) typically appear after the rule in the BMR.
            context = markdown[match.end():min(len(markdown), match.end() + 150)].lower()

            role = "GENERIC"
            # Find the nearest mention of QA vs SM patterns
            qa_idx = -1
            sm_idx = -1

            for pat in ["(qa)", " qa ", "-qa-", "for qa"]:
                idx = context.find(pat)
                if idx != -1:
                    qa_idx = idx if qa_idx == -1 else min(qa_idx, idx)

            for pat in ["(sm)", " sm ", "-sm-", "for sm", "by (sm)"]:
                idx = context.find(pat)
                if idx != -1:
                    sm_idx = idx if sm_idx == -1 else min(sm_idx, idx)

            if qa_idx != -1 and sm_idx != -1:
                # Both found - pick the one that appears first after the match
                role = "QA" if qa_idx < sm_idx else "SM"
            elif qa_idx != -1:
                role = "QA"
            elif sm_idx != -1:
                role = "SM"

            # Deduplicate: same (target, tolerance, role) pair only stored once
            key = (target_minutes, tolerance_minutes, role)
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "target_minutes": target_minutes,
                "tolerance_minutes": tolerance_minutes,
                "role": role,
                "raw_text": match.group(0)
            })

        return results

    def _extract_section_marker(self, page_obj: dict) -> Optional[str]:
        """
        Find a 'process step' / section header-like line.
        Returns a normalized marker string (lowercased) or None.
        """
        # Check rules_or_instructions first (often contains clean step titles)
        roi = page_obj.get("rules_or_instructions")
        if isinstance(roi, list):
            candidates = roi
        else:
            candidates = self._as_text(roi).splitlines()

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
                or any(w in line.lower() for w in self.SECTION_HINT_WORDS)
            )

            if looks_like_title:
                return re.sub(r"\s+", " ", line).strip().lower().rstrip(":")

        # Fallback: look into markdown_page (after HTML normalization)
        md = page_obj.get("markdown_page") or ""
        cleaned = self._clean_html_for_regex(md)
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

    def _looks_like_frequency_log(self, page_obj: dict) -> bool:
        """Check if page looks like a frequency log/table page.
        Checks both the markdown text AND the record column names."""
        # Check markdown text
        text = self._clean_html_for_regex(page_obj.get("markdown_page") or "").lower()
        has_date_time = ("date" in text and "time" in text)
        has_check_columns = ("checked by" in text or "done by" in text or "recorded by" in text)
        if has_date_time or has_check_columns:
            return True

        # Also check record column names (table headers)
        records = self._flatten_records(page_obj.get("records", []))
        if records:
            all_keys = set()
            for rec in records[:3]:  # check first few records
                if isinstance(rec, dict):
                    all_keys.update(k.lower() for k in rec.keys())
            has_date_time_keys = any("date" in k for k in all_keys) and any("time" in k for k in all_keys)
            has_role_keys = any("done by" in k or "checked by" in k or "recorded by" in k for k in all_keys)
            if has_date_time_keys or has_role_keys:
                return True

        return False

    def _detect_page_role(self, page_obj: dict) -> str:
        """Detect the page's role (SM/QA/GENERIC) from record column names.
        Looks for patterns like 'Done by (SM)', 'Done by (QA)', 'Sampling done by (QA)', etc."""
        records = self._flatten_records(page_obj.get("records", []))
        if not records:
            return "GENERIC"

        # Inspect column names from the first record
        for rec in records[:3]:
            if not isinstance(rec, dict):
                continue
            for key in rec.keys():
                k_lower = key.lower()
                if any(pattern in k_lower for pattern in ("done by", "checked by", "recorded by", "sampling done by")):
                    if "(sm)" in k_lower or "sm)" in k_lower:
                        return "SM"
                    if "(qa)" in k_lower or "qa)" in k_lower:
                        return "QA"
        return "GENERIC"

    def _compute_frequency_pages(
        self, bmr_pages: Dict[str, Any]
    ) -> Tuple[Dict[str, bool], Dict[str, List[Dict[str, Any]]], Dict[str, bool], Dict[str, str]]:
        """
        Returns:
          - is_freq_page: {page_no: bool} indicating if page is freq/continuation
          - freq_info: {page_no: [freq_dicts]} where freq_dicts contain target/tolerance/role
          - defines_own_freq: {page_no: bool} True if the page explicitly defines a
            new frequency (vs. inheriting from a prior page in the same section)
          - page_roles: {page_no: str} detected role (SM/QA/GENERIC) from table columns
        """
        keys = list(bmr_pages.keys())
        try:
            keys.sort(key=lambda k: int(k))
        except Exception:
            keys.sort()

        is_freq_page: Dict[str, bool] = {}
        freq_info: Dict[str, List[Dict[str, Any]]] = {}
        defines_own_freq: Dict[str, bool] = {}
        page_roles: Dict[str, str] = {}
        state = self.SectionState()

        for k in keys:
            page = bmr_pages[k]
            
            try:
                p_int = int(k)
            except ValueError:
                p_int = None

            # Detect page role from table column names
            page_roles[k] = self._detect_page_role(page)

            # Extract table keys for this page
            records = self._flatten_records(page.get("records", []))
            current_keys = set()
            for rec in records[:1]:
                if isinstance(rec, dict):
                    current_keys = set(str(key).lower() for key in rec.keys())
            
            # Determine if this page is a continuation based on robust table signature 
            # (>80% overlap to handle OCR noise) and strict sequential page numbering
            is_continuation_by_table = False
            if state.has_frequency and current_keys and state.table_keys:
                overlap = len(current_keys & state.table_keys) / max(1, len(state.table_keys))
                if overlap > 0.80:
                    # Table matches closely. Check if it's strictly sequential.
                    if p_int is not None and state.last_freq_page_no is not None:
                        if p_int == state.last_freq_page_no + 1:
                            is_continuation_by_table = True

            # 1) Detect section boundary
            marker = self._extract_section_marker(page)
            if marker is not None and marker != state.marker and not is_continuation_by_table:
                # New section starts here - reset state
                state.marker = marker
                state.has_frequency = False
                state.frequency_info = []
                state.table_keys = set()
                state.last_freq_page_no = None

            # 2) Check if this page explicitly defines frequency
            raw_markdown = page.get("markdown_page") or ""
            markdown = self._clean_html_for_regex(raw_markdown)
            
            # Also check rules_or_instructions as a secondary source
            if not self.FREQ_REGEX.search(markdown):
                rules = " ".join(page.get("rules_or_instructions", []))
                markdown = self._clean_html_for_regex(rules)

            freqs = self._parse_frequency_from_markdown(markdown)
            
            if freqs:
                # This page explicitly defines frequency - lock it for the section
                state.has_frequency = True
                state.frequency_info = freqs
                state.table_keys = current_keys
                state.last_freq_page_no = p_int
                is_freq_page[k] = True
                freq_info[k] = freqs
                defines_own_freq[k] = True
                continue

            # 3) Section-based carry-forward
            if state.has_frequency and (is_continuation_by_table or self._looks_like_frequency_log(page)):
                
                # If we're carrying forward solely based on `looks_like_frequency_log` (e.g. no explicit table keys yet),
                # we also require strict sequential pages to prevent unbounded bleeding across sections.
                is_valid_continuation = is_continuation_by_table
                
                if not is_valid_continuation and self._looks_like_frequency_log(page):
                    if p_int is not None and state.last_freq_page_no is not None:
                        if p_int == state.last_freq_page_no + 1:
                            is_valid_continuation = True
                
                if is_valid_continuation:
                    is_freq_page[k] = True
                    freq_info[k] = state.frequency_info
                    defines_own_freq[k] = False
                    state.last_freq_page_no = p_int
                    continue

            # If we fall through to here, the page is not a frequency page.
            # Breaking the continuation sequence means we also shouldn't carry over frequency to subsequent pages.
            # Because if page 80 is not a continuation, page 81 shouldn't suddenly become one.
            is_freq_page[k] = False
            defines_own_freq[k] = False
            state.has_frequency = False
            state.frequency_info = []
            state.table_keys = set()
            state.last_freq_page_no = None

        return is_freq_page, freq_info, defines_own_freq, page_roles

    def _parse_timestamp(self, date_str: str, time_str: str) -> Optional[datetime]:
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

    def _detect_role_from_record(self, rec: Dict[str, Any]) -> str:
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

    def _flatten_records(self, records: Any) -> List[Dict[str, Any]]:
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

    def _validate_page_frequency(
        self,
        page_no: str,
        page: Dict[str, Any],
        freq_list: List[Dict[str, Any]],
        page_role: str = "GENERIC",
        prev_page_last: Optional[Tuple[str, datetime]] = None
    ) -> Tuple[Dict[str, Any], Optional[Tuple[str, datetime]], List[Dict[str, Any]]]:
        """
        Validate frequency for a single page using adjacent records,
        INCLUDING a cross-page boundary check against the previous page's
        last record.

        Args:
            page_role: The role detected from the page's table column names
                       (SM/QA/GENERIC). Used to select the applicable frequency.
            prev_page_last: Optional (role, timestamp) of the last valid
                            record from the preceding frequency page.

        Returns:
            (result_dict, current_page_last, violations)
        """
        records = self._flatten_records(page.get("records", []))

        # Build frequency map by role
        freq_map: Dict[str, Dict[str, int]] = {}
        for freq in freq_list:
            role = freq["role"]
            freq_map[role] = {
                "target": freq["target_minutes"],
                "tolerance": freq["tolerance_minutes"]
            }

        # Resolve the applicable frequency for this page's role
        # Priority: exact role match > GENERIC fallback > first available
        applicable_freq = None
        if page_role in freq_map:
            applicable_freq = freq_map[page_role]
        elif "GENERIC" in freq_map:
            applicable_freq = freq_map["GENERIC"]
        elif freq_map:
            applicable_freq = next(iter(freq_map.values()))

        violations = []

        if not applicable_freq:
            # No frequency to validate against
            result = {
                "page_no": page_no,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": 0
            }
            return result, None, violations

        target = applicable_freq["target"]
        tolerance = applicable_freq["tolerance"]
        low = target - tolerance
        high = target + tolerance

        # --- Cross-page boundary check ---
        if prev_page_last and records:
            prev_role, prev_ts = prev_page_last
            first_rec = records[0]
            date_f = first_rec.get("Date") or first_rec.get("Sample Date") or first_rec.get("date") or ""
            time_f = first_rec.get("Time") or first_rec.get("time") or ""
            ts_f = self._parse_timestamp(date_f, time_f)

            if ts_f:
                gap_minutes = (ts_f - prev_ts).total_seconds() / 60
                if gap_minutes < low or gap_minutes > high:
                    violations.append({
                        "role": page_role,
                        "pair_index": "cross-page",
                        "timestamps": [
                            prev_ts.strftime("%Y-%m-%d %H:%M"),
                            ts_f.strftime("%Y-%m-%d %H:%M")
                        ],
                        "gap_minutes": round(gap_minutes, 2),
                        "allowed_window_minutes": [low, high]
                    })

        # --- Intra-page adjacent pair checks ---
        for i in range(len(records) - 1):
            rec1 = records[i]
            rec2 = records[i + 1]

            # Parse timestamps for both records
            date1 = rec1.get("Date") or rec1.get("Sample Date") or rec1.get("date") or ""
            time1 = rec1.get("Time") or rec1.get("time") or ""
            date2 = rec2.get("Date") or rec2.get("Sample Date") or rec2.get("date") or ""
            time2 = rec2.get("Time") or rec2.get("time") or ""

            ts1 = self._parse_timestamp(date1, time1)
            ts2 = self._parse_timestamp(date2, time2)

            # If either timestamp invalid, skip this pair
            if not ts1 or not ts2:
                continue

            # Compute gap in minutes
            gap_minutes = (ts2 - ts1).total_seconds() / 60

            if gap_minutes < low or gap_minutes > high:
                violations.append({
                    "role": page_role,
                    "pair_index": i,
                    "timestamps": [
                        ts1.strftime("%Y-%m-%d %H:%M"),
                        ts2.strftime("%Y-%m-%d %H:%M")
                    ],
                    "gap_minutes": round(gap_minutes, 2),
                    "allowed_window_minutes": [low, high]
                })

        # --- Compute last valid record for cross-page carry-over ---
        current_page_last: Optional[Tuple[str, datetime]] = None
        for rec in reversed(records):
            date_str = rec.get("Date") or rec.get("Sample Date") or rec.get("date") or ""
            time_str = rec.get("Time") or rec.get("time") or ""
            ts = self._parse_timestamp(date_str, time_str)
            if ts:
                current_page_last = (page_role, ts)
                break

        if violations:
            result = {
                "page_no": page_no,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": 1
            }
        else:
            result = {
                "page_no": page_no,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": 0
            }

        return result, current_page_last, violations

    # ===================== VALIDATION =====================

    def run_validation(self, bmr_pages: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Main entry point for process frequency validation.

        Pure Python frequency validation:
        1. Identify explicit frequency pages and continuation pages using section-based state
        2. Extract frequency info (target, tolerance, role)
        3. Parse timestamps, group by role, validate gaps
        4. Return anomaly results per page

        Args:
            bmr_pages: Dictionary of page data keyed by page number
                       Expected format: {"1": {page_data}, "2": {page_data}, ...}
                       Each page should have: markdown_page, records, rules_or_instructions

        Returns:
            (results, return_debug)
            - results: List of result dicts with keys: page_no, section_name, check_name, anomaly_status
            - return_debug: List of detailed debug dicts per page
        """
        # Classify pages using section-based state
        is_freq_page, freq_info, defines_own_freq, page_roles = self._compute_frequency_pages(bmr_pages)

        # Sort pages
        keys = list(bmr_pages.keys())
        try:
            keys.sort(key=lambda k: int(k))
        except Exception:
            keys.sort()

        results = []
        return_debug = []
        # Track the last valid record from the previous frequency page
        # for cross-page boundary checking
        prev_page_last: Optional[Tuple[str, datetime]] = None

        for page_no in keys:
            if not is_freq_page.get(page_no, False):
                # Not a frequency page - reset cross-page state
                prev_page_last = None
                results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })
                return_debug.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                    "skip_reason": "Not a frequency page"
                })
                continue

            # Validate frequency
            page = bmr_pages[page_no]
            freq_list = freq_info.get(page_no, [])

            if not freq_list:
                prev_page_last = None
                results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })
                return_debug.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                    "skip_reason": "Frequency page but no frequency info available"
                })
                continue

            # If this page defines its own frequency, it starts a new
            # frequency context — do NOT carry over from the previous page
            is_new_section = defines_own_freq.get(page_no, False)
            if is_new_section:
                prev_page_last = None

            page_role = page_roles.get(page_no, "GENERIC")

            result, current_page_last, violations = self._validate_page_frequency(
                page_no, page, freq_list, page_role, prev_page_last
            )
            prev_page_last = current_page_last
            results.append(result)

            # Build debug entry
            freq_rules = [f"{f['raw_text']} (rule_role={f['role']})" for f in freq_list]
            debug_entry = {
                "page_no": page_no,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": result["anomaly_status"],
                "page_role": page_role,
                "frequency_rules": freq_rules,
                "defines_own_frequency": is_new_section,
                "num_records": len(self._flatten_records(page.get("records", []))),
            }
            if violations:
                debug_entry["violations"] = violations
            else:
                debug_entry["skip_reason"] = "No frequency violations found"
            return_debug.append(debug_entry)

        return results, return_debug


# =====================================================
# ---------------- HELPER FUNCTION --------------------
# =====================================================
def load_json_file(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =====================================================
# ---------------- MAIN RUNNER ------------------------
# =====================================================
def print_debug_report(return_debug: List[Dict[str, Any]]) -> None:
    """Print a nicely formatted debug report."""
    separator = "=" * 80
    thin_sep = "-" * 80

    # Split into relevant (frequency) pages vs skipped
    freq_pages = [d for d in return_debug if d.get("skip_reason") != "Not a frequency page"]
    skipped_pages = [d for d in return_debug if d.get("skip_reason") == "Not a frequency page"]
    anomaly_pages = [d for d in return_debug if d.get("anomaly_status") == 1]

    print(f"\n{separator}")
    print(f"  CHECK 46 — PROCESS FREQUENCY VALIDATION REPORT")
    print(f"{separator}")
    print(f"  Total pages processed : {len(return_debug)}")
    print(f"  Frequency pages       : {len(freq_pages)}")
    print(f"  Skipped (non-freq)    : {len(skipped_pages)}")
    print(f"  Anomalies detected    : {len(anomaly_pages)}")
    print(f"{separator}\n")

    if not freq_pages:
        print("  No frequency pages detected in this BMR.\n")
        return

    # Detail for each frequency page
    for entry in freq_pages:
        page_no = entry["page_no"]
        status = "❌ ANOMALY" if entry["anomaly_status"] == 1 else "✅ PASS"
        defines_own = entry.get("defines_own_frequency", False)
        page_type = "DEFINES FREQUENCY" if defines_own else "CONTINUATION"
        num_records = entry.get("num_records", "?")
        page_role = entry.get("page_role", "GENERIC")

        print(f"  Page {page_no:>4s}  |  {status}  |  {page_type}  |  Role: {page_role}  |  Records: {num_records}")

        # Frequency rules
        rules = entry.get("frequency_rules", [])
        if rules:
            for rule in rules:
                print(f"           Rule: {rule}")

        # Violations
        violations = entry.get("violations", [])
        if violations:
            for v in violations:
                pair_idx = v.get("pair_index", "?")
                role = v.get("role", "?")
                ts = v.get("timestamps", ["?", "?"])
                gap = v.get("gap_minutes", "?")
                window = v.get("allowed_window_minutes", ["?", "?"])
                label = "CROSS-PAGE" if pair_idx == "cross-page" else f"pair {pair_idx}"
                print(f"           ⚠ Violation ({label}, role={role}): "
                      f"{ts[0]} → {ts[1]}  gap={gap} min  "
                      f"(allowed {window[0]}–{window[1]} min)")
        elif entry.get("skip_reason"):
            print(f"           {entry['skip_reason']}")

        print(f"  {thin_sep}")

    # Summary of anomalies
    if anomaly_pages:
        print(f"\n  ANOMALY SUMMARY:")
        for entry in anomaly_pages:
            print(f"    • Page {entry['page_no']}: "
                  f"{len(entry.get('violations', []))} violation(s)")
    else:
        print(f"\n  ✅ No anomalies detected across all frequency pages.")
    print()


if __name__ == "__main__":
    # Load the BMR JSON file
    ocr_json_path = "/home/softsensor/Desktop/Amneal/all_result_76_20feb.json"
    
    print("Running Check 46 - Process Frequency Check...")
    
    # Load the full JSON
    full_data = load_json_file(ocr_json_path)
    
    # Extract filled_master_json from steps and convert to expected format
    steps = full_data.get("steps", {})
    filled_master_json = steps.get("filled_master_json", [])
    
    bmr_pages = {}
    for page_obj in filled_master_json:
        page_no = str(page_obj.get("page", ""))
        if page_no:
            page_data = {
                "page": page_obj.get("page"),
                "markdown_page": page_obj.get("markdown_page", ""),
                "additional_keys": page_obj.get("additional_keys", []),
                "rules_or_instructions": [],
                "records": []
            }
            
            for content in page_obj.get("page_content", []):
                content_type = content.get("type", "")
                
                if content_type == "table":
                    table_json = content.get("table_json", {})
                    if "records" in table_json:
                        page_data["records"].extend(table_json["records"])
                    else:
                        if not page_data.get("table_data"):
                            page_data["table_data"] = []
                        page_data["table_data"].append(table_json)
                
                elif content_type == "kv_text_block":
                    kv_data = content.get("extracted_kv_text_block", {})
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
    
    print(f"Loaded {len(bmr_pages)} pages from filled_master_json")
    
    validator = ProcessFrequencyValidator()
    results, return_debug = validator.run_validation(bmr_pages)

    # Print formatted debug report
    print_debug_report(return_debug)

    # Print simplified results JSON
    print("\n" + "=" * 80)
    print("  SIMPLIFIED RESULTS (JSON)")
    print("=" * 80)
    print(json.dumps(results, indent=2))
