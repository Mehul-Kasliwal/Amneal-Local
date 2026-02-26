"""
Check 20 — Actual vs Observed Plus/Minus Range Checker (Optimized — Pure Python)

Extracts parameters with ± tolerance specs from BMR pages and validates
whether observed values fall within the [Target - Tolerance, Target + Tolerance] range.

Optimization:
    - NO LLM calls required. All extraction and computation done via Python regex.
    - Pre-filters pages to skip those without ± patterns.
    - Recursively collects text from structured table_json and kv_text_block.

Type: Pure Python (deterministic)
Author: Mehul Kasliwal (19th February 2026)

Usage:
    python check_20.py <path_to_bmr_json>
"""

import sys
import json
import re
from typing import List, Dict, Any, Tuple, Optional


# ─── Inlined config ──────────────────────────────────────────────────────────
CHECK_20_CONFIG = {
    "name": "Actual_observed_value_falls_outside_the_standard_plus/minus_range.",
    "process": "Process Parameters",
}

def get_check_name(check_id: str) -> str:
    return CHECK_20_CONFIG["name"]

def get_check_process(check_id: str) -> str:
    return CHECK_20_CONFIG["process"]


# ─── Regex patterns ──────────────────────────────────────────────────────────

# Pattern to match ± or +/- spec anywhere in text (quick pre-filter)
PM_QUICK_RE = re.compile(r'±|\+/?-')

# Pattern to match: "<param_name>: <observed_value> <optional_unit>.*?(Limit: <target> ± <tolerance> <optional_unit>)"
# Examples:
#   "Stirrer speed: 300 RPM (Limit: 200±50 RPM )"
#   "pH of Solution: 7.63. (Limit: 10 ± 1)"
#   "Temperature of Solution: 22.00. (Limit: 25°C±5°C)"
PARAM_PM_RE = re.compile(
    r'([A-Za-z][A-Za-z\s/().]{2,60}?)'   # group 1: parameter name (starts with letter)
    r'(?::\s*|\s+)'                        # optional colon separator or space
    r'(\d+\.?\d*)'                         # group 2: observed value (number)
    r'[^(]*?'                              # skip units/text until "("
    r'\(Limit(?:[:\s]*)\s*'                # "(Limit:" or "(Limit " prefix
    r'(\d+\.?\d*)'                         # group 3: target value
    r'\s*(?:°?[A-Za-z%]*\s*)?'             # optional unit between target and ±
    r'(?:±|\+/?-)\s*'                      # ± or +/- symbol
    r'(\d+\.?\d*)'                         # group 4: tolerance value
    r'\s*(?:[^)]*?)'                       # optional trailing unit/text
    r'\)',                                  # closing paren
    re.IGNORECASE
)


# ─── Main class ──────────────────────────────────────────────────────────────
class ActVsObsPlusMinusChecker:
    """
    Check 20: Actual vs Observed ± Range Checker
    ---
    Description:
    1. Pre-filter pages: skip any page whose text contains no ± or +/- symbols.
    2. Recursively collect all text from page_content (table_json, kv_text_block, markdown).
    3. Regex-extract parameter name, observed value, target, and tolerance.
    4. Compute [target - tolerance, target + tolerance] range in Python.
    5. Flag anomaly if observed value falls outside the inclusive range.
    ---
    Type: Pure Python (deterministic, no LLM)
    ---
    Attachment Required: NO
    ---
    Author: Mehul Kasliwal (19th February 2026)
    """

    SECTION_NAME = get_check_process("check_20")
    CHECK_NAME = get_check_name("check_20")

    def __init__(self):
        self.section_name = self.SECTION_NAME
        self.check_name = self.CHECK_NAME
        self.check_id = "check_20"

    # ─── Text collection ─────────────────────────────────────────────────

    def _deep_collect_text(self, obj: Any) -> List[str]:
        """
        Recursively extract all string values from nested dicts/lists.
        This captures text from table_json records, kv_text_block, etc.
        """
        texts = []
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                texts.extend(self._deep_collect_text(v))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(self._deep_collect_text(item))
        return texts

    def _collect_page_texts(self, page: Dict[str, Any]) -> List[str]:
        """
        Collect all text strings from a page's page_content and markdown_page.
        Returns a list of individual text fragments.
        """
        texts = []

        # Structured data from page_content
        page_content = page.get("page_content", [])
        if page_content:
            texts.extend(self._deep_collect_text(page_content))

        # Also check markdown_page as a fallback
        markdown = page.get("markdown_page", "")
        if markdown:
            texts.append(markdown)

        return texts

    # ─── Pre-filter ──────────────────────────────────────────────────────

    def _page_has_pm(self, texts: List[str]) -> bool:
        """Quick check: does any text fragment contain ± or +/-?"""
        for t in texts:
            if PM_QUICK_RE.search(t):
                return True
        return False

    # ─── Extraction ──────────────────────────────────────────────────────

    def _extract_pm_entries(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Apply regex to all text fragments and extract ± parameter entries.
        Returns list of dicts with: parameter, observed_value, target, tolerance, standard_range.
        """
        entries = []
        seen = set()  # avoid duplicates from same data appearing in multiple fields

        for text in texts:
            for match in PARAM_PM_RE.finditer(text):
                param_name = match.group(1).strip().rstrip('.')
                observed_str = match.group(2)
                target_str = match.group(3)
                tolerance_str = match.group(4)

                # De-duplicate by (param, observed, target, tolerance)
                key = (param_name.lower(), observed_str, target_str, tolerance_str)
                if key in seen:
                    continue
                seen.add(key)

                try:
                    observed = float(observed_str)
                    target = float(target_str)
                    tolerance = float(tolerance_str)
                except ValueError:
                    continue

                entries.append({
                    "parameter": param_name,
                    "observed_value": observed,
                    "target": target,
                    "tolerance": tolerance,
                    "standard_range": f"{target_str} ± {tolerance_str}",
                })

        return entries

    # ─── Range check ─────────────────────────────────────────────────────

    def _check_range(self, observed: float, target: float, tolerance: float) -> Tuple[int, float, float]:
        """
        Check if observed value is within [target - tolerance, target + tolerance].
        Returns: (anomaly_status, lower_bound, upper_bound)
        """
        lower = target - tolerance
        upper = target + tolerance
        if lower <= observed <= upper:
            return 0, lower, upper
        else:
            return 1, lower, upper

    # ─── Main entry point ────────────────────────────────────────────────

    def run(self, document_pages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run ± range checks across all pages.

        Args:
            document_pages: List of page dicts with 'page', 'page_content', 'markdown_page'.

        Returns:
            (results, return_debug) tuple where:
            - results: simplified output list with page_no, section_name, check_name, anomaly_status
            - return_debug: detailed debug information for every page and every ± entry found
        """
        results = []
        return_debug = []

        for page in document_pages:
            page_no = page.get("page")

            try:
                # Step 1: Collect all text from the page
                texts = self._collect_page_texts(page)

                # Step 2: Pre-filter — skip pages without ±
                if not self._page_has_pm(texts):
                    results.append({
                        "page_no": page_no,
                        "section_name": self.section_name,
                        "check_name": self.check_name,
                        "anomaly_status": 0,
                    })
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.section_name,
                        "check_name": self.check_name,
                        "anomaly_status": 0,
                        "skip_reason": "No ± or +/- pattern found on page",
                        "entries_found": 0,
                    })
                    continue

                # Step 3: Extract ± entries via regex
                entries = self._extract_pm_entries(texts)

                if not entries:
                    # Page has ± text but no extractable param pattern
                    # (e.g., ± appears in instructions/frequency, not in a Limit spec)
                    results.append({
                        "page_no": page_no,
                        "section_name": self.section_name,
                        "check_name": self.check_name,
                        "anomaly_status": 0,
                    })
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.section_name,
                        "check_name": self.check_name,
                        "anomaly_status": 0,
                        "skip_reason": "± symbol found but no extractable Limit spec with observed value",
                        "entries_found": 0,
                    })
                    continue

                # Step 4: Check each entry and determine anomaly
                page_anomaly = 0
                entry_details = []

                for entry in entries:
                    anomaly_status, lower, upper = self._check_range(
                        entry["observed_value"],
                        entry["target"],
                        entry["tolerance"],
                    )

                    if anomaly_status == 1:
                        page_anomaly = 1

                    entry_detail = {
                        "parameter": entry["parameter"],
                        "observed_value": entry["observed_value"],
                        "standard_range": entry["standard_range"],
                        "target": entry["target"],
                        "tolerance": entry["tolerance"],
                        "computed_lower_bound": lower,
                        "computed_upper_bound": upper,
                        "within_range": anomaly_status == 0,
                        "anomaly_status": anomaly_status,
                    }

                    if anomaly_status == 1:
                        # Add clear explanation for anomaly
                        if entry["observed_value"] < lower:
                            entry_detail["anomaly_detail"] = (
                                f"BELOW RANGE: {entry['observed_value']} < {lower} "
                                f"(target {entry['target']} - tolerance {entry['tolerance']})"
                            )
                        else:
                            entry_detail["anomaly_detail"] = (
                                f"ABOVE RANGE: {entry['observed_value']} > {upper} "
                                f"(target {entry['target']} + tolerance {entry['tolerance']})"
                            )

                    entry_details.append(entry_detail)

                results.append({
                    "page_no": page_no,
                    "section_name": self.section_name,
                    "check_name": self.check_name,
                    "anomaly_status": page_anomaly,
                })
                return_debug.append({
                    "page_no": page_no,
                    "section_name": self.section_name,
                    "check_name": self.check_name,
                    "anomaly_status": page_anomaly,
                    "entries_found": len(entry_details),
                    "entries": entry_details,
                })

            except Exception as e:
                results.append({
                    "page_no": page_no,
                    "section_name": self.section_name,
                    "check_name": self.check_name,
                    "anomaly_status": 0,
                })
                return_debug.append({
                    "page_no": page_no,
                    "section_name": self.section_name,
                    "check_name": self.check_name,
                    "anomaly_status": 0,
                    "error": str(e),
                })

        return results, return_debug


# ─── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # json_path = "/home/softsensor/Desktop/Amneal/challenge_bmr/05jan_AH250076_50Checks 1.json"
    # json_path = "/home/softsensor/Desktop/Amneal/complete_data_61 1.json"
    json_path = "/home/softsensor/Desktop/Amneal/New_BMRs/Emulsion_line_AH240074_filled_master_data.json"

    print("Running Check 20 - Actual vs Observed ± Range (Pure Python)...")

    try:
        # This JSON already contains the filled_master_json directly (list of page dicts)
        with open(json_path, "r", encoding="utf-8") as f:
            document_pages = json.load(f)
        print(f"Loaded {len(document_pages)} pages\n")

        checker = ActVsObsPlusMinusChecker()
        results, return_debug = checker.run(document_pages)

        # ── Summary ──
        print("=" * 60)
        print("CHECK 20 RESULTS — Actual vs Observed ± Range")
        print("=" * 60)
        for r in results:
            flag = "⚠ ANOMALY" if r["anomaly_status"] == 1 else "✓ OK"
            print(f"  Page {str(r['page_no']):>3s}: {flag}")

        total = len(results)
        anomaly_count = sum(1 for r in results if r["anomaly_status"] == 1)
        print(f"\nTotal pages checked: {total}")
        print(f"Anomalies found: {anomaly_count}")

        # ── Detailed debug for pages with entries ──
        print("\n" + "=" * 60)
        print("DEBUG — Detailed ± Entry Analysis")
        print("=" * 60)
        for d in return_debug:
            entries = d.get("entries", [])
            if entries:
                print(f"\n  Page {d['page_no']} — {d['entries_found']} ± entries found:")
                for i, e in enumerate(entries, 1):
                    status_icon = "❌ ANOMALY" if e["anomaly_status"] == 1 else "✅ OK"
                    print(f"    [{i}] {e['parameter']}")
                    print(f"        Observed: {e['observed_value']}")
                    print(f"        Spec:     {e['standard_range']}")
                    print(f"        Range:    [{e['computed_lower_bound']}, {e['computed_upper_bound']}]")
                    print(f"        Status:   {status_icon}")
                    if "anomaly_detail" in e:
                        print(f"        Detail:   {e['anomaly_detail']}")

        # ── Full JSON dump ──
        print("\n--- Full results JSON ---")
        print(json.dumps(results, indent=2))

        print("\n--- Full debug JSON ---")
        print(json.dumps(return_debug, indent=2, default=str))

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
