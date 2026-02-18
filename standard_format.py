import json, re
from typing import Dict, List, Any, Optional
from pprint import pprint as pp

class Autoclave_LoadRange_Validator:
    """
    Check 33 :- AI tool should be able to check the autoclave validated load range and verified the actual load against the validated minimum and maximum load, if any discrepancy observed should be highlighted.
    ---
    Description :
    Validates autoclave table load : min, max and others are in between and multiples of minimum units
    ----
    type : Pyhton bases
    ----
    Attachment Required : NO
    ----
    Author : Anil (30 Dec 2025)
    """

    # ===================== CONSTANTS =====================
    KEY_PHRASE = "CIP, SIP and Autoclave details"

    REQUIRED_KEYS = {
        'qty_total': "Total No. of bags",
        'qty_per_bag': "Qty. per bag",
        'qty_rs': "Qty. of RS to be sterilized"
    }

    LOAD_CHECK_FIELD = "Load Checked By (SM)"

    SECTION_NAME = "Process Execution"
    CHECK_NAME = "Load Range validation in Autocalve report"

    # ===================== HELPERS =====================

    def find_keyphrase_in_rules(self, obj: Any) -> bool:
        try:
            if isinstance(obj, dict):
                rules = obj.get("rules_or_instructions", [])
                if isinstance(rules, list):
                    for rule in rules:
                        if isinstance(rule, str) and self.KEY_PHRASE.lower() in rule.lower():
                            return True
                return any(self.find_keyphrase_in_rules(v) for v in obj.values())

            if isinstance(obj, list):
                return any(self.find_keyphrase_in_rules(i) for i in obj)
        except Exception:
            return False

        return False

    def recursive_search(self, obj: Any) -> bool:
        try:
            if isinstance(obj, dict):
                records = obj.get("records")
                if isinstance(records, list):
                    for rec in records:
                        if isinstance(rec, dict) and set(self.REQUIRED_KEYS.values()).issubset(rec):
                            return True
                return any(self.recursive_search(v) for v in obj.values())

            if isinstance(obj, list):
                return any(self.recursive_search(i) for i in obj)
        except Exception:
            return False

        return False

    def fetch_required_records(self, page_content: List[Dict], page_no: int) -> List[Dict]:
        extracted = []

        try:
            for block in page_content:
                if block.get("type") != "table":
                    continue

                records = block.get("table_json", {}).get("records", [])
                for record in records:
                    if (
                        isinstance(record, dict)
                        and set(self.REQUIRED_KEYS.values()).issubset(record)
                    ):
                        extracted.append({
                            "page_no": page_no,
                            "record": record
                        })
        except Exception:
            pass

        return extracted

    def extract_min_max(self, records_with_page: List[Dict]) -> Dict[str, Dict[str, int]]:
        min_vals = {k: float("inf") for k in self.REQUIRED_KEYS.values()}
        max_vals = {k: float("-inf") for k in self.REQUIRED_KEYS.values()}

        for item in records_with_page:
            try:
                rec = item["record"]

                if (
                    rec.get(self.REQUIRED_KEYS['qty_per_bag']) != 'NA'
                    and rec.get(self.LOAD_CHECK_FIELD) == "NA"
                ):
                    for key in self.REQUIRED_KEYS.values():
                        val = int(rec[key])
                        min_vals[key] = min(min_vals[key], val)
                        max_vals[key] = max(max_vals[key], val)
            except Exception:
                continue

        return {
            k: {"min": min_vals[k], "max": max_vals[k]}
            for k in self.REQUIRED_KEYS.values()
        }

    def validate_loaded_records(
        self,
        records_with_page: List[Dict],
        min_max: Dict[str, Dict[str, int]]
    ) -> List[Dict[str, Any]]:

        results = []

        for item in records_with_page:
            try:
                rec = item["record"]

                if rec.get(self.LOAD_CHECK_FIELD) == "NA":
                    continue

                total_bags = int(rec[self.REQUIRED_KEYS['qty_total']])
                qty_per_bag = int(rec[self.REQUIRED_KEYS['qty_per_bag']])
                qty_rs = int(rec[self.REQUIRED_KEYS['qty_rs']])

                valid = (
                    min_max[self.REQUIRED_KEYS['qty_total']]["min"] <= total_bags <=
                    min_max[self.REQUIRED_KEYS['qty_total']]["max"]
                    and min_max[self.REQUIRED_KEYS['qty_rs']]["min"] <= qty_rs <=
                    min_max[self.REQUIRED_KEYS['qty_rs']]["max"]
                    and qty_per_bag > 0
                    and qty_rs % qty_per_bag == 0
                )

                results.append({
                    "page_no": item["page_no"],
                    "valid": valid
                })

            except Exception:
                results.append({
                    "page_no": item.get("page_no"),
                    "valid": False
                })

        return results

    # ===================== ENTRY POINT =====================
    def run_validation(self, document_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []

        for idx, page in enumerate(document_pages, start=1):
            try:
                page_content = page.get("page_content", [])

                # -------- STEP 1: relevance check --------
                # cond1 = self.find_keyphrase_in_rules(page_content)
                cond2 = self.recursive_search(page_content)

                if not (cond2):
                    results.append({
                        "page_no": idx,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                ## DEBUGING ##


                # -------- STEP 2: extract records --------
                records_with_page = self.fetch_required_records(page_content, idx)

                # -------- STEP 3: min / max --------
                min_max = self.extract_min_max(records_with_page)

                # -------- STEP 4: validation --------
                validation_results = self.validate_loaded_records(
                    records_with_page, min_max
                )

                # -------- STEP 5: anomaly decision --------
                anomaly_status = 0
                for res in validation_results:
                    if res.get("valid") is False:
                        anomaly_status = 1
                        break

                results.append({
                    "page_no": idx,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": anomaly_status
                })

            except Exception:
                results.append({
                    "page_no": idx,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })

        return results


# =====================================================
# ---------------- MAIN RUNNER ------------------------
# =====================================================
if __name__ == "__main__":
    filepath = '../JSON/BMR_0074_FILLED_MASTER_latest.json'
    with open(filepath, "r", encoding="utf-8") as f:
        document_pages = json.load(f)
    obj = Autoclave_LoadRange_Validator()
    results = obj.run_validation(document_pages)