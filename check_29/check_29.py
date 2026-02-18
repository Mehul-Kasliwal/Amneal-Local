import json
import re
from typing import Dict, List, Any, Optional
from pprint import pprint as pp


class ARNumberValidator:
    """
    Check 29. AI tool should be able to verify the AR. No. form reference attached documents and actual issued quantity should be verified by standard quantity.
    ---
    Description:
    1. Find BMR pages with dispensing tables (Material Code, A.R. No.)
    2. Match against Attachment 3 records by Material Code
    3. Verify A.R. No. matches between BMR and attachment
    ---
    Type: Python-based
    ---
    Attachment Required: YES
    ---
    Author: Mehul Kasliwal (19 Jan 2026)
    """

    # ===================== CONSTANTS =====================

    SECTION_NAME = "AR Number Verification"
    CHECK_NAME = "Cross-Validation Check"

    # Column name synonyms for BMR tables
    MATERIAL_CODE_SYNONYMS = [
        "Material code",
        "Material Code",
        "Material code.",
    ]

    AR_NO_SYNONYMS = [
        "A.R. No.",
        "A.R. No",
        "AR No.",
        "AR No",
        "AR. No.",
        "AR. No",
    ]

    # Values considered as blank or NA
    BLANK_VALUES = {"", "na", "na.", "n/a", "n/a.", "n.a", "n.a.", "*", "-"}

    # ===================== HELPER METHODS =====================

    def is_blank_or_na(self, v: Any) -> bool:
        """Check if a value is blank or represents N/A."""
        if v is None:
            return True
        if not isinstance(v, str):
            return False
        return v.strip().lower() in self.BLANK_VALUES



    def normalize_ar_no(self, ar_no: str) -> List[str]:
        """
        Normalize A.R. No. handling multiple values and OCR spacing issues.
        First removes all spaces to handle cases like "230240 82" -> "23024082"
        Then splits by commas/plus for multiple AR numbers.
        Returns a list of normalized AR numbers.
        """
        if not isinstance(ar_no, str) or self.is_blank_or_na(ar_no):
            return []
        
        # First, remove all spaces to handle OCR issues like "230240 82" -> "23024082"
        ar_no_cleaned = re.sub(r'\s+', '', ar_no.strip())
        
        # Split by common separators (comma, plus) but not space since we removed those
        parts = re.split(r'[,+]+', ar_no_cleaned)
        result = []
        for p in parts:
            # Extract alphanumeric characters only
            cleaned = re.sub(r'[^\dA-Za-z]', '', p)
            if cleaned and len(cleaned) >= 5:  # AR numbers are typically 8 digits
                result.append(cleaned.upper())
        return result

    def find_col(self, records: List[Dict], synonyms: List[str]) -> Optional[str]:
        """Find which column name variant exists in the records."""
        for syn in synonyms:
            if any(isinstance(r, dict) and syn in r for r in records):
                return syn
        return None

    def iter_dict_rows(self, records: List) -> List[Dict]:
        """Flatten nested record structures, yielding individual dict rows."""
        result = []
        for r in records:
            if isinstance(r, dict):
                result.append(r)
            elif isinstance(r, list):
                result.extend(self.iter_dict_rows(r))
        return result

    # ===================== TABLE DETECTION =====================

    def has_ar_table(self, records: List[Dict]) -> bool:
        """Check if records represent a table with Material Code and A.R. No. columns."""
        flat = self.iter_dict_rows(records)
        if not flat:
            return False

        has_material = self.find_col(flat, self.MATERIAL_CODE_SYNONYMS) is not None
        has_ar = self.find_col(flat, self.AR_NO_SYNONYMS) is not None

        return has_material and has_ar

    def extract_bmr_records(self, page_content: List) -> List[Dict]:
        """Extract dispensing records from BMR page content."""
        all_records = []

        def search_records(obj):
            if isinstance(obj, dict):
                if "records" in obj and isinstance(obj["records"], list):
                    all_records.extend(self.iter_dict_rows(obj["records"]))
                if "table_json" in obj and isinstance(obj["table_json"], dict):
                    if "records" in obj["table_json"]:
                        all_records.extend(self.iter_dict_rows(obj["table_json"]["records"]))
                for v in obj.values():
                    search_records(v)
            elif isinstance(obj, list):
                for item in obj:
                    search_records(item)

        search_records(page_content)
        return all_records

    # ===================== ATTACHMENT LOADING =====================

    def load_attachment_data(self, attachment_data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Load attachment data and index by normalized Material Code.
        Returns: {material_code: [list of records with that code]}
        """
        index = {}

        for page in attachment_data:
            page_content = page.get("page_content", [])
            if isinstance(page_content, dict):
                page_content = [page_content]

            def process_record(record: Dict):
                """Process a single record to extract material code and add to index."""
                code = None
                # Try different key names for Material Code
                for key in ["Material Code", "Material code", "Material code."]:
                    if key in record:
                        code = record.get(key, "").strip().upper() if isinstance(record.get(key, ""), str) else ""
                        break
                
                if code:
                    if code not in index:
                        index[code] = []
                    index[code].append(record)

            for content in page_content:
                if isinstance(content, dict):
                    # Check if it's a single record (has Material Code directly)
                    has_material = any(key in content for key in ["Material Code", "Material code", "Material code."])
                    if has_material:
                        process_record(content)

                    # Check for nested records
                    if "records" in content:
                        for record in self.iter_dict_rows(content["records"]):
                            process_record(record)
                    
                    # Check for section_title with records
                    if "section_title" in content and "records" in content:
                        for record in self.iter_dict_rows(content["records"]):
                            process_record(record)

        return index

    # ===================== VALIDATION =====================

    def validate_row(
        self,
        bmr_row: Dict,
        attachment_records: List[Dict],
        material_col: str,
        ar_col: str,
    ) -> List[Dict]:
        """Validate a single BMR row against attachment records."""
        anomalies = []

        raw_code = bmr_row.get(material_col, "")
        material_code = raw_code.strip().upper() if isinstance(raw_code, str) else ""
        if not material_code:
            return anomalies

        bmr_ar = bmr_row.get(ar_col, "")
        bmr_ar_list = self.normalize_ar_no(bmr_ar)

        # Skip if no valid AR number in BMR
        if not bmr_ar_list:
            return anomalies

        # Get attachment data for this material
        if not attachment_records:
            # Material not found in attachment - could be an issue
            return anomalies

        # Aggregate all AR numbers from attachment
        attachment_ar_set = set()

        for att_record in attachment_records:
            # Get AR number from attachment - try different key names
            att_ar = ""
            for key in ["AR. No.", "AR No.", "A.R. No.", "AR No", "A.R. No"]:
                if key in att_record:
                    att_ar = att_record.get(key, "")
                    break
            
            for ar in self.normalize_ar_no(att_ar):
                attachment_ar_set.add(ar)

        # Check: A.R. No. Match
        if bmr_ar_list and attachment_ar_set:
            ar_matches = any(ar in attachment_ar_set for ar in bmr_ar_list)
            if not ar_matches:
                anomalies.append({
                    "parameter": f"A.R. No. Mismatch — {material_code}",
                    "issue": "A.R. number in BMR does not match attachment",
                    "bmr_value": bmr_ar,
                    "attachment_value": ", ".join(sorted(attachment_ar_set))
                })

        return anomalies

    # ===================== ENTRY POINT =====================

    def run_validation(
        self,
        bmr_data: List[Dict],
        attachment_data: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Main entry point for AR number validation.

        Args:
            bmr_data: List of page dicts from BMR JSON
            attachment_data: List of page dicts from Attachment 3 JSON

        Returns:
            List of result dicts, one per page with anomalies
        """
        results = []
        return_debug = []  # Detailed debug information

        # Load and index attachment data
        attachment_index = self.load_attachment_data(attachment_data)

        for page_dict in bmr_data:
            page_no = str(page_dict.get("page", "unknown"))
            page_content = page_dict.get("page_content", [])

            try:
                # Extract records from this page
                records = self.extract_bmr_records(page_content)
                if not records:
                    # Debug info
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": "No records found on page"
                    })
                    # Track pages with no records
                    results.append({
                        "page": int(page_no),
                        "section_name": "Quality Control",
                        "check_name": "AR_NO_Match_Structured",
                        "anomaly_status": 0
                    })
                    continue

                # Check if this is a table with Material Code and A.R. No.
                if not self.has_ar_table(records):
                    # Debug info
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": "Not an AR table (missing required columns)"
                    })
                    # Track pages without AR table
                    results.append({
                        "page": int(page_no),
                        "section_name": "Quality Control",
                        "check_name": "AR_NO_Match_Structured",
                        "anomaly_status": 0
                    })
                    continue

                # Find column names
                material_col = self.find_col(records, self.MATERIAL_CODE_SYNONYMS)
                ar_col = self.find_col(records, self.AR_NO_SYNONYMS)

                if not all([material_col, ar_col]):
                    # Debug info
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": f"Incomplete columns (material_col={material_col}, ar_col={ar_col})"
                    })
                    # Track pages with incomplete columns
                    results.append({
                        "page": int(page_no),
                        "section_name": "Quality Control",
                        "check_name": "AR_NO_Match_Structured",
                        "anomaly_status": 0
                    })
                    continue

                # Validate each row - track processed material codes to avoid duplicates
                page_anomalies = []
                processed_materials = set()  # Track material codes we've already validated
                
                for row in records:
                    raw_code = row.get(material_col, "")
                    material_code = raw_code.strip().upper() if isinstance(raw_code, str) else ""
                    
                    # Skip if we've already processed this material code on this page
                    if material_code in processed_materials or not material_code:
                        continue
                    
                    processed_materials.add(material_code)
                    attachment_records = attachment_index.get(material_code, [])

                    row_anomalies = self.validate_row(
                        row, attachment_records,
                        material_col, ar_col
                    )
                    page_anomalies.extend(row_anomalies)

                # Record result for this page
                if page_anomalies:
                    # Debug info
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 1,
                        "anomalies": page_anomalies
                    })
                    # Simplified output
                    results.append({
                        "page": int(page_no),
                        "section_name": "Quality Control",
                        "check_name": "AR_NO_Match_Structured",
                        "anomaly_status": 1
                    })
                else:
                    # Debug info
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": "No anomalies found"
                    })
                    # Simplified output
                    results.append({
                        "page": int(page_no),
                        "section_name": "Quality Control",
                        "check_name": "AR_NO_Match_Structured",
                        "anomaly_status": 0
                    })

            except Exception as e:
                # Debug info
                return_debug.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                    "error": str(e)
                })
                # Even for errors, return in the same simplified format
                results.append({
                    "page": int(page_no),
                    "section_name": "Quality Control",
                    "check_name": "AR_NO_Match_Structured",
                    "anomaly_status": 0
                })

        # Print debug information
        print("\n" + "="*60)
        print("DEBUG INFORMATION (return_debug)")
        print("="*60)
        for debug_entry in return_debug:
            pp(debug_entry)
            print("-"*40)
        print("="*60 + "\n")

        return results


# =====================================================
# ---------------- MAIN RUNNER ------------------------
# =====================================================
if __name__ == "__main__":
    import os

    # File paths
    bmr_filepath = '61_FILLED_MASTER_15_JAN.json'
    attachment_filepath = 'attachments_3_azure_di_ocr_bmr_61 1.json'

    # Load data
    with open(bmr_filepath, "r", encoding="utf-8") as f:
        bmr_data = json.load(f)

    with open(attachment_filepath, "r", encoding="utf-8") as f:
        attachment_data = json.load(f)

    # Run validation
    validator = ARNumberValidator()
    results = validator.run_validation(bmr_data, attachment_data)

    # Output as JSON (results are already in the final format)
    print(json.dumps(results, indent=2))
