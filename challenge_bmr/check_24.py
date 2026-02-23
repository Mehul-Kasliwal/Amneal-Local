#### ==========================================================
# Check 24 : AI tool should be able to cross verify the A. R. No. with reference attachments.
# { "type" : "Python-based", "Attachment_Required" : "Yes ( Attachment-2,Attachment-3 )" , "Author" : "MD SABIULLAH,Mehul Kasliwal (22 Jan 2026)" }
#### ==========================================================

import re
import json
import os
from typing import Any, Dict, List, Optional
 
 
 
class Check24ARNoVerifyResults:
    """
    ----
    Check 24 : AI tool should be able to cross verify the A. R. No. with reference attachments.
    ----
    Description: Validates that A.R. No from BMR's "Potency Calculation for API"
    section matches the Amneal Receiving No in at least one page of the Certificate of Analysis (CoA) attachments.
    LENIENT: Passes if ANY page has the matching AR No (doesn't require all pages to be consistent).
    ----
    Type: Python-based
    ----
    Attachment Required: Yes ( Attachment-2 )
    ----
    Author: MD SABIULLAH (22 Jan 2026)
    """
    
    # =========================================================================
    # VALIDATION CONSTANTS
    # =========================================================================
    
    SECTION_NAME = "Quality Control"
    CHECK_NAME = "AR_NO_Verify_Results"
    
    # =========================================================================
    # BMR DETECTION KEYWORDS
    # =========================================================================
    
    BMR_KEY_PHRASES = [
        "POTENCY CALCULATION FOR API",
        "Potency Calculation for"
    ]
    
    # =========================================================================
    # BMR COLUMN KEYWORDS
    # =========================================================================
    
    BMR_RAW_MATERIAL_KEYS = ["raw material", "raw materials"]
    BMR_EXCLUDE_MATERIAL_KEYWORDS = ["recorded by", "checked by"]
    BMR_AR_NO_KEYS = ["arno", "a r no", "a.r.no", "ar no"]
    
    # =========================================================================
    # ATTACHMENT FILTERING
    # =========================================================================
    
    ATT_SKIP_SECTION_TITLES = ["RM Disposition Form"]
    ATT_COA_MARKER = "CERTIFICATE OF ANALYSIS"
    
    # =========================================================================
    # MARKDOWN REGEX PATTERNS
    # =========================================================================
    
    # Pattern to extract Amneal Receiving No from markdown
    AMNEAL_RECEIVING_NO_PATTERN = r'Amneal\s+Receiving\s+No\.?\s*:?\s*(?:</td>\s*<td[^>]*>)?([A-Z0-9]+)'
    
    # =========================================================================
    # INVALID VALUE TOKENS
    # =========================================================================
    
    INVALID_VALUE_TOKENS = {
        "na", "n/a", "n.a", "n a", "ma", "nan", "none",
        "null", "-", "_", "", "n"
    }
    
    # =========================================================================
    # NORMALIZATION UTILITIES
    # =========================================================================
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for fuzzy matching (remove spaces, special chars, lowercase)"""
        if not isinstance(text, str):
            return ""
        return re.sub(r"[^a-z0-9]+", "", text.lower())
    
    @classmethod
    def _is_invalid_value(cls, value: Any) -> bool:
        """Check if value is invalid (NA, null, etc.)"""
        if value is None:
            return True
        
        val = str(value).strip().lower()
        val_norm = re.sub(r"[^a-z0-9]+", "", val)
        
        return (
            val in cls.INVALID_VALUE_TOKENS
            or val_norm in cls.INVALID_VALUE_TOKENS
        )
    
    # =========================================================================
    # MARKDOWN EXTRACTION METHODS
    # =========================================================================
    
    @classmethod
    def _extract_amneal_receiving_no_from_markdown(cls, markdown: str) -> Optional[str]:
        """Extract Amneal Receiving No from markdown using regex"""
        if not markdown:
            return None
        
        match = re.search(cls.AMNEAL_RECEIVING_NO_PATTERN, markdown, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    # =========================================================================
    # BMR EXTRACTION
    # =========================================================================
    
    @classmethod
    def _find_matching_key(
        cls,
        row: Dict[str, Any],
        candidate_keys: List[str]
    ) -> Optional[str]:
        """Find matching key in row using fuzzy matching"""
        normalized_row_keys = {
            cls._normalize_text(k): k for k in row.keys()
        }
        
        for candidate in candidate_keys:
            candidate_norm = cls._normalize_text(candidate)
            for row_key_norm, original_key in normalized_row_keys.items():
                if candidate_norm in row_key_norm:
                    return original_key
        
        return None
    
    @classmethod
    def _is_excluded_material(cls, material: str) -> bool:
        """Check if material name should be excluded (e.g., 'Recorded By')"""
        material_norm = cls._normalize_text(material)
        for ex in cls.BMR_EXCLUDE_MATERIAL_KEYWORDS:
            if cls._normalize_text(ex) in material_norm:
                return True
        return False
    
    @classmethod
    def _find_keyphrase_in_rules(cls, obj: Any) -> bool:
        """Check if any BMR key phrase exists in rules_or_instructions"""
        try:
            if isinstance(obj, dict):
                rules = obj.get("rules_or_instructions", [])
                if isinstance(rules, list):
                    for rule in rules:
                        if isinstance(rule, str):
                            rule_lower = rule.lower()
                            for phrase in cls.BMR_KEY_PHRASES:
                                if phrase.lower() in rule_lower:
                                    return True
                
                return any(
                    cls._find_keyphrase_in_rules(v)
                    for v in obj.values()
                )
            
            if isinstance(obj, list):
                return any(
                    cls._find_keyphrase_in_rules(i)
                    for i in obj
                )
        
        except Exception:
            return False
        
        return False
    
    @classmethod
    def _normalize_field(cls, field: Optional[Any]) -> Optional[Dict[str, Optional[str]]]:
        """Normalize A1/A2 dictionary values and convert invalid values to None"""
        if not isinstance(field, dict):
            return None
        
        return {
            k: (None if cls._is_invalid_value(v) else v)
            for k, v in field.items()
        }
    
    @classmethod
    def _extract_bmr_entries(
        cls,
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract potency calculation entries from BMR pages"""
        extracted = []
        
        for page_data in pages:
            if not isinstance(page_data, dict):
                continue
            
            # Get page number from the page data
            page_no = page_data.get("page", page_data.get("page_no", "unknown"))
            
            # Skip pages without potency calculation
            if not cls._find_keyphrase_in_rules(page_data):
                continue
            
            records = page_data.get("records", [])
            
            for row in records:
                if not isinstance(row, dict):
                    continue
                
                # Find material name
                material_key = cls._find_matching_key(row, cls.BMR_RAW_MATERIAL_KEYS)
                if not material_key:
                    continue
                
                material_value = str(row.get(material_key, "")).strip()
                if not material_value:
                    continue
                
                if cls._is_excluded_material(material_value):
                    continue
                
                # Find AR No
                ar_key = cls._find_matching_key(row, cls.BMR_AR_NO_KEYS)
                
                extracted.append({
                    "page_no": page_no,
                    "material": material_value,
                    "ar_no": cls._normalize_field(row.get(ar_key)) if ar_key else None,
                })
        
        return extracted
    
    # =========================================================================
    # ATTACHMENT EXTRACTION (MARKDOWN-BASED)
    # =========================================================================
    
    @classmethod
    def _should_skip_page_markdown(cls, page: Dict[str, Any]) -> bool:
        """Skip pages that are RM Disposition Forms (check markdown)"""
        markdown = page.get("markdown_page", "")
        
        for skip_title in cls.ATT_SKIP_SECTION_TITLES:
            if skip_title in markdown:
                return True
        
        return False
    
    @classmethod
    def _is_coa_page_markdown(cls, page: Dict[str, Any]) -> bool:
        """Check if page is a Certificate of Analysis (check markdown)"""
        markdown = page.get("markdown_page", "")
        return cls.ATT_COA_MARKER in markdown
    
    @classmethod
    def _find_coa_by_ar_no_markdown(
        cls,
        attachment_data: List[Dict[str, Any]],
        ar_no: str
    ) -> List[Dict[str, Any]]:
        """
        Find all CoA pages matching the given AR number (using markdown)
        
        Note: The BMR's 'A.R. No' field actually contains the Amneal Receiving No,
        so we match it against the CoA's 'Amneal Receiving No' field.
        """
        if not ar_no:
            return []
        
        ar_no_clean = str(ar_no).strip()
        matching_pages = []
        
        for page in attachment_data:
            # Skip RM Disposition Forms
            if cls._should_skip_page_markdown(page):
                continue
            
            # Only process CoA pages
            if not cls._is_coa_page_markdown(page):
                continue
            
            # Extract Amneal Receiving No from markdown
            markdown = page.get("markdown_page", "")
            page_ar_no = cls._extract_amneal_receiving_no_from_markdown(markdown)
            
            if page_ar_no and page_ar_no == ar_no_clean:
                matching_pages.append(page)
        
        return matching_pages
    
    # =========================================================================
    # VALIDATION LOGIC
    # =========================================================================
    
    @classmethod
    def _validate_ar_no(
        cls,
        bmr_ar_no: Optional[str],
        attachment_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that AR No exists in CoA attachments.
        
        LENIENT MODE: Passes if ANY page in the attachments has the matching AR No.
        Does NOT require all pages of a multi-page CoA to have the same number.
        """
        # Skip if AR No is invalid
        if not bmr_ar_no or bmr_ar_no in ["", "NA", "N/A"]:
            return {"status": "skip", "message": "No AR number"}
        
        # Find matching CoA using markdown extraction (simple search)
        coa_pages = cls._find_coa_by_ar_no_markdown(attachment_data, bmr_ar_no)
        
        if not coa_pages:
            return {
                "status": "fail",
                "message": f"No CoA found for AR No: {bmr_ar_no}"
            }
        
        # If at least one CoA page found, it's a match (lenient)
        return {
            "status": "pass",
            "message": f"AR No {bmr_ar_no} matches Amneal Receiving No in CoA (found on {len(coa_pages)} page(s))"
        }
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    @classmethod
    def run_validation(
        cls,
        bmr_data: List[Dict],
        attachment_data: List[Dict]
    ) -> List[Dict]:
        """
        Run AR No matching validation
        
        Args:
            bmr_data: List of BMR page dictionaries or Dict of pages (auto-converted)
            attachment_data: List of attachment pages
        
        Returns:
            List of validation results for each page with format:
            {
                "page": <int>,
                "section_name": "Quality Control",
                "check_name": "AR_NO_Match",
                "anomaly_status": 0 or 1
            }
        """
        # Normalize bmr_data to list format if it's a dictionary
        if isinstance(bmr_data, dict):
            bmr_data = [
                {"page": page_no, **page_data}
                for page_no, page_data in bmr_data.items()
            ]
            
        # Flatten attachment_data if it contains nested lists
        if isinstance(attachment_data, list):
            flat_data = []
            for item in attachment_data:
                if isinstance(item, list):
                    flat_data.extend(item)
                elif isinstance(item, dict):
                    flat_data.append(item)
            attachment_data = flat_data
        
        results = []
        
        # Extract BMR potency calculation entries
        bmr_entries = cls._extract_bmr_entries(bmr_data)
        
        # Track which pages have potency calculations
        pages_with_potency = set()
        
        for entry in bmr_entries:
            page_no = entry.get("page_no")
            material = entry.get("material")
            
            pages_with_potency.add(page_no)
            
            # Extract A1 AR No
            ar_no_a1 = None
            ar_no_dict = entry.get("ar_no")
            if isinstance(ar_no_dict, dict):
                ar_no_a1 = ar_no_dict.get("A1")
            
            # Validate A1
            a1_result = cls._validate_ar_no(ar_no_a1, attachment_data)
            
            # Extract A2 AR No (if present)
            ar_no_a2 = None
            if isinstance(ar_no_dict, dict):
                ar_no_a2 = ar_no_dict.get("A2")
            
            # Validate A2 (if present)
            a2_result = None
            if ar_no_a2 and ar_no_a2 not in ["", "NA", "N/A"]:
                a2_result = cls._validate_ar_no(ar_no_a2, attachment_data)
            
            # Determine overall anomaly status
            anomaly_status = 0
            
            if a1_result["status"] == "fail":
                anomaly_status = 1
            
            if a2_result and a2_result["status"] == "fail":
                anomaly_status = 1
            
            results.append({
                "page": int(page_no),
                "section_name": cls.SECTION_NAME,
                "check_name": cls.CHECK_NAME,
                "anomaly_status": anomaly_status
            })
        
        # Add status 0 for all other pages
        all_pages = set()
        for page_data in bmr_data:
            if isinstance(page_data, dict):
                page_no = page_data.get("page", page_data.get("page_no"))
                if page_no is not None:
                    all_pages.add(str(page_no))
        
        for page_no in all_pages:
            if page_no not in pages_with_potency:
                results.append({
                    "page": int(page_no),
                    "section_name": cls.SECTION_NAME,
                    "check_name": cls.CHECK_NAME,
                    "anomaly_status": 0
                })
        
        # Sort by page number
        results.sort(key=lambda x: x["page"])
        
        return results
 
 
class ARNumberValidator:
    """
    Check 24. AI tool should be able to verify the AR. No. form reference attached documents and actual issued quantity should be verified by standard quantity.
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
    Author: Mehul Kasliwal (22 Jan 2026)
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

    # Column name synonyms for the SECOND attachment (e.g., Attachment 9)
    # Material Code is stored under "Material Code No." instead of "Material Code"
    ATT2_MATERIAL_CODE_KEYS = [
        "Material Code No.",
        "Material Code No.:",
        "Material Code No",
    ]

    # AR Number is stored under "Batch No." instead of "AR. No." / "A.R. No."
    ATT2_AR_NO_KEYS = [
        "Batch No.",
        "Batch No",
        "Batch No.:",
    ]
 
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

    def normalize_material_code(self, code: str) -> str:
        """
        Normalize material code to handle OCR variations.
        E.g., 'RML-0910' -> 'RMI-0910', 'PMI- 0481' -> 'PMI-0481'
        """
        if not isinstance(code, str):
            return ""
        # Remove extra spaces
        code = re.sub(r'\s+', '', code.strip().upper())
        code = code.replace(".", "-")
        return code
 
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
                        code = self.normalize_material_code(record.get(key, ""))
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

    def load_attachment_2_data(self, attachment_data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Load SECOND attachment data (e.g., Attachment 9) and index by normalized Material Code.
        
        This attachment uses different field names:
        - Material Code: "Material Code No." (instead of "Material Code")
        - AR Number: "Batch No." (instead of "AR. No.")
        
        The AR value from "Batch No." is mapped into "AR. No." in the returned records
        so that validate_row() can find it using its existing lookup logic.
        
        Returns: {material_code: [list of records with that code]}
        """
        index = {}

        for page in attachment_data:
            page_content = page.get("page_content", [])
            if isinstance(page_content, dict):
                page_content = [page_content]

            def process_record_att2(record: Dict):
                """Process a single record from the second attachment format."""
                code = None
                # Try different key names for Material Code in second attachment
                for key in self.ATT2_MATERIAL_CODE_KEYS:
                    if key in record:
                        code = self.normalize_material_code(record.get(key, ""))
                        break
                
                if not code:
                    return
                
                # Map the AR number from "Batch No." into "AR. No." 
                # so validate_row() can pick it up with its existing logic
                mapped_record = dict(record)
                for ar_key in self.ATT2_AR_NO_KEYS:
                    if ar_key in mapped_record:
                        ar_value = mapped_record[ar_key]
                        if ar_value and not self.is_blank_or_na(ar_value):
                            mapped_record["AR. No."] = ar_value
                        break
                
                if code not in index:
                    index[code] = []
                index[code].append(mapped_record)

            for content in page_content:
                if isinstance(content, dict):
                    # Check if it's a single record (has Material Code No. directly)
                    has_material = any(key in content for key in self.ATT2_MATERIAL_CODE_KEYS)
                    if has_material:
                        process_record_att2(content)

                    # Check for nested records
                    if "records" in content:
                        for record in self.iter_dict_rows(content["records"]):
                            process_record_att2(record)
                    
                    # Check for section_title with records
                    if "section_title" in content and "records" in content:
                        for record in self.iter_dict_rows(content["records"]):
                            process_record_att2(record)

        return index

    def load_two_attachments(self, attachment_1: List[Dict], attachment_2: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Load two attachment datasets and index by normalized Material Code.
        
        - attachment_1 is loaded using the standard format (e.g., Attachment 3)
        - attachment_2 is loaded using the alternate format (e.g., Attachment 9)
        
        Returns: {
            'attachment_1': {material_code: [list of records]},
            'attachment_2': {material_code: [list of records]}
        }
        """
        return {
            'attachment_1': self.load_attachment_data(attachment_1),
            'attachment_2': self.load_attachment_2_data(attachment_2)
        }
 
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
 
        material_code = self.normalize_material_code(bmr_row.get(material_col, ""))
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
        attachment_1: List[Dict],
        attachment_2: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Main entry point for AR number validation.
 
        Args:
            bmr_data: List of page dicts from BMR JSON
            attachment_1: List of page dicts from first attachment (Attachment 3)
            attachment_2: Optional list of page dicts from second attachment (fallback)
 
        Returns:
            List of result dicts, one per page with anomalies
        """
        results = []
 
        # Load and index attachment data from both sources
        if attachment_2 is not None:
            attachment_indices = self.load_two_attachments(attachment_1, attachment_2)
            attachment_index_1 = attachment_indices['attachment_1']
            attachment_index_2 = attachment_indices['attachment_2']
        else:
            # Only one attachment provided
            attachment_index_1 = self.load_attachment_data(attachment_1)
            attachment_index_2 = {}
 
        for page_dict in bmr_data:
            page_no = str(page_dict.get("page", "unknown"))
            page_content = page_dict.get("page_content", [])
 
            try:
                # Extract records from this page
                records = self.extract_bmr_records(page_content)
                if not records:
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
                    material_code = self.normalize_material_code(row.get(material_col, ""))
                    
                    # Skip if we've already processed this material code on this page
                    if material_code in processed_materials or not material_code:
                        continue
                    
                    processed_materials.add(material_code)
                    # Try first attachment, fallback to second if not found
                    attachment_records = attachment_index_1.get(material_code, [])
                    if not attachment_records and attachment_index_2:
                        attachment_records = attachment_index_2.get(material_code, [])

                    row_anomalies = self.validate_row(
                        row, attachment_records,
                        material_col, ar_col
                    )
                    page_anomalies.extend(row_anomalies)
 
                # Record result for this page - directly in final format
                results.append({
                    "page": int(page_no),
                    "section_name": "Quality Control",
                    "check_name": "AR_NO_Match_Structured",
                    "anomaly_status": 1 if page_anomalies else 0
                })
 
            except Exception as e:
                # Even for errors, return in the same simplified format
                results.append({
                    "page": int(page_no),
                    "section_name": "Quality Control",
                    "check_name": "AR_NO_Match_Structured",
                    "anomaly_status": 0
                })
 
        return results
 
 
class CombinedARValidator:
    """
    Combined AR Number Validator - Orchestrates both validation approaches
    
    This class combines results from:
    - Check24ARNoVerifyResults (Class A): Validates AR No from filtered JSON (pages 1-28, Potency Calculation)
    - ARNumberValidator (Class B): Validates AR No from master JSON (pages 30-49, Dispensing tables)
    
    The combined result uses OR logic: if either validator detects an anomaly on a page,
    the combined result will show anomaly_status = 1 for that page.
    
    Logic: combined_anomaly_status = max(class_a_status, class_b_status)
    
    Example:
    - Page 34: Class A = 0, Class B = 1 → Combined = 1
    - Page 20: Class A = 1, Class B = 0 → Combined = 1
    - Page 50: Class A = 0, Class B = 0 → Combined = 0
    """
    
    SECTION_NAME = "Quality Control"
    CHECK_NAME = "AR_NO_Verify_Results"
    
    @classmethod
    def run_validation(
        cls,
        filtered_json_data: List[Dict],
        master_json_data: List[Dict],
        attachment_2_data: List[Dict],
        attachment_3_data: List[Dict],
        attachment_4_data: List[Dict] = None
    ) -> List[Dict]:
        """
        Run combined AR number validation using both approaches
        
        Args:
            filtered_json_data: Filtered JSON data (dict format, used by Class A)
            master_json_data: Master JSON data (list format, used by Class B)
            attachment_2_data: Attachment 2 data (for Class A)
            attachment_3_data: Attachment 3 data (primary for Class B)
            attachment_4_data: Attachment 4 data (fallback for Class B, used when material code not found in attachment_3)
        
        Returns:
            List of combined validation results for all pages:
            [
                {
                    "page": int,
                    "section_name": "Quality Control",
                    "check_name": "AR_NO_Combined_Validation",
                    "anomaly_status": 0 or 1
                },
                ...
            ]
        """
        # Run Class A validation (Check24ARNoVerifyResults - Filtered JSON + Attachment 2)
        class_a_results = Check24ARNoVerifyResults.run_validation(
            filtered_json_data,
            attachment_2_data
        )
        
        # Run Class B validation (ARNumberValidator - Master JSON + Attachment 3 + Attachment 4 as fallback)
        validator_b = ARNumberValidator()
        class_b_results = validator_b.run_validation(
            master_json_data,
            attachment_3_data,
            attachment_4_data  # Pass attachment_4 as fallback
        )
        
        # Create a dictionary to merge results by page number
        merged_results = {}
        
        # Process Class A results
        for result in class_a_results:
            page_no = result["page"]
            merged_results[page_no] = result["anomaly_status"]
        
        # Process Class B results - use max() to implement OR logic
        for result in class_b_results:
            page_no = result["page"]
            if page_no in merged_results:
                # OR logic: take max of both statuses
                merged_results[page_no] = max(merged_results[page_no], result["anomaly_status"])
            else:
                merged_results[page_no] = result["anomaly_status"]
        
        # Convert back to list format
        combined_results = []
        for page_no in sorted(merged_results.keys()):
            combined_results.append({
                "page": page_no,
                "section_name": cls.SECTION_NAME,
                "check_name": cls.CHECK_NAME,
                "anomaly_status": merged_results[page_no]
            })
        
        return combined_results
#


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
if __name__ == "__main__":
    # Path to JSON
    ocr_json_path = "/home/softsensor/Desktop/Amneal/all_result_76_20feb.json"
    json_file_path = "/home/softsensor/Desktop/Amneal/all_result_76_20feb.json"

    print("=" * 70)
    print("Running Check 24 - AR Number Verification")
    print("=" * 70)

    
    # Load the main JSON
    data = load_json_file(json_file_path)
    steps = data.get("steps", {})

    # ---- Extract filled_master_json (BMR data) ----
    filled_master_json = steps.get("filled_master_json", [])
    print(f"Loaded {len(filled_master_json)} pages from filled_master_json")

    # Build BMR pages as list (for Check24ARNoVerifyResults and ARNumberValidator)
    bmr_pages_list = []
    for page_obj in filled_master_json:
        page_no = page_obj.get("page", "unknown")
        page_data = {
            "page": page_no,
            "page_content": page_obj.get("page_content", []),
            "markdown_page": page_obj.get("markdown_page", ""),
            "additional_keys": page_obj.get("additional_keys", []),
            "rules_or_instructions": [],
            "records": []
        }

        # Extract records and rules from page_content
        for content in page_obj.get("page_content", []):
            content_type = content.get("type", "")

            if content_type == "table":
                table_json = content.get("table_json", {})
                if "records" in table_json:
                    page_data["records"].extend(table_json["records"])

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

        bmr_pages_list.append(page_data)

    # ---- Extract attachment data ----
    attachment_result = steps.get("attachment_result", {})

    # Attachment 2 (CoA of API) - used by Check24ARNoVerifyResults
    attachment_2_raw = attachment_result.get("24", {})
    attachment_2_data = []
    for att_key, att_pages in attachment_2_raw.items():
        if isinstance(att_pages, list):
            attachment_2_data.extend(att_pages)
    print(f"Attachment 2 (CoA): {len(attachment_2_data)} pages")

    # Attachment 3 (Dispensing tables) - used by ARNumberValidator
    attachment_3_data = []
    attachment_3_path = "/home/softsensor/Desktop/Amneal/challenge_bmr/Attachment 3.json"
    
    if os.path.exists(attachment_3_path):
        print(f"DEBUG: Found Attachment 3 at {attachment_3_path}")
        attachment_3_data = load_json_file(attachment_3_path)
        print(f"DEBUG: Loaded {len(attachment_3_data)} pages from Attachment 3")
    else:
        print("DEBUG: Attachment 3.json not found, falling back to embedded attachments")
        # Try multiple possible attachment keys for dispensing data
        for att_key in attachment_result:
            att_section = attachment_result[att_key]
            if isinstance(att_section, dict):
                for sub_key, sub_pages in att_section.items():
                    if isinstance(sub_pages, list):
                        attachment_3_data.extend(sub_pages)
    
    print(f"All attachments flattened: {len(attachment_3_data)} pages")

    # Attachment 4 (fallback for ARNumberValidator when material code not found in Attachment 3)
    attachment_4_data = []
    attachment_4_path = "/home/softsensor/Desktop/Amneal/challenge_bmr/Attachment 9.json"
    
    if os.path.exists(attachment_4_path):
        print(f"DEBUG: Found Attachment 4 at {attachment_4_path}")
        attachment_4_data = load_json_file(attachment_4_path)
        print(f"DEBUG: Loaded {len(attachment_4_data)} pages from Attachment 4")
    else:
        print("DEBUG: Attachment 4.json not found, ARNumberValidator will use only Attachment 3")

    # ================================================================
    # Run CombinedARValidator
    # ================================================================
    print("\n" + "=" * 70)
    print("Combined: CombinedARValidator (OR logic)")
    print("=" * 70)

    combined_results = CombinedARValidator.run_validation(
        filtered_json_data=bmr_pages_list,
        master_json_data=bmr_pages_list,
        attachment_2_data=attachment_2_data,
        attachment_3_data=attachment_3_data,
        attachment_4_data=attachment_4_data if attachment_4_data else None
    )
    print(f"\nCombined results ({len(combined_results)} pages):")
    anomaly_pages_c = [r for r in combined_results if r["anomaly_status"] == 1]
    print(f"  Pages with anomalies: {len(anomaly_pages_c)}")
    for r in anomaly_pages_c:
        print(f"    Page {r['page']}: anomaly_status={r['anomaly_status']}")
    if not anomaly_pages_c:
        print("  No anomalies detected.")

    # ================================================================
    # Print full results as JSON
    # ================================================================
    print("\n" + "=" * 70)
    print("Full Results (JSON)")
    print("=" * 70)
    print(json.dumps(combined_results, indent=2))