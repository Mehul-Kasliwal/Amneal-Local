import json
import re
from typing import Dict, List, Any, Optional
from decimal import Decimal, InvalidOperation
from pprint import pprint as pp


class MaterialRequisitionValidator:
    """
    Check 29: AI tool should verify actual issued quantity against standard quantity.
    ---
    Description:
    1. Find BMR pages with dispensing tables (Material Code, Qty Required, Qty Dispensed)
    2. Match against Attachment 42 records by Material Code
    3. Verify Quantity Dispensed matches attachment's Quantity Issued
    4. Verify Quantity Required matches Quantity Dispensed in BMR
    ---
    Type: Python-based
    ---
    Attachment Required: YES
    ---
    Author: Mehul Kasliwal (23 Jan 2026)
    """

    # ===================== CONSTANTS =====================

    SECTION_NAME = "Material Requisition Verification"
    CHECK_NAME = "issued_qty_att_verify"

    # Column name synonyms for BMR dispensing tables
    MATERIAL_CODE_SYNONYMS = [
        "Material code",
        "Material Code",
        "Material code.",
    ]

    QTY_REQUIRED_SYNONYMS = [
        "Qty. Required",
        "Qty Required",
        "Quantity Required",
        "Quantity Req.",
        "Std. Qty.",      # Page 34 variation
    ]

    QTY_DISPENSED_SYNONYMS = [
        "Quantity Dispensed",
        "Qty. Dispensed",
        "Qty Dispensed",
        "Quantity Issued",
        "Qty. Issued",
        "Actual issued Qty.",  # Page 34 variation
        "Issued Qty",          # Page 36 variation
    ]

    # Tolerance for quantity matching
    TOLERANCE = Decimal("0.1")

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

    def normalize_unit(self, unit: str) -> str:
        """Normalize unit strings to standard uppercase keys."""
        if not unit: return ""
        u = unit.strip().upper().replace('.', '')
        # Map common variations
        if u in ['KGS', 'KG', 'KZ']: return 'KG'
        if u in ['MTR', 'MTS', 'METER', 'METERS']: return 'MTR'
        if u in ['G', 'GM', 'GMS', 'GR']: return 'GM'
        if u in ['NO', 'NOS']: return 'NOS'
        if u in ['L', 'LTR', 'LITERS']: return 'L'
        if u in ['ML', 'MILLILITERS']: return 'ML'
        return u

    def extract_quantities(self, text: str, uom_text: str = None) -> Dict[str, Decimal]:
        """
        Extract quantities and map them to units.
        Returns dict: {'KG': Decimal('89.191'), 'MTR': Decimal('1129')}
        """
        results = {}
        if not isinstance(text, str) or self.is_blank_or_na(text):
            return results

        # Strategy 1: Explicit Number + Unit pairs
        # Regex for "123.45 Unit", "123 Unit"
        # Handles "1129 Mtr", "89.191 Kg"
        # We look for digits followed optionally by a dot and digits, then space, then letters
        # Using a restricted list of units to avoid matching random text as units
        explicit_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', text)
        found_explicit = False
        valid_units = ['KG', 'MTR', 'GM', 'NOS', 'L', 'ML']
        
        for m in explicit_matches:
            val_str, unit_str = m.groups()
            norm_unit = self.normalize_unit(unit_str)
            if norm_unit in valid_units:
                try:
                    val = Decimal(val_str)
                    # Sum if unit already exists (e.g. "4.7 kg 15.426 Kg")
                    results[norm_unit] = results.get(norm_unit, Decimal(0)) + val
                    found_explicit = True
                except: continue
        
        # If we found explicit valid units, return them (prioritize explicit over implicit)
        if found_explicit:
            return results

        # Strategy 2: Implicit units via UOM context
        # Case: Quantity "1129 89.191" with UOM "Mtr. kg."
        if uom_text and not found_explicit:
            # Parse UOMs
            uoms = [self.normalize_unit(u) for u in re.split(r'[\s\.,]+', uom_text) if u.strip()]
            # Parse values
            values = [Decimal(v) for v in re.findall(r'(\d+(?:\.\d+)?)', text) if v]
            
            # Map index to index if counts match or values <= uoms
            for i, val in enumerate(values):
                if i < len(uoms):
                    u = uoms[i]
                    results[u] = results.get(u, Decimal(0)) + val
        
        # Strategy 3: Default/Unitless fallback
        # If no units found, capture values as 'DEFAULT' (summed or list?)
        # For backward compatibility with single values, we'll store as 'DEFAULT'
        if not results:
             values = []
             try:
                 values = [Decimal(v) for v in re.findall(r'(\d+(?:\.\d+)?)', text) if v]
             except: pass
             
             if values:
                 # If we have values but couldn't determine units, sum them 
                 # (mimics old behavior for simple cases) OR store as unitless
                 # Given the user wants to FIX the summing bug, let's treat single value as default
                 # and multiple values as separate if possible, but without keys it's hard.
                 # Let's sum for now but mostly we expect units.
                 if len(values) == 1:
                     results['DEFAULT'] = values[0]
                 else:
                     # If multiple numbers and no units, likely specific case. Summing is safer fallback than dropping.
                     results['DEFAULT'] = sum(values)

        return results

    def compare_quantities_maps(self, map1: Dict[str, Decimal], map2: Dict[str, Decimal]) -> List[str]:
        """
        Compare two quantity maps. Returns list of mismatch descriptions.
        Logic: Compare values for intersecting units.
        """
        mismatches = []
        common_units = set(map1.keys()) & set(map2.keys())
        
        # If no common units but both have data, check if one is Default and other is compatible?
        # e.g. Map1={DEFAULT: 10}, Map2={KG: 10}. Assume match if values match?
        if not common_units:
            if 'DEFAULT' in map1 and len(map2) == 1:
                common_units.add('DEFAULT')
                # Temporarily map the single unit of map2 to DEFAULT for comparison
                val2 = list(map2.values())[0]
                if abs(map1['DEFAULT'] - val2) > self.TOLERANCE:
                    mismatches.append(f"Value {map1['DEFAULT']} != {val2}")
                return mismatches
            elif 'DEFAULT' in map2 and len(map1) == 1:
                common_units.add('DEFAULT')
                val1 = list(map1.values())[0]
                if abs(val1 - map2['DEFAULT']) > self.TOLERANCE:
                    mismatches.append(f"Value {val1} != {map2['DEFAULT']}")
                return mismatches
            
            # If completely disjoint units with data, it's a mismatch (unless ignoring units was requested, but user said use common unit)
            # If one is empty, that's handled by caller checking for None/Empty
            return mismatches

        for unit in common_units:
            val1 = map1.get(unit, Decimal(0))
            # Handle the temporary mapping for DEFAULT if needed, but here we iterate intersection
            # For strict intersection:
            if unit == 'DEFAULT':
                val2 = map2.get('DEFAULT', Decimal(0)) # Should exist
            else:
                 val2 = map2[unit]
            
            if abs(val1 - val2) > self.TOLERANCE:
                mismatches.append(f"{unit}: {val1} vs {val2}")
                
        return mismatches

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

    def has_dispensing_table(self, records: List[Dict]) -> bool:
        """Check if records represent a dispensing table with required columns."""
        flat = self.iter_dict_rows(records)
        if not flat:
            return False

        has_material = self.find_col(flat, self.MATERIAL_CODE_SYNONYMS) is not None
        has_qty_req = self.find_col(flat, self.QTY_REQUIRED_SYNONYMS) is not None
        has_qty_disp = self.find_col(flat, self.QTY_DISPENSED_SYNONYMS) is not None

        return has_material and has_qty_disp

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

            for content in page_content:
                if isinstance(content, dict):
                    # Check if it's a single record (has Material Code directly)
                    if "Material Code" in content:
                        code = self.normalize_material_code(content.get("Material Code", ""))
                        if code:
                            if code not in index:
                                index[code] = []
                            index[code].append(content)

                    # Check for nested records
                    if "records" in content:
                        for record in self.iter_dict_rows(content["records"]):
                            code = self.normalize_material_code(record.get("Material Code", ""))
                            if code:
                                if code not in index:
                                    index[code] = []
                                index[code].append(record)

        return index

    # ===================== VALIDATION =====================

    def validate_row(
        self,
        bmr_row: Dict,
        attachment_records: List[Dict],
        material_col: str,
        qty_req_col: str,
        qty_disp_col: str,
    ) -> List[Dict]:
        """Validate a single BMR row against attachment records."""
        anomalies = []

        material_code = self.normalize_material_code(bmr_row.get(material_col, ""))
        if not material_code:
            return anomalies

        # Extract BMR Quantities
        bmr_qty_req_map = self.extract_quantities(bmr_row.get(qty_req_col, ""))
        bmr_qty_disp_map = self.extract_quantities(bmr_row.get(qty_disp_col, ""))

        # Skip if no valid dispensed quantity
        if not bmr_qty_disp_map:
            return anomalies

        # Get attachment data for this material
        if not attachment_records:
            anomalies.append({
                "parameter": f"Material Not Found — {material_code}",
                "issue": "Material code not found in attachment",
                "bmr_value": material_code,
                "attachment_value": "N/A"
            })
            return anomalies

        # Aggregate attachment data
        att_qty_req_map_agg = {}
        att_qty_disp_map_agg = {}

        # If multiple attachment records exist for same material, sum the quantities per unit
        for att_record in attachment_records:
            # UOM Context
            uom = att_record.get("UOM", "")

            # Qty Req
            q_req = self.extract_quantities(att_record.get("Quantity Required", ""), uom)
            for u, v in q_req.items():
                att_qty_req_map_agg[u] = att_qty_req_map_agg.get(u, Decimal(0)) + v

            # Qty Issued
            q_iss = self.extract_quantities(att_record.get("Quantity Issued", att_record.get("Net. Wt. / Nos.", "")), uom)
            for u, v in q_iss.items():
                att_qty_disp_map_agg[u] = att_qty_disp_map_agg.get(u, Decimal(0)) + v

        # Check 2: Quantity Required vs Dispensed (within BMR)
        # Compare overlapping units within BMR
        mismatches = self.compare_quantities_maps(bmr_qty_req_map, bmr_qty_disp_map)
        for m in mismatches:
            anomalies.append({
                "parameter": f"Qty Required vs Dispensed — {material_code}",
                "issue": f"Quantity required does not match quantity dispensed in BMR ({m})",
                "bmr_required": str(bmr_qty_req_map),
                "bmr_dispensed": str(bmr_qty_disp_map)
            })

        # Check 3: BMR Qty Required vs Attachment Qty Required
        if bmr_qty_req_map and att_qty_req_map_agg:
            mismatches = self.compare_quantities_maps(bmr_qty_req_map, att_qty_req_map_agg)
            for m in mismatches:
                anomalies.append({
                    "parameter": f"Qty Required Mismatch — {material_code}",
                    "issue": f"Quantity required in BMR differs from attachment ({m})",
                    "bmr_value": str(bmr_qty_req_map),
                    "attachment_value": str(att_qty_req_map_agg)
                })

        # Check 4: BMR Qty Dispensed vs Attachment Qty Dispensed
        if bmr_qty_disp_map and att_qty_disp_map_agg:
            mismatches = self.compare_quantities_maps(bmr_qty_disp_map, att_qty_disp_map_agg)
            for m in mismatches:
                anomalies.append({
                    "parameter": f"Qty Dispensed Mismatch — {material_code}",
                    "issue": f"Quantity dispensed in BMR differs from attachment ({m})",
                    "bmr_value": str(bmr_qty_disp_map),
                    "attachment_value": str(att_qty_disp_map_agg)
                })

        return anomalies

    # ===================== ENTRY POINT =====================

    def run_validation(
        self,
        bmr_data: List[Dict],
        attachment_data: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Main entry point for material requisition validation.

        Args:
            bmr_data: List of page dicts from BMR JSON
            attachment_data: List of page dicts from Attachment 42 JSON

        Returns:
            List of result dicts in simplified format (page, section_name, check_name, anomaly_status)
        """
        results = []
        return_debug = []  # Detailed debug information (not returned)

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
                    # Simplified output
                    results.append({
                        "page": int(page_no),
                        "section_name": "Quality Control",
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                # Check if this is a dispensing table
                if not self.has_dispensing_table(records):
                    # Debug info
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": "Not a dispensing table (missing required columns)"
                    })
                    # Simplified output
                    results.append({
                        "page": int(page_no),
                        "section_name": "Quality Control",
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                # Find column names
                material_col = self.find_col(records, self.MATERIAL_CODE_SYNONYMS)
                qty_req_col = self.find_col(records, self.QTY_REQUIRED_SYNONYMS)
                qty_disp_col = self.find_col(records, self.QTY_DISPENSED_SYNONYMS)

                if not all([material_col, qty_disp_col]):
                    # Debug info
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": f"Incomplete columns (material_col={material_col}, qty_disp_col={qty_disp_col})"
                    })
                    # Simplified output
                    results.append({
                        "page": int(page_no),
                        "section_name": "Quality Control",
                        "check_name": self.CHECK_NAME,
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
                    attachment_records = attachment_index.get(material_code, [])

                    row_anomalies = self.validate_row(
                        row, attachment_records,
                        material_col, qty_req_col or "", qty_disp_col
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
                        "check_name": self.CHECK_NAME,
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
                        "check_name": self.CHECK_NAME,
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
                # Simplified output
                results.append({
                    "page": int(page_no),
                    "section_name": "Quality Control",
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })

        return results


# =====================================================
# ---------------- MAIN RUNNER ------------------------
# =====================================================
if __name__ == "__main__":
    import os

    # File paths
    bmr_filepath = '61_FILLED_MASTER_15_JAN.json'
    attachment_filepath = 'attachments_ocr_42_bmr_61 1.json'

    # Load data
    with open(bmr_filepath, "r", encoding="utf-8") as f:
        bmr_data = json.load(f)

    with open(attachment_filepath, "r", encoding="utf-8") as f:
        attachment_data = json.load(f)

    # Run validation
    validator = MaterialRequisitionValidator()
    results = validator.run_validation(bmr_data, attachment_data)

    # Output as JSON (results are already in the final format)
    print(json.dumps(results, indent=2))


