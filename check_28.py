import json
import re
from typing import Dict, List, Any, Optional
from decimal import Decimal, InvalidOperation
from pprint import pprint as pp

class AccessQtyVerified:
    """
    Check 28. Verify Primary Packing Material Issued Quantity against Required Quantity.
    ---
    Description :
    On pages for "PRIMARY PACKING MATERIAL ISSUANCE AND DISPENSING", validates that the Issued Quantity is greater than or equal to the Required Quantity.
    ----
    type : Python bases
    ----
    Attachment Required : NO
    ----
    Author : Mehul Kasliwal (15 Jan 2026)
    """
    # ===================== CONSTANTS =====================
    
    SECTION_NAME = "Process Parameters"
    CHECK_NAME = "% access qty verified"
    
    # Column name synonyms for required quantity
    REQUIRED_QTY_SYNONYMS = [
        'Qty. Required', 
        'Quantity Req.', 
        'Quantity Required', 
        'Std. Qty.'
    ]
    
    # Column name synonyms for observed/dispensed quantity
    OBSERVED_QTY_SYNONYMS = [
        'Dispensed', 
        'Issued Qty', 
        'Quantity Dispensed', 
        'Actual issued Qty.'
    ]
    
    # Values considered as blank or NA
    BLANK_VALUES = {"", "na", "na.", "n/a", "n/a.", "n.a", "n.a."}

    # ===================== HELPER METHODS =====================

    def is_blank_or_na(self, v: Any) -> bool:
        """Check if a value is blank or represents N/A."""
        if not isinstance(v, str):
            return False
        return v.strip().lower() in self.BLANK_VALUES

    def iter_dict_rows(self, records: List) -> List[Dict]:
        """Flatten nested record structures, yielding individual dict rows."""
        result = []
        for r in records:
            if isinstance(r, dict):
                result.append(r)
            elif isinstance(r, list):
                result.extend(self.iter_dict_rows(r))
        return result

    def normalize_unit(self, unit: str) -> str:
        u = unit.lower().strip().rstrip('.')
        if u in ['kg', 'kgs']: return 'kg'
        if u in ['g', 'gm', 'gms']: return 'g'
        if u in ['mtr', 'mtrs', 'meter', 'meters']: return 'mtr'
        if u in ['nos', 'no']: return 'nos'
        return u

    def parse_quantities_by_unit(self, s: Any) -> Dict[str, Decimal]:
        """
        Parse quantities and group them by unit.
        Returns a dict like {'kg': Decimal('10.5'), 'mtr': Decimal('100')}
        """
        if not isinstance(s, str) or self.is_blank_or_na(s) or 'line' in s.lower():
            return {}
        
        # Regex to capture value and unit
        # Pattern: number followed by unit
        unit_pattern = r'(\d[\d,]*\.?\d*)\s*(Kg|kg|Kgs|kgs|g|gm|Gms|gms|Mtr|Mtrs|Meter|Meters|Nos|Nos\.|No\.|No)\b'
        matches = re.findall(unit_pattern, s, re.I)
        
        results = {}
        found_any = False
        
        for val_str, unit_str in matches:
            found_any = True
            try:
                val = Decimal(val_str.replace(',', ''))
                norm_unit = self.normalize_unit(unit_str)
                results[norm_unit] = results.get(norm_unit, Decimal(0)) + val
            except InvalidOperation:
                continue
                
        # Fallback: if no units found but there is a number, treat as dimensionless/unknown unit
        if not found_any:
             fallback_pattern = r'(\d[\d,]*\.?\d*)'
             m = re.search(fallback_pattern, s.strip())
             if m:
                 try:
                     val = Decimal(m.group(1).replace(',', ''))
                     results['__no_unit__'] = val
                 except InvalidOperation:
                     pass

        return results

    def find_col(self, records: List[Dict], synonyms: List[str]) -> Optional[str]:
        """Find which column name variant exists in the records."""
        for syn in synonyms:
            if any(isinstance(r, dict) and syn in r for r in records):
                return syn
        return None

    # ===================== VALIDATION LOGIC =====================

    def validate_page(self, records: List[Dict], page_no=None) -> List[Dict]:
        anomalies = []
        flat = self.iter_dict_rows(records)
        
        req_col = self.find_col(flat, self.REQUIRED_QTY_SYNONYMS)
        obs_col = self.find_col(flat, self.OBSERVED_QTY_SYNONYMS)

        if not req_col or not obs_col:
            # Optionally log missing columns if needed
            return []

        for i, row in enumerate(flat):
            req_val = row.get(req_col)
            obs_val = row.get(obs_col)
            
            # Parse quantities
            req_map = self.parse_quantities_by_unit(req_val)
            obs_map = self.parse_quantities_by_unit(obs_val)

            if not req_map or not obs_map:
                continue

            # Compare matching units
            converted_match = False
            
            # Iterate over units present in Observed value
            for unit, obs_qty in obs_map.items():
                if unit in req_map:
                    req_qty = req_map[unit]
                    converted_match = True
                    # Condition: Issued (Observed) >= Required
                    if obs_qty < req_qty:
                         material_name = row.get("Material", row.get("Item", "Unknown Material"))
                         anomalies.append({
                            "parameter": f"Issued Qty vs Required Qty - {material_name}",
                            "observed_value": float(obs_qty),
                            "standard_range": f">= {float(req_qty)}"
                         })
                         break # One failure per row is enough
                else:
                    # Unit mismatch or conversion needed (e.g. Mtr vs Kg)
                    pass
            
            # Special case: Handling no units
            if not converted_match and '__no_unit__' in obs_map and '__no_unit__' in req_map:
                 obs_qty = obs_map['__no_unit__']
                 req_qty = req_map['__no_unit__']
                 if obs_qty < req_qty:
                     material_name = row.get("Material", row.get("Item", "Unknown Material"))
                     anomalies.append({
                        "parameter": f"Issued Qty vs Required Qty - {material_name}",
                        "observed_value": float(obs_qty),
                        "standard_range": f">= {float(req_qty)}"
                     })

        return anomalies

    def run_validation(self, bmr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []

        for page_no, page in bmr_data.items():
            # Check for INCLUSION condition
            rules = [str(r).strip().upper() for r in page.get("rules_or_instructions", [])]
            cond_1 = any("PRIMARY PACKING MATERIAL ISSUANCE AND DISPENSING" in r for r in rules)
            cond_2 = any("PRIMARY PACKING MATERIALS:" in r for r in rules)
            
            should_validate = cond_1 and cond_2

            if not should_validate:
                results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })
                continue
            
            try:
                records = page.get("records", [])
                anomalies = self.validate_page(records, page_no=page_no)
                
                results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 1 if anomalies else 0
                })
            except Exception:
                results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })

        return results


# ===================== RUN SCRIPT =====================

def load_json_file(filepath: str) -> dict:
    """Load JSON data from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_bmr_data(json_data: dict | list) -> dict:
    """
    Extract BMR data from the JSON structure.
    Converts the filled_master_json array into a page-keyed dictionary.
    """
    if isinstance(json_data, list):
        filled_master_json = json_data
    else:
        filled_master_json = json_data.get("steps", {}).get("filled_master_json", [])
    
    if not filled_master_json:
        print("Warning: No filled_master_json found in the JSON data.")
        return {}
    
    bmr_data = {}
    for page_obj in filled_master_json:
        page_no = str(page_obj.get("page", ""))
        if not page_no:
            continue
            
        page_content = page_obj.get("page_content", [])
        records = []
        rules_or_instructions = []
        
        for content_item in page_content:
            if content_item.get("type") == "table":
                table_json = content_item.get("table_json", {})
                
                if isinstance(table_json, dict):
                    if "records" in table_json:
                        records.extend(table_json["records"])
                    else:
                        records.append(table_json)
                elif isinstance(table_json, list):
                    for item in table_json:
                        if isinstance(item, dict):
                            if "records" in item:
                                records.extend(item["records"])
                            else:
                                records.append(item)
                        elif isinstance(item, list):
                            records.extend(item)
            
            if content_item.get("type") == "kv_text_block":
                kv_block = content_item.get("extracted_kv_text_block", {})
                if isinstance(kv_block, list) and kv_block:
                    rules = kv_block[0].get("rules_or_instructions", [])
                    rules_or_instructions.extend(rules)
                elif isinstance(kv_block, dict):
                    rules = kv_block.get("rules_or_instructions", [])
                    rules_or_instructions.extend(rules)
        
        bmr_data[page_no] = {
            "records": records,
            "rules_or_instructions": rules_or_instructions
        }
    
    return bmr_data


if __name__ == "__main__":
    import os
    
    # Path to the JSON file
    json_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "New_BMRs/Emulsion_line_AH240074_filled_master_data.json"
    )
    
    print(f"Loading JSON file: {json_file}")
    json_data = load_json_file(json_file)
    print(f"JSON loaded successfully.")
    
    bmr_data = extract_bmr_data(json_data)
    print(f"Extracted {len(bmr_data)} pages from the BMR data.")
    
    validator = AccessQtyVerified()
    
    print("\nRunning % Access Qty Verified Check...")
    print("=" * 60)
    
    results = validator.run_validation(bmr_data)
    
    print(f"\nValidation completed. Total pages processed: {len(results)}")
    print("=" * 60)
    
    anomaly_pages = [r for r in results if r.get("anomaly_status") == 1]
    normal_pages = [r for r in results if r.get("anomaly_status") == 0]
    
    print(f"\nSummary:")
    print(f"  - Pages with anomalies: {len(anomaly_pages)}")
    print(f"  - Pages without anomalies: {len(normal_pages)}")
    
    if anomaly_pages:
        print(f"\nPages with anomalies detected:")
        for result in anomaly_pages:
            print(f"  - Page {result['page_no']}: {result['check_name']}")
    
    print("\n" + "=" * 60)
    print("Full Results:")
    print("=" * 60)
    pp(results)
