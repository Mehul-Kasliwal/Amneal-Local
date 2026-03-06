import json
import re
from typing import Dict, List, Any, Optional
from decimal import Decimal, InvalidOperation
from pprint import pprint as pp

class QuantityVarianceValidator:
    """
    Check 27. AI tool should be able to check issue quantity against the STD quantity.
    ---
    Description :
    Validates issue quantity against the STD quantity.
    ----
    type : Python bases
    ----
    Attachment Required : NO
    ----
    Author : Mehul Kasliwal (15 Jan 2026)
    """
    # ===================== CONSTANTS =====================
    
    CHECK_ID = "check_27"
    # SECTION_NAME = get_check_process(CHECK_ID)
    # CHECK_NAME = get_check_name(CHECK_ID)
    SECTION_NAME = "Process Parameters"
    CHECK_NAME = "Quantity Variance Check"
    
    # Column name synonyms for required quantity
    REQUIRED_QTY_SYNONYMS = [
        'Qty. Required', 
        'Quantity Req.', 
        'Quantity Required', 
        'Std. Qty.',
        'Std. Qty. (Nos.)',
        'Std. Qty. (Nos.) #'
    ]
    
    # Column name synonyms for observed/dispensed quantity
    OBSERVED_QTY_SYNONYMS = [
        'Dispensed', 
        'Issued Qty', 
        'Quantity Dispensed', 
        'Actual issued Qty.',
        'Net Wt.',
        'Qty. Issued',
        'Qty. Issued (Nos.)'
    ]
    
    # Trolley specification field names
    TROLLEY_SPEC_FIELDS = {
        'bags_per_tray': "Number of bags per tray",
        'trays_per_trolley': "No of trays per trolley",
        'min_load': "Minimum load (One trolley)"
    }
    
    # Trolley quantity field synonyms
    TROLLEY_QTY_SYNONYMS = [
        "Quantity Nos. Bags", 
        "Quantity Nos. Bag", 
        "Quantity Bags"
    ]
    
    # Trolley names to skip during validation
    TROLLEY_SKIP_NAMES = ("bags for sensor placement", "total")
    
    # Tolerance for ingredient quantity matching
    TOLERANCE = Decimal("0.001")
    
    # Values considered as blank or NA
    BLANK_VALUES = {"", "na", "na.", "n/a", "n/a.", "n.a", "n.a."}

    # Page headings that check_28 (AccessQtyVerified) runs on — skip these pages
    CHECK_28_HEADINGS = [
        "PRIMARY PACKING MATERIAL ISSUANCE AND DISPENSING",
        "PRIMARY PACKING MATERIALS:",
    ]

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

    def parse_quantity(self, s: Any) -> Optional[Decimal]:
        """
        Parse a quantity from a string, handling multiple numbers by summing them.
        Only sums numbers that appear to be quantity values (with units).
        Ignores date components and other numeric artifacts.
        
        Examples:
            "4.700 kg 15.426 kg" -> 20.126 (sum of both quantities)
            "21.274 kg" -> 21.274
            "25.364Kg# Recorded by: (signature) 08-APR-25" -> 25.364 (ignores dates)
            "(diagonal line)" -> None
        """
        if not isinstance(s, str) or self.is_blank_or_na(s) or 'line' in s.lower():
            return None
        
        # First, try to extract numbers that have units (Kg, kg, g, gm)
        unit_pattern = r'(\d[\d,]*\.?\d*)\s*(?:Kg|kg|g|gm)\b'
        unit_matches = re.findall(unit_pattern, s, re.I)
        
        if unit_matches:
            total = Decimal('0')
            for match in unit_matches:
                try:
                    total += Decimal(match.replace(',', ''))
                except InvalidOperation:
                    continue
            return total if total > 0 else None
        
        # If no unit-based numbers found, look for decimal numbers at the start
        first_num_pattern = r'^[^\d]*(\d[\d,]*\.\d+|\d{3,}[\d,]*)'
        m = re.search(first_num_pattern, s.strip())
        if m:
            try:
                return Decimal(m.group(1).replace(',', ''))
            except InvalidOperation:
                pass
        
        # Fallback: just the first number
        fallback_pattern = r'(\d[\d,]*\.?\d*)'
        m = re.search(fallback_pattern, s.strip())
        if m:
            try:
                return Decimal(m.group(1).replace(',', ''))
            except InvalidOperation:
                pass
        
        return None

    def normalize_unit(self, unit: str) -> str:
        """Normalize unit strings to standard uppercase keys."""
        if not unit:
            return ""
        u = unit.strip().upper().replace('.', '')
        if u in ['KGS', 'KG', 'KZ']:
            return 'KG'
        if u in ['G', 'GM', 'GMS', 'GR', 'GRAM', 'GRAMS']:
            return 'GM'
        if u in ['MTR', 'MTS', 'METER', 'METERS']:
            return 'MTR'
        if u in ['NO', 'NOS']:
            return 'NOS'
        if u in ['L', 'LTR', 'LITERS', 'LITRES']:
            return 'L'
        if u in ['ML', 'MILLILITERS']:
            return 'ML'
        return u

    # Conversion factors to a common base unit
    UNIT_TO_BASE = {
        'KG': ('MASS', Decimal('1000')),     # 1 KG = 1000 GM
        'GM': ('MASS', Decimal('1')),         # base unit for mass
        'L':  ('VOLUME', Decimal('1000')),    # 1 L = 1000 ML
        'ML': ('VOLUME', Decimal('1')),       # base unit for volume
    }

    def parse_quantity_with_unit(self, s: Any) -> Optional[tuple]:
        """
        Parse a quantity string and return (total_value, normalized_unit).
        Sums multiple values with the same unit in a single field.
        Returns None if no quantity can be parsed.
        
        Examples:
            "0.020 Kg"        -> (Decimal('0.020'), 'KG')
            "20.000 gm"       -> (Decimal('20.000'), 'GM')
            "46840 NOS"       -> (Decimal('46840'), 'NOS')
            "17500 7500"      -> (Decimal('25000'), '')       # summed unitless
            "17500 NOS 7500"  -> (Decimal('25000'), 'NOS')    # summed with unit
            "4.7 kg 15.426 Kg"-> (Decimal('20.126'), 'KG')    # summed same unit
            "48791"           -> (Decimal('48791'), '')
        """
        if not isinstance(s, str) or self.is_blank_or_na(s) or 'line' in s.lower():
            return None

        valid_units = ('KG', 'GM', 'NOS', 'L', 'ML', 'MTR')

        # Strategy 1: Find explicit Number + Unit pairs and sum by unit
        unit_pattern = r'(\d[\d,]*\.?\d*)\s*([A-Za-z]+)'
        matches = re.findall(unit_pattern, s)
        unit_totals = {}  # {normalized_unit: Decimal total}

        if matches:
            for val_str, unit_str in matches:
                norm_unit = self.normalize_unit(unit_str)
                # Skip date-like tokens (e.g. "Jul", "APR")
                if norm_unit in valid_units:
                    try:
                        val = Decimal(val_str.replace(',', ''))
                        unit_totals[norm_unit] = unit_totals.get(norm_unit, Decimal(0)) + val
                    except InvalidOperation:
                        continue

        if unit_totals:
            # Return the first (or only) unit with its summed total
            unit = next(iter(unit_totals))
            return (unit_totals[unit], unit)

        # Strategy 2: Unitless fallback — sum all numbers in the string
        all_numbers = re.findall(r'(\d[\d,]*\.?\d*)', s)
        if all_numbers:
            total = Decimal(0)
            for n in all_numbers:
                try:
                    total += Decimal(n.replace(',', ''))
                except InvalidOperation:
                    continue
            if total > 0:
                return (total, '')

        return None

    def quantities_match(self, std_val: Decimal, std_unit: str, obs_val: Decimal, obs_unit: str, tolerance: Decimal = Decimal('0')) -> bool:
        """
        Compare two quantities with unit awareness.
        Converts to a common base unit if both belong to the same dimension (mass, volume).
        
        Examples:
            0.020 KG vs 20.000 GM  -> True  (both = 20 GM)
            48791 NOS vs 46840 NOS -> False
        """
        # If same unit (or both unitless), direct compare
        if std_unit == obs_unit:
            return abs(std_val - obs_val) <= tolerance

        # Try unit conversion
        std_base_info = self.UNIT_TO_BASE.get(std_unit)
        obs_base_info = self.UNIT_TO_BASE.get(obs_unit)

        if std_base_info and obs_base_info:
            std_dimension, std_factor = std_base_info
            obs_dimension, obs_factor = obs_base_info
            if std_dimension == obs_dimension:
                # Convert both to base unit and compare
                std_base = std_val * std_factor
                obs_base = obs_val * obs_factor
                return abs(std_base - obs_base) <= (tolerance * max(std_factor, obs_factor))

        # Different/unknown dimensions — fall back to raw number comparison
        return abs(std_val - obs_val) <= tolerance

    def find_col(self, records: List[Dict], synonyms: List[str]) -> Optional[str]:
        """Find which column name variant exists in the records."""
        for syn in synonyms:
            if any(isinstance(r, dict) and syn in r for r in records):
                return syn
        return None

    # ===================== TROLLEY SPEC METHODS =====================

    def extract_trolley_spec(self, records: List[Dict]) -> Optional[Dict]:
        """
        Extract trolley loading specifications from records.
        
        Returns:
            {"min_load": Decimal, "max_load": Decimal} or None if not found
        """
        bags = trays = min_load = None

        for row in records:
            if self.TROLLEY_SPEC_FIELDS['bags_per_tray'] in row:
                val = row[self.TROLLEY_SPEC_FIELDS['bags_per_tray']]
                if not self.is_blank_or_na(val):
                    bags = self.parse_quantity(val)

            if self.TROLLEY_SPEC_FIELDS['trays_per_trolley'] in row:
                val = row[self.TROLLEY_SPEC_FIELDS['trays_per_trolley']]
                if not self.is_blank_or_na(val):
                    trays = self.parse_quantity(val)

            if self.TROLLEY_SPEC_FIELDS['min_load'] in row:
                val = row[self.TROLLEY_SPEC_FIELDS['min_load']]
                if not self.is_blank_or_na(val):
                    min_load = self.parse_quantity(val)

        if bags is None or trays is None or min_load is None:
            return None

        try:
            max_load = bags * trays
        except Exception:
            return None

        return {"min_load": min_load, "max_load": max_load}

    def get_trolley_qty(self, row: Dict) -> Optional[Decimal]:
        """Get trolley quantity from a row using synonyms."""
        for key in self.TROLLEY_QTY_SYNONYMS:
            if key in row:
                return self.parse_quantity(row.get(key))
        return None

    def trolley_load_anomalies(self, records: List[Dict], spec: Dict) -> List[Dict]:
        """
        Check if each trolley's bag count is within the valid range.
        
        Args:
            records: Flattened list of record dicts
            spec: {"min_load": Decimal, "max_load": Decimal}
            
        Returns:
            List of anomaly dicts
        """
        min_load = spec["min_load"]
        max_load = spec["max_load"]
        anomalies = []

        for row in records:
            if "Trolley" not in row:
                continue

            trolley_name = str(row.get("Trolley", "")).strip()
            if trolley_name.lower() in self.TROLLEY_SKIP_NAMES:
                continue

            qty = self.get_trolley_qty(row)
            if qty is None:
                continue

            if qty < min_load or qty > max_load:
                anomalies.append({
                    "parameter": f"Trolley Load — {trolley_name}",
                    "issue": f"Trolley load ({float(qty)} Bags) is out of expected bounds ({float(min_load)} to {float(max_load)} Bags).",
                    "observed_value": float(qty),
                    "expected_range": f"{float(min_load)} - {float(max_load)} Bags",
                    "standard_range": f"{float(min_load)} to {float(max_load)} Bags"
                })

        return anomalies

    # ===================== QUANTITY MATCHING METHODS =====================

    def quantity_anomalies(
        self, 
        records: List[Dict], 
        req_col: str, 
        obs_col: str
    ) -> List[Dict]:
        """
        Check quantity variances based on material type rules.
        
        Rules:
        - Twist-off port items: Exact match required
        - Material items: Exact match required  
        - Ingredients items: Within tolerance (±0.001)
        - Fallback: Exact match required
        """
        anomalies = []

        for row in records:
            # Try all synonym variants for each row to handle mixed-table pages
            std_parsed = None
            std_raw_col = None
            for syn in self.REQUIRED_QTY_SYNONYMS:
                if syn in row:
                    std_parsed = self.parse_quantity_with_unit(row.get(syn))
                    if std_parsed is not None:
                        std_raw_col = syn
                        break
            
            obs_parsed = None
            obs_raw_col = None
            for syn in self.OBSERVED_QTY_SYNONYMS:
                if syn in row:
                    obs_parsed = self.parse_quantity_with_unit(row.get(syn))
                    if obs_parsed is not None:
                        obs_raw_col = syn
                        break
            
            if std_parsed is None or obs_parsed is None:
                continue

            std_val, std_unit = std_parsed
            obs_val, obs_unit = obs_parsed

            item = str(row.get("Item", ""))

            # Rule 1: Twist-off port
            if "twist-off port" in item.lower():
                if not self.quantities_match(std_val, std_unit, obs_val, obs_unit):
                    anomalies.append({
                        "parameter": f"Quantity Dispensed — {row.get('Item', 'N/A')}",
                        "issue": f"Dispensed quantity ({int(obs_val)} {obs_unit}) does not match standard required quantity ({int(std_val)} {std_unit}).",
                        "observed_value": f"{int(obs_val)} {obs_unit}".strip(),
                        "expected_value": f"{int(std_val)} {std_unit}".strip(),
                        "standard_range": f"== {int(std_val)} {std_unit}".strip()
                    })
                continue

            # Rule 2: Material column exists
            if "Material" in row:
                if not self.quantities_match(std_val, std_unit, obs_val, obs_unit):
                    anomalies.append({
                        "parameter": f"Issued Qty — {row.get('Material', 'N/A')}",
                        "issue": f"Issued quantity ({float(obs_val)} {obs_unit}) does not match required standard quantity ({float(std_val)} {std_unit}).",
                        "observed_value": f"{float(obs_val)} {obs_unit}".strip(),
                        "expected_value": f"{float(std_val)} {std_unit}".strip(),
                        "standard_range": f"== {float(std_val)} {std_unit}".strip()
                    })
                continue

            # Rule 3: Ingredients column exists
            if "Ingredients" in row:
                if not self.quantities_match(std_val, std_unit, obs_val, obs_unit, self.TOLERANCE):
                    anomalies.append({
                        "parameter": f"Quantity Dispensed — {row.get('Ingredients', 'N/A')}",
                        "issue": f"Quantity dispensed ({float(obs_val)} {obs_unit}) is outside the tolerance of ±{float(self.TOLERANCE)} from standard ({float(std_val)} {std_unit}).",
                        "observed_value": f"{float(obs_val)} {obs_unit}".strip(),
                        "expected_value": f"{float(std_val)} {std_unit}".strip(),
                        "tolerance_allowed": float(self.TOLERANCE),
                        "standard_range": f"== {float(std_val)} {std_unit} (±{float(self.TOLERANCE)})".strip()
                    })
                continue

            # Fallback: exact match (unit-aware)
            if not self.quantities_match(std_val, std_unit, obs_val, obs_unit):
                label = row.get('Item') or row.get('Material') or row.get('Materials') or row.get('Ingredients') or 'N/A'
                anomalies.append({
                    "parameter": f"Quantity Dispensed — {label}",
                    "issue": f"Dispensed quantity ({float(obs_val)} {obs_unit}) does not match standard required quantity ({float(std_val)} {std_unit}).",
                    "observed_value": f"{float(obs_val)} {obs_unit}".strip(),
                    "expected_value": f"{float(std_val)} {std_unit}".strip(),
                    "standard_range": f"== {float(std_val)} {std_unit}".strip()
                })

        return anomalies

    # ===================== PAGE-LEVEL VALIDATION =====================

    def validate_page(
        self, 
        records: List[Dict], 
        trolley_spec: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Validate a single page for quantity anomalies.
        
        Args:
            records: Raw records from the page
            trolley_spec: Optional trolley spec for load validation
            
        Returns:
            List of anomaly dicts found on this page
        """
        flat = self.iter_dict_rows(records)
        if not flat:
            return []

        req_col = self.find_col(flat, self.REQUIRED_QTY_SYNONYMS)
        obs_col = self.find_col(flat, self.OBSERVED_QTY_SYNONYMS)

        # If no standard/observed columns, do only trolley check
        if not req_col or not obs_col:
            if trolley_spec:
                return self.trolley_load_anomalies(flat, trolley_spec)
            return []

        # Get quantity anomalies
        anomalies = self.quantity_anomalies(flat, req_col, obs_col)

        # Add trolley anomalies if spec provided
        if trolley_spec:
            anomalies.extend(self.trolley_load_anomalies(flat, trolley_spec))

        return anomalies

    # ===================== ENTRY POINT =====================

    def run_validation(self, bmr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main entry point for quantity variance validation.
        
        Args:
            bmr_data: Dictionary with page numbers as keys, page content as values.
                      Each page should have a "records" key with the table data.
        
        Returns:
            List of result dicts, one per page:
            {
                "page_no": str,
                "section_name": "Process Parameters",
                "check_name": "Quantity Variance Check",
                "anomaly_status": 0 or 1
            }
        """
        results = []
        self.debug_results = []  # Store debug information
        trolley_spec = None
        trolley_spec_next_page_allowed = False

        for page_no, page in bmr_data.items():
            # Check for skip condition: skip pages that check_28 runs on (based on page heading)
            headings = [str(h).strip().upper() for h in page.get("headings", [])]
            rules = [str(r).strip().upper() for r in page.get("rules_or_instructions", [])]
            all_page_text = headings + rules

            is_check_28_page = all(
                any(chk_heading in text for text in all_page_text)
                for chk_heading in self.CHECK_28_HEADINGS
            )

            if is_check_28_page:
                self.debug_results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                    "skip_reason": "Skipped — page heading belongs to check_28 (PRIMARY PACKING MATERIAL)"
                })
                results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })
                continue

            try:
                records = page.get("records", [])
                flat = self.iter_dict_rows(records)

                # Check if page has relevant data
                if not flat:
                    self.debug_results.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": "No records found on page"
                    })
                    results.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                # Find columns
                req_col = self.find_col(flat, self.REQUIRED_QTY_SYNONYMS)
                obs_col = self.find_col(flat, self.OBSERVED_QTY_SYNONYMS)
                has_trolley = any("Trolley" in row for row in flat if isinstance(row, dict))

                # Check if this page has relevant quantity data
                if not req_col and not obs_col and not has_trolley:
                    self.debug_results.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": "No quantity columns or trolley data found"
                    })
                    results.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                # -------- STEP 1: Extract trolley spec for this page --------
                new_spec = self.extract_trolley_spec(flat)

                if new_spec:
                    trolley_spec = new_spec
                    trolley_spec_next_page_allowed = True
                else:
                    if trolley_spec_next_page_allowed:
                        trolley_spec_next_page_allowed = False
                    else:
                        trolley_spec = None

                # -------- STEP 2: Validate the page --------
                anomalies = self.validate_page(records, trolley_spec)

                # -------- STEP 3: Build result --------
                if anomalies:
                    self.debug_results.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 1,
                        "columns_found": {"req_col": req_col, "obs_col": obs_col, "has_trolley": has_trolley},
                        "anomalies": anomalies
                    })
                else:
                    self.debug_results.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "columns_found": {"req_col": req_col, "obs_col": obs_col, "has_trolley": has_trolley},
                        "skip_reason": "No anomalies found (quantities matched)"
                    })
                
                results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 1 if anomalies else 0
                })

            except Exception as e:
                self.debug_results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                    "error": str(e)
                })
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
        headings = []
        
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
                    heading = kv_block[0].get("heading")
                    if heading:
                        headings.append(heading)
                    rules = kv_block[0].get("rules_or_instructions", [])
                    rules_or_instructions.extend(rules)
                elif isinstance(kv_block, dict):
                    heading = kv_block.get("heading")
                    if heading:
                        headings.append(heading)
                    rules = kv_block.get("rules_or_instructions", [])
                    rules_or_instructions.extend(rules)
        
        bmr_data[page_no] = {
            "records": records,
            "rules_or_instructions": rules_or_instructions,
            "headings": headings
        }
    
    return bmr_data


if __name__ == "__main__":
    import os
    
    # Path to the JSON file
    # json_file = os.path.join(
    #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #     "New_BMRs/Emulsion_line_AH240074_filled_master_data.json"
    # )

    json_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "New_BMRs/TS_line_(BMR-PA-040-10)_filled_master_data.json"
    )
    
    print(f"Loading JSON file: {json_file}")
    json_data = load_json_file(json_file)
    print(f"JSON loaded successfully.")
    
    bmr_data = extract_bmr_data(json_data)
    print(f"Extracted {len(bmr_data)} pages from the BMR data.")
    
    validator = QuantityVarianceValidator()
    
    print("\nRunning Quantity Variance Validation...")
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
    
    # Print all debug results
    print("\n" + "=" * 60)
    print("DEBUG RESULTS (all pages):")
    print("=" * 60)
    pp(validator.debug_results)
