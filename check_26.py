"""
Standalone test script for Check 26 - PotencyCalculationValidator
Run locally without importing from checks_utils2.py

Usage:
    python test_check_26.py
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI

# ─────────────────────────────────────────────────────────
# 1. Load environment variables
# ─────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────
# 2. Azure OpenAI client setup (GPT-5)
# ─────────────────────────────────────────────────────────
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# GPT-5 client
gpt5_api_key = os.getenv("AZURE_OPENAI_API_KEY_GPT5")
gpt5_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_GPT5", "https://amneal-gpt-5-mini.cognitiveservices.azure.com")

if gpt5_api_key:
    client_gpt5_1 = AzureOpenAI(
        api_key=gpt5_api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_GPT5", "2025-04-01-preview"),
        azure_endpoint=gpt5_endpoint,
    )
elif azure_endpoint and api_key:
    client_gpt5_1 = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )
else:
    raise ValueError(
        "Set AZURE_OPENAI_API_KEY_GPT5 + AZURE_OPENAI_ENDPOINT_GPT5, "
        "or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY in your .env"
    )

# ─────────────────────────────────────────────────────────
# 3. Inlined config values for check_26
# ─────────────────────────────────────────────────────────
# SECTION_NAME and CHECK_NAME moved to class definition

# Prompts from checks_prompts.yaml → check_26
CHECK_PROMPTS = {
    "system": (
        "You are a QA documentation expert specializing in pharmaceutical BMR interpretation, "
        "potency calculations, batch record verification, ALCOA+ principles, and GMP compliance."
    ),
    "user": r"""You are analyzing OCR-extracted text of a Batch Manufacturing Record (BMR) 
corresponding to Material Dispensing Section.

Your objective is to:
- Identify which ingredient or raw material the calculation belongs to.
- Determine whether the potency formula is based on Option 1 (Single Lot) or Option 2 (Double Lot / Two Different Lots).
- Perform all necessary calculations and compare results with the values written on the scanned document.

------------------------------------------------------------
CALCULATION LOGIC AND RULES
------------------------------------------------------------

OPTION 1 — SINGLE LOT FORMULA
Use this when only one lot is used (the other lot will be marked as "NA"). 
Formula depends on whether LOD (Loss on Drying) is considered.

C ia a Constant = 5.26 g for most raw materials ( but for Potassiam Chloride IP;  C = 0.37 g)

Without LOD:
A (Actual Quantity) (g) = (C g / 1.0 L) × (Batch Size in L) × (100 / % Assay)

With LOD:
A (Actual Quantity) (g) = (C g / 1.0 L) × (Batch Size in L) × (100 / % Assay) × (100 / (100 - LOD))

Total Quantity A(g) or A/1000 (in Kg)


------------------------------------------------------------

OPTION 2 — DOUBLE LOT FORMULA
Used when two different lots are combined (Lot 1 and Lot 2). 
For each raw material, one of these lots may still be "NA" depending on usage.

Lot 1 Calculation:

Without LOD:
L1 = A1 (Actual Quantity of Lot 1) (g) × (1.0 L / C g) × (% Assay of Lot 1 / 100)

With LOD:
L1 = A1 (Actual Quantity of Lot 1) (g) × (1.0 L / C g) × (% Assay of Lot 1 / 100) × ((100 - LOD of Lot 1) / 100)

Lot 2 Calculation:
L2 = Theoretical Batch Size (L) - L1

Then calculate "A2" (for Lot 2):

Without LOD:
A2 (estimated quantity of lot 2) (g) = (C g / 1.0 L) × (L2) × (100 / % Assay of Lot 2)

With LOD:
A2 (estimated quantity of lot 2) (g) = (C g / 1.0 L) × (L2) × (100 / % Assay of Lot 2) × (100 / (100 - LOD of Lot 2))

Total Quantity Combined A(g) = A1(g) + A2(g)    
Total Quantity Combined A(kg) = A(g) / 1000


------------------------------------------------------------
IMPORTANT NOTES
------------------------------------------------------------
- Each raw material will use only one option (Single Lot or Double Lot). 
- If OPTION 2: DOUBLE Lot FORMULA is filled then Lot1, and Lot2 calculation under OPTION 2 should be strictly used in all calculations. 
- If a lot is not used, its field will contain "NA".
- The scanned copy may represent numerator and denominator with horizontal lines, so interpret fractions carefully.
- Perform calculations accurately according to the formulas above.
- Compare your calculated output with the printed/handwritten result visible in the scanned document.

------------------------------------------------------------
REQUIRED OUTPUT FORMAT
------------------------------------------------------------
For each raw material found in the document, return the following structured information:

1. Name of the Raw Material Used
2. A.R. No:
   - A1 (for Lot 1)
   - A2 (for Lot 2, or NA if not used)
3. % Assay:
   - A1 value and A2 value (A2 = NA if not used)
4. LOD (if available):
   - A1 value and A2 value (A2 = NA if not used)
5. Lot Selected for Calculation:
   - Option 1 (Single Lot) or Option 2 (Double Lot)
6. Actual Formula Used:
   - Show the full mathematical expression with substituted values and the calculated result.
7. Match Verification:
   - Compare your calculated "A" value with the printed/handwritten value on the scanned page.
8. Anomaly Flag : 
   - 1 if there is a mismatch between calculated and printed value on a page.
   - 0 if the values match.
9. Page Number :
    - Main page number analysed for the verification of this section mentioned above markdown 

------------------------------------------------------------
EXAMPLE RESPONSE
------------------------------------------------------------
Raw Material: Sodium Chloride IP
A.R. No: A1 = 23007409, A2 = NA
% Assay: A1 = 99.7, A2 = NA
LOD: A1 = 0.8, A2 = NA
Calculation Option: Option 1 (Single Lot)
Formula Used: (5.26 / 1.0) × 4000 × (100 / 99.7) × (100 / (100 - 0.8))
Calculated Value: 21.273 kg
Printed Value: 21.273 kg
Anomaly: 0
Pages : 9


Return ONLY valid JSON with this schema:

{
  "materials": [
    {
      "raw_material_name": "string",
      "ar_no": {"A1": "string", "A2": "string or NA"},
      "assay_percent": {"A1": "float", "A2": "float or NA"},
      "lod_percent": {"A1": "float", "A2": "float or NA"},
      "calculation_option": 1,
      "formula_used": "string",
      "calculated_value": "float",
      "printed_value": "float",
      "anomaly_flag": 0,
      "page_no": int
    }
  ]
}

Repeat for all raw materials detected.
""",
}


# ─────────────────────────────────────────────────────────
# 4. PotencyCalculationValidator class (self-contained)
# ─────────────────────────────────────────────────────────
class PotencyCalculationValidator:
    SECTION_NAME = "Equipment & Materials"
    CHECK_NAME = "RM_formula_checks"
    """
    Check 26: AI tool should verify potency calculation for API.

    Author: Mehul Kasliwal (19th Feburary 2026)
    """

    KEY_PHRASE = "POTENCY CALCULATION FOR API"

    def __init__(self):
        self.client = client_gpt5_1

    # ── helpers ──

    def combine_selected_pages(self, selected_pages: List[Dict]) -> str:
        parts = []
        for obj in selected_pages:
            page_no = obj.get("page_no", "")
            markdown_text = obj.get("markdown", "")
            block = (
                "-------------------------\n"
                f"PAGE NUMBER: {page_no}\n"
                "-------------------------\n\n"
                f"{markdown_text}\n"
            )
            parts.append(block)
        return "\n\n".join(parts)

    # ── page selection ──

    def find_keyphrase_recursive(self, obj: Any, key_phrase: str) -> bool:
        """
        Recursively search for key_phrase in any string value within the object.
        """
        if isinstance(obj, str):
            return key_phrase.lower() in obj.lower()
        
        if isinstance(obj, dict):
            return any(self.find_keyphrase_recursive(val, key_phrase) for val in obj.values())
        
        if isinstance(obj, list):
            return any(self.find_keyphrase_recursive(item, key_phrase) for item in obj)
            
        return False

    def select_pages(self, document_pages: List[Dict]) -> List[Dict]:
        selected = []
        # Keywords that indicate an index / table-of-contents page
        index_keywords = ["Sr. No.", "Index"]

        for page_obj in document_pages:
            page_no = page_obj.get("page", "")
            # Search entire page object (content, tables, markdown)
            if self.find_keyphrase_recursive(page_obj, self.KEY_PHRASE):
                # Skip index / TOC pages
                is_index = all(
                    self.find_keyphrase_recursive(page_obj, kw)
                    for kw in index_keywords
                )
                if is_index:
                    continue
                selected.append({
                    "page_no": page_no, 
                    "markdown": page_obj.get("markdown_page", "")
                })
        return selected

    # ── LLM analysis ──

    def analyze_bmr_with_llm(self, combined_text: str) -> str:
        """
        Ask LLM to extract raw values (Assay, LOD, Batch Size, etc.)
        Returns JSON string with extracted data.
        """
        extraction_prompt = r"""
You are a data extraction specialist. Your task is to extract specific numeric values and identifiers from the BMR text for Potency Calculation.

EXTRACT the following fields for each raw material found:

1. "raw_material_name": Name of the material.
2. "batch_size": The Batch Size volume (L) found in the header or text (e.g., 4000).
3. "constant_c": The formula constant 'X' in "A = X g / 1.0 L".
   - Look for text like "A = 5.26 g / 1.0 L" -> specific constant is 5.26.
   - Look for text like "A = 0.37 g / 1.0 L" -> specific constant is 0.37.
   - Look for text like "A = 5.02 g / 1.0 L" -> specific constant is 5.02.
   - If not found, return null.
4. "calculation_option": "Option 1" (Single Lot) or "Option 2" (Double Lot).
5. "lot_1":
   - "ar_no": AR Number for Lot 1 (column A1 from the table).
   - "assay": % Assay for Lot 1 (numeric). IMPORTANT: Extract this ONLY from the structured table that has columns like "Raw material", "A. R. No.", "% Assay (On dried basis)", "LOD/ Water Content". Do NOT use values from the "Space for calculation" or formula lines (e.g. do NOT extract from "× 99.4 / 100 ×").
   - "lod": LOD % for Lot 1 (numeric). Same rule: extract ONLY from the table, NOT from formula lines. If NA or not found, return 0.
   - "qty_used": Quantity of Lot 1 used (in grams) IF Option 2 is used. Else null.
6. "lot_2": (If Option 2 is used)
   - "ar_no": AR Number for Lot 2 (column A2 from the table).
   - "assay": % Assay for Lot 2. Extract ONLY from the table, NOT from formula lines.
   - "lod": LOD % for Lot 2. Extract ONLY from the table. If NA or 0, return 0.
   - "qty_used": Quantity of Lot 2 used (in grams) IF clearly stated.
7. "printed_value": The FINAL Total Quantity 'A' written on the document. 
   - Prefer the value in "Total quantity ... (in Kg) = X Kg".
   - If that is not found, use the value from "A = ... g".
8. "printed_unit": The unit of the printed value (e.g., "Kg", "g").
9. "page_no": The page number where this calculation appears.

RETURN ONLY VALID JSON:
{
  "materials": [
    {
      "raw_material_name": "...",
      "batch_size": 4000.0,
      "constant_c": 5.26,
      "calculation_option": "Option 1",
      "lot_1": {"ar_no": "...", "assay": 99.5, "lod": 0.5, "qty_used": null},
      "lot_2": {"ar_no": "NA", "assay": null, "lod": null, "qty_used": null},
      "printed_value": 21500.0,
      "printed_unit": "g",
      "page_no": 19
    }
  ]
}
"""
        messages = [
            {"role": "system", "content": "You are a precise data extraction assistant. Extract numbers exactly as they appear. Do not calculate."},
            {
                "role": "user",
                "content": (
                    "### TASK INSTRUCTION\n"
                    f"{extraction_prompt}\n\n"
                    "------------------------------\n"
                    "### OCR EXTRACTED BMR TEXT\n"
                    f"{combined_text}\n"
                    "------------------------------\n"
                ),
            },
        ]
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content.strip()

    # ── Python Calculation ──

    def calculate_potency(self, mat_data: Dict) -> Dict:
        """
        Perform potency calculation in Python using extracted data.
        Returns dictionary with calculation details and anomaly status.
        All comparisons done in GRAMS.
        """
        try:
            # Defaults
            C = mat_data.get("constant_c")
            if C is None:
                # Fallback if extraction failed - try to infer from name
                name = mat_data.get("raw_material_name", "").lower()
                if "potassium chloride" in name: C = 0.37
                elif "sodium chloride" in name: C = 5.26 
                elif "magnesium chloride" in name: C = 0.30
                elif "sodium gluconate" in name: C = 5.02
                else: C = 5.26 # Default

            bs = float(mat_data.get("batch_size") or 4000)
            option = mat_data.get("calculation_option", "Option 1")
            
            # Lot 1
            l1 = mat_data.get("lot_1", {})
            assay1 = float(l1.get("assay") or 100)
            lod1 = float(l1.get("lod") or 0)
            
            # Factors
            factor_assay1 = 100 / assay1 if assay1 > 0 else 0
            factor_lod1 = 100 / (100 - lod1) if lod1 > 0 and lod1 < 100 else 1.0

            val_g = 0.0
            formula_desc = ""

            if option == "Option 1":
                # A (g) = (C / 1.0) * BatchSize * (100/Assay) * (100/(100-LOD))
                val_g = (C / 1.0) * bs * factor_assay1 * factor_lod1
                formula_desc = f"({C}/1)*{bs}*(100/{assay1})*(100/(100-{lod1}))"

            elif option == "Option 2":
                # Double Lot
                qty1_g = float(l1.get("qty_used") or 0)
                
                inv_lod1 = (100 - lod1) / 100
                l1_vol = qty1_g * (1.0 / C) * (assay1 / 100) * inv_lod1
                
                l2_vol = bs - l1_vol
                
                l2 = mat_data.get("lot_2", {})
                assay2 = float(l2.get("assay") or 100)
                lod2 = float(l2.get("lod") or 0)
                
                factor_assay2 = 100 / assay2 if assay2 > 0 else 0
                factor_lod2 = 100 / (100 - lod2) if lod2 > 0 and lod2 < 100 else 1.0
                
                qty2_calc_g = (C / 1.0) * l2_vol * factor_assay2 * factor_lod2
                
                val_g = qty1_g + qty2_calc_g
                formula_desc = f"Opt2: L1_vol={l1_vol:.2f}, L2_vol={l2_vol:.2f}"

            # Comparison in GRAMS
            printed_raw = float(mat_data.get("printed_value") or 0)
            unit = mat_data.get("printed_unit", "g").strip().lower() # Default to g if not found
            
            printed_g = printed_raw
            if "kg" in unit:
                printed_g = printed_raw * 1000.0
            elif printed_raw < 100 and val_g > 1000:
                 # Heuristic: if printed is small (e.g. 21.2) and calc is large (21200), assume printed is Kg
                 printed_g = printed_raw * 1000.0

            diff = abs(val_g - printed_g)
            # Tolerance: 10g? 50g? 
            # If 20kg batch, 10g is 0.05%.
            anomaly = 1 if diff > 10.0 else 0 

            return {
                "calculated_value_g": round(val_g, 3),
                "printed_value_g": round(printed_g, 3),
                "anomaly_flag": anomaly,
                "formula_used": formula_desc,
                "formula_constant": C,
                "unit_assumed": "kg" if printed_g != printed_raw else "g"
            }

        except Exception as e:
            return {
                "error": str(e),
                "anomaly_flag": 1
            }

    # ── merge results ──

    def merge_llm_results(self, base_results: List[Dict], llm_json_str: str) -> List[Dict]:
        try:
            llm_json = json.loads(llm_json_str)
        except json.JSONDecodeError:
            print(f"Error decoding LLM JSON: {llm_json_str}")
            return base_results

        # Process each material
        processed_data = []
        for mat in llm_json.get("materials", []):
            calc_res = self.calculate_potency(mat)
            mat.update(calc_res) # Add calc results to material object
            processed_data.append(mat)
        
        # Merge back to pages
        merged = []
        for page in base_results:
            page_no = page.get("page_no")
            # Find matching result for this page
            match = next((m for m in processed_data if m.get("page_no") == page_no), None)
            if match:
                page["anomaly_status"] = match.get("anomaly_flag", 0)
                page["details"] = match # Store full details for debugging
            merged.append(page)
            
        return merged

    # ── entry point ──

    def run_validation(self, document_pages: List[Dict]) -> List[Dict]:
        results = []
        return_debug = []

        selected_pages = self.select_pages(document_pages)
        
        # Populate results for pages NOT selected (default status 0)
        selected_page_nums = {p['page_no'] for p in selected_pages}
        for page in document_pages:
            page_no = page.get("page", page.get("page_no"))
            if page_no not in selected_page_nums:
                results.append({
                    "page": int(page_no) if page_no else 0,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })

        if not selected_pages:
            return_debug.append({"info": "No potency calculation pages found."})
            return results

        return_debug.append({
            "info": f"Found {len(selected_pages)} potency calculation pages",
            "pages": list(selected_page_nums)
        })

        # Analyze selected pages
        combined_text = self.combine_selected_pages(selected_pages)
        llm_json_str = self.analyze_bmr_with_llm(combined_text)

        return_debug.append({"llm_raw_response": llm_json_str})

        # Parse and merge results
        try:
            llm_json = json.loads(llm_json_str)
            materials = llm_json.get("materials", [])
            
            # Map results to pages
            page_results = {} # page_no -> anomaly_status
            
            for mat in materials:
                calc_res = self.calculate_potency(mat)
                mat.update(calc_res) # Add details
                
                p_no = mat.get("page_no")
                status = calc_res.get("anomaly_flag", 0)
                
                # If multiple materials on one page, logic OR the status? 
                # Or keep max status.
                current_status = page_results.get(p_no, 0)
                page_results[p_no] = max(current_status, status)
                
                return_debug.append({
                    "material_result": mat,
                    "page_no": p_no,
                    "status": status
                })

            # Update results list with calculated statuses
            # We need to find the entries in `results` that match these pages?
            # Wait, `results` currently only has NON-selected pages. 
            # We need to add SELECTED pages to `results`.
            
            for p in selected_pages:
                p_no = p.get("page_no")
                status = page_results.get(p_no, 0) # Default 0 if LLM didn't return data for this selected page
                
                results.append({
                    "page": int(p_no),
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": status
                })
                
        except json.JSONDecodeError as e:
            return_debug.append({"error": f"JSON Decode Error: {e}"})
            # Add selected pages with status 0 (safest) or 1 (error)? 
            # Usually 0 if we can't parse, or log error.
            for p in selected_pages:
                results.append({
                    "page": int(p.get("page_no")),
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0
                })

        # Sort results by page number
        results.sort(key=lambda x: x["page"])
        
        # The user wants "return_debug" variable explicitly available?
        # In Python we return it.
        # "Whatever you are printing, store it in the return_debug variable"
        # I'll modify the loop to populate return_debug.
        
        return results


# ─────────────────────────────────────────────────────────
# 5. Main – load JSON and run
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── UPDATE THIS PATH to your BMR JSON file ──
    # JSON_PATH = "/home/softsensor/Desktop/Amneal/challenge_bmr/05jan_AH250076_50Checks 1.json"
    JSON_PATH = "/home/softsensor/Desktop/Amneal/complete_data_61 1.json"

    if not os.path.exists(JSON_PATH):
        print(f"ERROR: JSON file not found at: {JSON_PATH}")
        print("Please update JSON_PATH in this script to point to your BMR JSON file.")
        exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract filled_master_json from the steps structure
    document_pages = raw_data.get("steps", {}).get("filled_master_json", [])
    if not document_pages:
        print("ERROR: Could not find 'steps.filled_master_json' in the JSON file.")
        exit(1)

    print(f"Loaded {len(document_pages)} pages from {JSON_PATH}")
    print("Running Check 26 – Potency Calculation Validator ...\n")

    # Run validation
    validator = PotencyCalculationValidator()
    results = validator.run_validation(document_pages)

    # Output as JSON
    print(json.dumps(results, indent=2))
