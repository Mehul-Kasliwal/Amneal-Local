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
SECTION_NAME = "Equipment & Materials"          # get_check_process("check_26")
CHECK_NAME   = "RM_formula_checks"              # get_check_name("check_26")

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
    """
    Check 26: AI tool should verify potency calculation for API.
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

    def find_keyphrase_in_rules(self, obj: Any, key_phrase: str) -> bool:
        if isinstance(obj, dict):
            if "rules_or_instructions" in obj:
                rules = obj["rules_or_instructions"]
                if isinstance(rules, list):
                    for rule in rules:
                        if isinstance(rule, str) and key_phrase.lower() in rule.lower():
                            return True
            return any(self.find_keyphrase_in_rules(val, key_phrase) for val in obj.values())
        elif isinstance(obj, list):
            return any(self.find_keyphrase_in_rules(item, key_phrase) for item in obj)
        return False

    def select_pages(self, document_pages: List[Dict]) -> List[Dict]:
        selected = []
        for page_obj in document_pages:
            page_no = page_obj.get("page", "")
            content = page_obj.get("page_content", [])
            markdown_page = page_obj.get("markdown_page", "")
            if self.find_keyphrase_in_rules(content, self.KEY_PHRASE):
                selected.append({"page_no": page_no, "markdown": markdown_page})
        return selected

    # ── LLM analysis ──

    def analyze_bmr_with_llm(self, combined_text: str) -> str:
        messages = [
            {"role": "system", "content": CHECK_PROMPTS["system"]},
            {
                "role": "user",
                "content": (
                    "### TASK INSTRUCTION\n"
                    f"{CHECK_PROMPTS['user']}\n\n"
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

    # ── merge results ──

    def merge_llm_results(self, base_results: List[Dict], llm_json_str: str) -> List[Dict]:
        llm_json = json.loads(llm_json_str)
        flagged_pages = []
        for rm in llm_json.get("materials", []):
            if rm.get("anomaly_flag") == 1:
                flagged_pages.append(rm.get("page_no"))

        merged = []
        for page in base_results:
            if page.get("page_no") in flagged_pages:
                page["anomaly_status"] = 1
            merged.append(page)
        return merged

    # ── entry point ──

    def run_validation(self, document_pages: List[Dict]) -> List[Dict]:
        base_results = [
            {
                "page_no": idx,
                "section_name": SECTION_NAME,
                "check_name": CHECK_NAME,
                "anomaly_status": 0,
            }
            for idx in range(1, len(document_pages) + 1)
        ]

        selected_pages = self.select_pages(document_pages)
        if not selected_pages:
            print("No potency calculation pages found.")
            return base_results

        print(f"Found {len(selected_pages)} potency calculation pages: "
              f"{[p['page_no'] for p in selected_pages]}")

        combined_text = self.combine_selected_pages(selected_pages)
        llm_json_str = self.analyze_bmr_with_llm(combined_text)

        # Print the raw LLM response for debugging
        print("\n===== LLM RAW RESPONSE =====")
        print(llm_json_str)
        print("============================\n")

        final_results = self.merge_llm_results(base_results, llm_json_str)
        return final_results


# ─────────────────────────────────────────────────────────
# 5. Main – load JSON and run
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── UPDATE THIS PATH to your BMR JSON file ──
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

    validator = PotencyCalculationValidator()
    results = validator.run_validation(document_pages)

    # Show only pages with anomalies
    anomalies = [r for r in results if r["anomaly_status"] == 1]

    print("\n===== FINAL RESULTS =====")
    print(f"Total pages: {len(results)}")
    print(f"Pages with anomalies: {len(anomalies)}")

    if anomalies:
        print("\nAnomalous pages:")
        for a in anomalies:
            print(f"  Page {a['page_no']}: anomaly_status={a['anomaly_status']}")
    else:
        print("\nNo anomalies detected.")

    # Optionally dump all results
    # print("\nFull results:")
    # print(json.dumps(results, indent=2))
