"""
Check 26 - Potency Calculation Verification (OOP Version)
==========================================================
Validates potency calculations for API ingredients in the BMR by comparing 
calculated values against printed/handwritten values using LLM analysis.
"""

import json
import os
from typing import Any, Dict, List
from openai import AzureOpenAI


class PotencyCalculationValidator:
    """
    Check 26: AI tool should verify potency calculation for API.
    ---
    Description:
    1. Find BMR pages with potency calculation rules (keyword: "POTENCY CALCULATION FOR API")
    2. Extract markdown content from selected pages
    3. Use LLM to analyze calculations and compare with printed values
    4. Report anomalies where calculated values don't match printed values
    ---
    Type: LLM-based
    ---
    Attachment Required: NO
    ---
    Author: Mehul Kasliwal (04 Feb 2026)
    """

    # ===================== CONSTANTS =====================

    SECTION_NAME = "Material Dispensing"
    CHECK_NAME = "potency_calculation_verification"
    KEY_PHRASE = "POTENCY CALCULATION FOR API"

    SYSTEM_PROMPT = """
You are a QA documentation expert specializing in pharmaceutical BMR interpretation, 
potency calculations, batch record verification, ALCOA+ principles, and GMP compliance.
"""

    ANALYSIS_PROMPT = """
You are analyzing OCR-extracted text of a Batch Manufacturing Record (BMR) 
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
"""

    # ===================== HELPER METHODS =====================

    # def md_to_clean_text(self, md_string: str) -> str:
    #     """
    #     Convert markdown → cleaned plain text (in memory).
    #     NOTE: This function is commented out as it's no longer required.
    #     """
    #     html = markdown.markdown(md_string)
    #     soup = BeautifulSoup(html, "lxml")
    # 
    #     text = soup.get_text(separator="\n")
    #     cleaned = "\n".join(
    #         line.strip() for line in text.splitlines() if line.strip()
    #     )
    # 
    #     return cleaned

    def __init__(self, api_key: str = None, api_version: str = None, azure_endpoint: str = None):
        """
        Initialize the validator with Azure OpenAI credentials.
        
        Args:
            api_key: Azure OpenAI API key
            api_version: API version (default: 2024-02-15-preview)
            azure_endpoint: Azure endpoint URL
        """
        self.client = AzureOpenAI(
            api_key=api_key or "YOUR_API_KEY",
            api_version=api_version or "2024-02-15-preview",
            azure_endpoint=azure_endpoint or "YOUR_ENDPOINT"
        )

    def combine_selected_pages(self, selected_pages: List[Dict]) -> str:
        """
        Concatenate markdown from selected pages into a single LLM-friendly string.
        Note: md_to_clean_text conversion removed as it's no longer required.
        """
        parts = []
        for obj in selected_pages:
            page_no = obj.get("page_no", "")
            markdown_text = obj.get("markdown", "")

            # Directly use the markdown text without cleaning
            block = (
                "-------------------------\n"
                f"PAGE NUMBER: {page_no}\n"
                "-------------------------\n\n"
                f"{markdown_text}\n"
            )
            parts.append(block)

        return "\n\n".join(parts)

    # ===================== PAGE SELECTION LOGIC =====================

    def find_keyphrase_in_rules(self, obj: Any, key_phrase: str) -> bool:
        """
        Recursively search for key_phrase in rules_or_instructions fields.
        """
        if isinstance(obj, dict):
            if "rules_or_instructions" in obj:
                rules = obj["rules_or_instructions"]
                if isinstance(rules, list):
                    for rule in rules:
                        if isinstance(rule, str) and key_phrase.lower() in rule.lower():
                            return True

            # Search nested fields
            return any(
                self.find_keyphrase_in_rules(val, key_phrase) for val in obj.values()
            )

        elif isinstance(obj, list):
            return any(
                self.find_keyphrase_in_rules(item, key_phrase) for item in obj
            )

        return False

    def select_pages(self, document_pages: List[Dict]) -> List[Dict]:
        """
        Select only pages that contain potency calculation rules.
        """
        selected = []

        for page_obj in document_pages:
            page_no = page_obj.get("page", "")
            content = page_obj.get("page_content", [])
            markdown_page = page_obj.get("markdown_page", "")

            if self.find_keyphrase_in_rules(content, self.KEY_PHRASE):
                selected.append({
                    "page_no": page_no,
                    "markdown": markdown_page
                })

        return selected

    # ===================== LLM ANALYSIS =====================

    def analyze_bmr_with_llm(self, combined_text: str) -> str:
        """
        Send merged potency text to Azure OpenAI for analysis.
        Returns JSON string.
        """
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
    ### TASK INSTRUCTION
    {self.ANALYSIS_PROMPT}
    
    ------------------------------
    ### OCR EXTRACTED BMR TEXT
    {combined_text}
    ------------------------------
    """
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content.strip()

    # ===================== RESULT MERGING =====================

    def merge_llm_results(self, base_results: List[Dict], llm_json_str: str) -> List[Dict]:
        """
        Merge anomaly flags from LLM JSON into base results.
        """
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

    # ===================== ENTRY POINT =====================

    def run_validation(self, document_pages: List[Dict]) -> List[Dict]:
        """
        Main entry point for potency calculation validation.

        Args:
            document_pages: List of page dicts from BMR JSON
                           Expected format: [{"page": 1, "page_content": [...], "markdown_page": "..."}, ...]

        Returns:
            List of result dicts with keys: page_no, section_name, check_name, anomaly_status
        """
        # Build base results structure for all pages
        base_results = [
            {
                "page_no": idx,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": 0
            }
            for idx in range(1, len(document_pages) + 1)
        ]

        # Select pages with potency calculation rules
        selected_pages = self.select_pages(document_pages)
        
        if not selected_pages:
            print("No potency calculation pages found.")
            return base_results

        print(f"Found {len(selected_pages)} potency calculation pages: {[p['page_no'] for p in selected_pages]}")

        # Combine selected pages into LLM-ready text
        combined_text = self.combine_selected_pages(selected_pages)

        # Analyze with LLM
        llm_json_str = self.analyze_bmr_with_llm(combined_text)

        # Merge anomaly flags
        final_results = self.merge_llm_results(base_results, llm_json_str)
        
        return final_results


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
    # Load the complete_data_61 JSON file
    ocr_json_path = "/home/softsensor/Desktop/Amneal/complete_data_61 1.json"
    
    print("Running Check 26 - Potency Calculation Verification...")
    
    # Load the full JSON
    full_data = load_json_file(ocr_json_path)
    
    # Extract filled_master_json from steps
    steps = full_data.get("steps", {})
    filled_master_json = steps.get("filled_master_json", [])
    
    print(f"Loaded {len(filled_master_json)} pages from filled_master_json")
    
    # Run validation with Azure OpenAI credentials
    validator = PotencyCalculationValidator(
        api_key=os.environ.get("AZURE_GPT5_MINI_API_KEY", ""),
        api_version=os.environ.get("AZURE_GPT5_MINI_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.environ.get("AZURE_GPT5_MINI_ENDPOINT", "https://amneal-gpt-5-mini.cognitiveservices.azure.com")
    )
    results = validator.run_validation(filled_master_json)
    
    # Output as JSON
    print(json.dumps(results, indent=2))
