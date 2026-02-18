import json
import os
import re
from typing import Union, List, Dict, Any
from openai import AzureOpenAI


class SampleQuantityCalculationValidator:
    """
    Check 42: AI tool should verify sampling quantities and conversions to kilograms.
    ---
    Description:
    1. Locate entries under Manufacturing Process → Sampling sections
    2. Verify sampling quantities and conversions to kilograms
    3. Use canonical formula: Sample_kg = (Sample_mL × Density_g_per_mL) / 1000
    4. Flag anomalies if |written_kg − calc_kg| > 0.001
    ---
    Type: LLM-based
    ---
    Attachment Required: NO
    ---
    Author: Yogesh (Converted to OOP format 30 Jan 2026)
    """

    # ===================== CONSTANTS =====================

    SECTION_NAME = "Document Compliance"
    CHECK_NAME = "check sample quantity and calculate the total sample quantity"
    CHUNK_SIZE_PAGES = 4

    SYSTEM_PROMPT = """
You are a BMR calculation verification assistant.

Your responsibility is to VERIFY MATHEMATICAL CALCULATIONS in sampling quantity conversions
from Manufacturing Process → Sampling sections in Batch Manufacturing Records.

IMPORTANT SCOPE RULES:
- You MUST verify EVERY calculation on EVERY page that contains sampling formulas
- You MUST NOT skip any page with "Total sample (In Kg)" expressions
- You MUST perform the actual arithmetic calculation yourself and compare

────────────────────────────────────
WHAT TO LOOK FOR
────────────────────────────────────
Locate expressions following this pattern:
"Total sample (In Kg): <mL> mL × (Bulk density <density> g/mL) / 1000 = <written_kg> Kg"

Also look for variations like:
- "Total sample (In Kg): 100 mL X (Bulk density 0.99 g/mL) / 1000 = 0.9 Kg"
- "Process Loss: 00 mL X (Bulk density 0.99 g/mL) / 1000 = 00 Kg"
- Any expression with mL × density / 1000 = result

Accept these formatting variations:
- × / x / X for multiplication
- /1000 or ÷1000 for division
- Units: mL/ml/ML, g/mL, g per mL
- OCR noise, spacing errors, line breaks

────────────────────────────────────
CANONICAL FORMULA (MUST USE)
────────────────────────────────────
calc_kg = (Sample_mL × Density_g_per_mL) / 1000

STEP-BY-STEP CALCULATION EXAMPLE:
Given: 100 mL × (Bulk density 0.99 g/mL) / 1000 = 0.9 Kg

Step 1: Extract values
- Sample_mL = 100
- Density_g_per_mL = 0.99

Step 2: Calculate
- calc_kg = (100 × 0.99) / 1000
- calc_kg = 99 / 1000
- calc_kg = 0.099

Step 3: Round to 3 decimals
- calc_kg = 0.099 Kg

Step 4: Compare with written value
- written_kg = 0.9 Kg
- |written_kg − calc_kg| = |0.9 − 0.099| = 0.801 Kg

Step 5: Apply tolerance (±0.001 kg)
- 0.801 > 0.001 → ANOMALY DETECTED!

────────────────────────────────────
ANOMALY CONDITIONS
────────────────────────────────────
Set anomaly_status = 1 if ANY of the following occur on a page:

1. CALCULATION MISMATCH: |written_kg − calc_kg| > 0.001 kg
   Example: "100 mL × 0.99 g/mL / 1000 = 0.9 Kg" → calc = 0.099, written = 0.9 → ANOMALY

2. MISSING VALUES: Sampling expression exists but mL or density is missing/invalid
   Exception: If values are explicitly "NA" or "00", this is acceptable (no anomaly)

3. INVALID NUMBERS: Values cannot be parsed as numbers (except "NA" which is acceptable)

Set anomaly_status = 0 if:
- Page has no sampling calculations
- Calculation is correct within ±0.001 kg tolerance
- Values are explicitly marked as "NA" or "00" (intentionally empty)

────────────────────────────────────
COMMON ERRORS TO CATCH
────────────────────────────────────
- Missing a decimal point: 0.99 → written as 0.9 (should be 0.099)
- Off by factor of 10: 0.099 written as 0.9
- Arithmetic errors in multiplication or division

────────────────────────────────────
OUTPUT CONTRACT (STRICT)
────────────────────────────────────
Return ONE JSON array with ONE object PER OCR PAGE, preserving input order:

{
  "page_no": <integer>,
  "section_name": "Document Compliance",
  "check_name": "check sample quantity and calculate the total sample quantity",
  "anomaly_status": 0 or 1
}

Rules:
- One element per page in the OCR input (in order)
- anomaly_status = 1 if any calculation error is detected on that page
- anomaly_status = 0 otherwise
- No prose, no explanations, no extra keys
"""

    TASK_DESC = """
From the OCR markdown, perform calculation verification:

STEP 1: Scan EACH page for sampling calculation expressions
- Look for: "Total sample (In Kg): <mL> mL × (Bulk density <density> g/mL) / 1000 = <written_kg> Kg"
- Also check for "Process Loss" expressions with same formula pattern

STEP 2: For EACH calculation found, perform these substeps:
a) Extract: Sample_mL, Density_g_per_mL, written_kg
b) Calculate: calc_kg = (Sample_mL × Density_g_per_mL) / 1000
c) Round calc_kg to 3 decimals
d) Compare: |written_kg − calc_kg|

STEP 3: Apply tolerance check
- If |written_kg − calc_kg| > 0.001 → anomaly_status = 1
- If values are "NA" or "00" → anomaly_status = 0 (acceptable)
- If no sampling calculation on page → anomaly_status = 0

CRITICAL VERIFICATION EXAMPLE:
Input: "Total sample (In Kg): 100 mL X (Bulk density 0.99 g/mL) / 1000 = 0.9 Kg"
- Sample_mL = 100
- Density = 0.99
- calc_kg = (100 × 0.99) / 1000 = 0.099 Kg
- written_kg = 0.9 Kg
- Difference = |0.9 - 0.099| = 0.801 Kg > 0.001 → ANOMALY!

Return the JSON array with anomaly_status set accordingly for each page.
"""

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

    # ===================== HELPER METHODS =====================

    def _load_pages(self, json_like: Union[str, list, dict]) -> List[Dict[str, Any]]:
        """Load pages from JSON file path or object."""
        if isinstance(json_like, str):
            with open(json_like, "r", encoding="utf-8") as f:
                pages = json.load(f)
        elif isinstance(json_like, (list, dict)):
            pages = json_like
        else:
            raise TypeError("json_path must be a file path (str) or a loaded JSON object (dict/list).")

        if isinstance(pages, list):
            return pages
        if isinstance(pages, dict):
            if all(isinstance(k, str) and k.isdigit() for k in pages.keys()):
                ordered_keys = sorted(pages.keys(), key=lambda k: int(k))
                return [pages[k] for k in ordered_keys]
            for key in ("pages", "data", "results"):
                if isinstance(pages.get(key), list):
                    return pages[key]
            return [pages]
        raise ValueError("Loaded JSON is not a list/dict of pages.")

    def _page_block(self, page_no: int, page: Dict[str, Any]) -> str:
        """Format a page for the LLM prompt."""
        md = page.get("markdown_page", "") or ""
        return f"--- PAGE {page_no} START ---\n{md}\n--- PAGE {page_no} END ---"

    def _default_rows_for_slice(self, start_idx: int, count: int, anomaly: int = 0) -> List[Dict[str, Any]]:
        """Generate default result rows for a slice of pages."""
        return [
            {
                "page_no": start_idx + i + 1,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": 1 if anomaly else 0,
            }
            for i in range(count)
        ]

    # ===================== VALIDATION =====================

    def run_validation(self, json_path_or_obj: Union[str, list, dict]) -> List[Dict[str, Any]]:
        """
        Main entry point for sample quantity calculation validation.

        Args:
            json_path_or_obj: Path to JSON file or loaded JSON object containing BMR pages

        Returns:
            List of result dicts with keys: page_no, section_name, check_name, anomaly_status
        """
        all_pages = self._load_pages(json_path_or_obj)
        if not all_pages:
            return []

        results: List[Dict[str, Any]] = []
        total_pages = len(all_pages)
        chunk = max(1, int(self.CHUNK_SIZE_PAGES))

        for start in range(0, total_pages, chunk):
            end = min(start + chunk, total_pages)
            slice_pages = all_pages[start:end]
            
            # Build markdown content
            blocks = []
            for offset, p in enumerate(slice_pages):
                absolute_page_no = start + offset + 1
                blocks.append(self._page_block(absolute_page_no, p))
            markdown_content = "\n\n".join(blocks)

            # Call model
            response = self.client.chat.completions.create(
                model="gpt-5",  # or your deployment name
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"OCR MARKDOWN:\n{markdown_content}\n\nTask:\n{self.TASK_DESC}"},
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            try:
                arr = json.loads(content)
                if isinstance(arr, dict):
                    arr = arr.get("data", arr.get("results", []))
            except:
                arr = []
            
            # Normalize results
            for idx in range(len(slice_pages)):
                item = arr[idx] if idx < len(arr) else {}
                results.append({
                    "page_no": start + idx + 1,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 1 if item.get("anomaly_status", 0) else 0,
                })

        return results


# =====================================================
# ---------------- MAIN RUNNER ------------------------
# =====================================================
if __name__ == "__main__":
    from extract_data import get_filled_master_json
    
    # Load OCR data from the complete_data JSON file
    complete_data_path = "/home/softsensor/Desktop/Amneal/yog_checks/complete_data_61 1.json"
    
    print("Running Check 42 - Sample Quantity Calculation...")
    
    # Extract the filled_master_json from the complete data
    filled_master_json = get_filled_master_json(complete_data_path)
    print(f"Loaded {len(filled_master_json)} pages from filled_master_json")
    
    # Run the sample quantity check using the extracted page data
    validator = SampleQuantityCalculationValidator(
        api_key=os.environ.get("AZURE_GPT5_MINI_API_KEY", ""),
        api_version=os.environ.get("AZURE_GPT5_MINI_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.environ.get("AZURE_GPT5_MINI_ENDPOINT", "https://amneal-gpt-5-mini.cognitiveservices.azure.com")
    )
    results = validator.run_validation(filled_master_json)
    print(json.dumps(results, indent=2))
