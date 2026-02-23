import json
import os
import re
from typing import Union, List, Dict, Any, Tuple, Optional
from openai import AzureOpenAI


class SampleQuantityCalculationValidator:
    """
    Check 42: AI tool should verify sampling quantities and conversions to kilograms.
    ---
    Description:
    1. Recursively search ALL OCR data (page_content tables, kv_text_blocks, markdown_page)
       for sampling formula expressions
    2. Extract mL, density, and written_kg values using regex
    3. Verify conversions using Python: calc_kg = (mL × density) / 1000
    4. Flag anomaly if |written_kg − calc_kg| > 0.001
    5. Fall back to LLM only when regex extraction fails on a page with sampling content
    ---
    Type: Hybrid (Python extraction + computation, LLM fallback)
    ---
    Attachment Required: NO
    ---
    Author: Mehul Kasliwal (30 Jan 2026)
    """

    # ===================== CONSTANTS =====================

    SECTION_NAME = "Document Compliance"
    CHECK_NAME = "check sample quantity and calculate the total sample quantity"
    TOLERANCE = 0.001  # kg

    # Regex to match sampling formula expressions.
    # Handles OCR variations: ×/X/x/*, spacing, optional "Bulk density" label, etc.
    # Groups: (1) mL value, (2) density value, (3) written kg value
    FORMULA_REGEX = re.compile(
        r'(\d+(?:\.\d+)?|NA|na|Na|00)\s*'        # (1) mL value or NA/00
        r'm[Ll]\s*'                                # unit: mL
        r'[×xX\*]\s*'                             # multiplication sign
        r'\(?'                                     # optional opening paren
        r'\s*(?:[Bb]ulk\s*[Dd]ensity\s*)?'         # optional "Bulk density" label
        r'(\d+(?:\.\d+)?)\s*'                      # (2) density value
        r'g\s*/\s*m[Ll]'                           # unit: g/mL
        r'\s*\)?'                                  # optional closing paren
        r'\s*/\s*1000\s*'                          # / 1000
        r'=\s*'                                    # =
        r'(\d+(?:\.\d+)?|NA|na|Na|00)\s*'          # (3) written kg value or NA/00
        r'[Kk]g',                                  # unit: Kg
        re.IGNORECASE
    )

    # Broader regex to detect pages that MIGHT contain sampling formulas
    # (even if the full formula regex doesn't match due to OCR noise)
    SAMPLING_HINT_REGEX = re.compile(
        r'total\s*sample.*?(?:in\s*kg|kg)|'
        r'process\s*loss.*?m[Ll].*?kg|'
        r'm[Ll]\s*[×xX\*].*?g\s*/\s*m[Ll].*?1000',
        re.IGNORECASE | re.DOTALL
    )

    # Minimal LLM prompt — extraction only, no math
    LLM_EXTRACTION_PROMPT = """
From the following OCR text, extract ALL sampling formula expressions.

Look for patterns like:
"<mL> mL × (Bulk density <density> g/mL) / 1000 = <result> Kg"

For EACH formula found, extract these three numbers:
- sample_mL: the value in mL (number before "mL")
- density: the density value (number before "g/mL")
- written_kg: the result (number before "Kg")

If a value is "NA", "00", or missing, use the string "NA".

Return STRICT JSON only:
{
  "formulas": [
    {"sample_mL": <number_or_"NA">, "density": <number>, "written_kg": <number_or_"NA">}
  ]
}

If no sampling formulas are found, return: {"formulas": []}

OCR TEXT:
"""

    def __init__(self, api_key: str = None, api_version: str = None, azure_endpoint: str = None):
        """
        Initialize the validator with Azure OpenAI credentials.

        Args:
            api_key: Azure OpenAI API key
            api_version: API version
            azure_endpoint: Azure endpoint URL
        """
        self.client = AzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_GPT5_MINI_API_KEY", ""),
            api_version=api_version or os.environ.get("AZURE_GPT5_MINI_API_VERSION", "2025-04-01-preview"),
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_GPT5_MINI_ENDPOINT", "https://amneal-gpt-5-mini.cognitiveservices.azure.com"),
        )

    # ===================== HELPER: LOAD PAGES =====================

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

    # ===================== RECURSIVE TEXT EXTRACTION =====================

    def _collect_all_text(self, obj: Any) -> List[str]:
        """
        Recursively collect ALL string values from any nested structure
        (dicts, lists, strings). This searches page_content, markdown_page,
        additional_keys, and any other field.
        """
        texts = []
        if isinstance(obj, str):
            if obj.strip():
                texts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                texts.extend(self._collect_all_text(v))
        elif isinstance(obj, list):
            for v in obj:
                texts.extend(self._collect_all_text(v))
        return texts

    # ===================== REGEX-BASED DETECTION & EXTRACTION =====================

    def _has_sampling_content(self, all_text: List[str]) -> bool:
        """
        Check if any text in the page hints at sampling formula content.
        Uses a broad regex to catch pages that MIGHT have formulas.
        """
        combined = " ".join(all_text)
        return bool(self.SAMPLING_HINT_REGEX.search(combined))

    def _extract_formulas_regex(self, all_text: List[str]) -> List[Tuple[str, str, str]]:
        """
        Extract (sample_mL, density, written_kg) tuples from all text on a page
        using the formula regex. Returns raw string values.
        """
        formulas = []
        for text in all_text:
            for match in self.FORMULA_REGEX.finditer(text):
                ml_str = match.group(1)
                density_str = match.group(2)
                kg_str = match.group(3)
                formulas.append((ml_str, density_str, kg_str))
        return formulas

    # ===================== PYTHON-BASED VERIFICATION =====================

    def _is_na_or_zero(self, value: str) -> bool:
        """Check if a value is NA, 00, or empty (intentionally blank)."""
        v = value.strip().upper()
        return v in ("NA", "N/A", "00", "0", "0.0", "")

    def _verify_calculation(self, ml_str: str, density_str: str, kg_str: str) -> Tuple[int, Optional[Dict]]:
        """
        Verify a single sampling formula calculation in Python.

        Returns:
            (anomaly_status, detail_dict)
            anomaly_status: 0 if correct or NA, 1 if mismatch
            detail_dict: calculation details for debug output
        """
        detail = {
            "raw_mL": ml_str,
            "raw_density": density_str,
            "raw_written_kg": kg_str,
        }

        # If mL or kg is NA/00 → acceptable, no anomaly
        if self._is_na_or_zero(ml_str) or self._is_na_or_zero(kg_str):
            detail["status"] = "NA/zero values — acceptable"
            return 0, detail

        try:
            sample_ml = float(ml_str)
            density = float(density_str)
            written_kg = float(kg_str)
        except (ValueError, TypeError):
            detail["status"] = "Could not parse numbers"
            return 1, detail  # Invalid numbers → anomaly

        calc_kg = round((sample_ml * density) / 1000, 3)
        diff = abs(written_kg - calc_kg)

        detail["parsed_mL"] = sample_ml
        detail["parsed_density"] = density
        detail["parsed_written_kg"] = written_kg
        detail["calculated_kg"] = calc_kg
        detail["difference"] = round(diff, 6)

        if diff > self.TOLERANCE:
            detail["status"] = f"ANOMALY: |{written_kg} - {calc_kg}| = {diff:.6f} > {self.TOLERANCE}"
            return 1, detail
        else:
            detail["status"] = f"OK: |{written_kg} - {calc_kg}| = {diff:.6f} ≤ {self.TOLERANCE}"
            return 0, detail

    # ===================== LLM FALLBACK =====================

    def _extract_formulas_with_llm(self, all_text: List[str]) -> List[Tuple[str, str, str]]:
        """
        Fallback: send page text to LLM to extract the three numbers.
        Used only when regex fails but sampling content is detected.
        """
        # Combine text, truncate to avoid huge prompts
        combined = "\n".join(all_text)[:4000]

        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a data extraction specialist. Extract numbers exactly as they appear. Do NOT perform any calculations."},
                    {"role": "user", "content": self.LLM_EXTRACTION_PROMPT + combined},
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            formulas = result.get("formulas", [])

            extracted = []
            for f in formulas:
                ml = str(f.get("sample_mL", "NA"))
                density = str(f.get("density", ""))
                kg = str(f.get("written_kg", "NA"))
                if density:  # Need at least density to be valid
                    extracted.append((ml, density, kg))
            return extracted
        except Exception:
            return []

    # ===================== MAIN VALIDATION =====================

    def run_validation(self, json_path_or_obj: Union[str, list, dict]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Main entry point for sample quantity calculation validation.

        Architecture:
        1. For each page, recursively collect ALL text from all OCR fields
        2. Check if page has sampling content (broad regex)
           → No match: anomaly_status = 0, skip. NO LLM call.
        3. Try regex extraction of (mL, density, written_kg)
           → Success: Python verifies math → set anomaly_status
           → Fail: LLM fallback to extract 3 numbers → Python verifies
        4. Return (results, return_debug) tuple

        Args:
            json_path_or_obj: Path to JSON file or loaded JSON object

        Returns:
            Tuple of (results, return_debug):
            - results: List of {'page_no', 'section_name', 'check_name', 'anomaly_status'}
            - return_debug: List of detailed debug dicts per page
        """
        all_pages = self._load_pages(json_path_or_obj)
        if not all_pages:
            return [], []

        results: List[Dict[str, Any]] = []
        return_debug: List[Dict[str, Any]] = []

        for idx, page in enumerate(all_pages):
            page_no = page.get("page", idx + 1)
            anomaly = 0

            # Step 1: Recursively collect ALL text from every part of the OCR data
            all_text = self._collect_all_text(page)

            # Step 2: Quick check — does this page have any sampling content?
            if not self._has_sampling_content(all_text):
                results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                })
                return_debug.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                    "extraction_method": "skipped",
                    "skip_reason": "No sampling content detected on page",
                })
                continue

            # Step 3: Try regex extraction
            formulas = self._extract_formulas_regex(all_text)
            extraction_method = "regex"

            # Step 4: If regex found nothing, try LLM fallback
            if not formulas:
                formulas = self._extract_formulas_with_llm(all_text)
                extraction_method = "llm_fallback"

            # Step 5: If still no formulas found, mark as no anomaly
            if not formulas:
                results.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                })
                return_debug.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                    "extraction_method": extraction_method,
                    "skip_reason": "Sampling hint detected but no formula could be extracted",
                })
                continue

            # Step 6: Verify each formula in Python
            page_anomaly = 0
            formula_details = []
            for ml_str, density_str, kg_str in formulas:
                status, detail = self._verify_calculation(ml_str, density_str, kg_str)
                detail["extraction_method"] = extraction_method
                formula_details.append(detail)
                if status == 1:
                    page_anomaly = 1

            anomaly = page_anomaly
            results.append({
                "page_no": page_no,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": anomaly,
            })
            return_debug.append({
                "page_no": page_no,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": anomaly,
                "extraction_method": extraction_method,
                "formulas_found": len(formulas),
                "formula_details": formula_details,
            })

        return results, return_debug


# =====================================================
# ---------------- MAIN RUNNER ------------------------
# =====================================================
if __name__ == "__main__":
    from extract_data import get_filled_master_json

    # Load OCR data from the complete_data JSON file
    complete_data_path = "/home/softsensor/Des ktop/Amneal/yog_checks/complete_data_61 1.json"

    print("Running Check 42 - Sample Quantity Calculation (Optimized: Python-first)...")

    # Extract the filled_master_json from the complete data
    filled_master_json = get_filled_master_json(complete_data_path)
    print(f"Loaded {len(filled_master_json)} pages from filled_master_json")

    # Run the sample quantity check
    validator = SampleQuantityCalculationValidator(
        api_key=os.environ.get("AZURE_GPT5_MINI_API_KEY", ""),
        api_version=os.environ.get("AZURE_GPT5_MINI_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.environ.get("AZURE_GPT5_MINI_ENDPOINT", "https://amneal-gpt-5-mini.cognitiveservices.azure.com"),
    )
    results, return_debug = validator.run_validation(filled_master_json)

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))

    print("\n=== DEBUG INFO ===")
    print(json.dumps(return_debug, indent=2, default=str))

    # Summary
    total = len(results)
    anomalies = sum(1 for r in results if r["anomaly_status"] == 1)
    regex_pages = sum(1 for d in return_debug if d.get("extraction_method") == "regex")
    llm_pages = sum(1 for d in return_debug if d.get("extraction_method") == "llm_fallback")
    skipped_pages = sum(1 for d in return_debug if d.get("extraction_method") == "skipped")

    print(f"\n=== SUMMARY ===")
    print(f"Total pages: {total}")
    print(f"Anomalies found: {anomalies}")
    print(f"Pages processed by regex: {regex_pages}")
    print(f"Pages processed by LLM fallback: {llm_pages}")
    print(f"Pages skipped (no sampling content): {skipped_pages}")
