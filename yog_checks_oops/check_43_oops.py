import json
import os
import re
from typing import Union, List, Dict, Any, Tuple, Optional
from openai import AzureOpenAI


class SampleQuantityVerificationValidator:
    """
    Check 43: AI tool should verify actual sample quantity with standard sample quantity and reconcile.
    ---
    Description:
    1. Search ALL OCR data for sampling formulas and compute total calc_kg using regex.
    2. Search ALL OCR data for Reconciliation tables containing "Sample of bulk solution" and extract recon_kg using LLM.
    3. Verify that total calc_kg matches recon_kg within tolerance (±0.001 kg) in Python.
    4. Flag anomalies ONLY on reconciliation pages where mismatches are detected.
    5. Fall back to LLM for sampling formulas only when regex extraction fails on a page with relevant content.
    ---
    Type: Hybrid (Python extraction + computation, LLM extraction fallback/recon computation)
    ---
    Attachment Required: NO
    ---
    Author: Mehul Kasliwal (30 Jan 2026)
    """

    # ===================== CONSTANTS =====================
    SECTION_NAME = "Document Compliance"
    CHECK_NAME = "verified actual sample quantity with standard sample quantity and reconcile the sample quantity"
    TOLERANCE = 0.001  # kg

    # Regex to match sampling formula expressions.
    FORMULA_REGEX = re.compile(
        r'(\d+(?:\.\d+)?|NA|na|Na|00)\s*'
        r'm[Ll]\s*'
        r'[×xX\*]\s*'
        r'\(?'
        r'\s*(?:[Bb]ulk\s*[Dd]ensity\s*)?'
        r'(\d+(?:\.\d+)?)\s*'
        r'g\s*/\s*m[Ll]'
        r'\s*\)?'
        r'\s*/\s*1000\s*'
        r'=\s*'
        r'(\d+(?:\.\d+)?|NA|na|Na|00)\s*'
        r'[Kk]g',
        re.IGNORECASE
    )

    # Broader regex to detect pages that MIGHT contain sampling formulas
    SAMPLING_HINT_REGEX = re.compile(
        r'total\s*sample.*?(?:in\s*kg|kg)|'
        r'process\s*loss.*?m[Ll].*?kg|'
        r'm[Ll]\s*[×xX\*].*?g\s*/\s*m[Ll].*?1000',
        re.IGNORECASE | re.DOTALL
    )

    # Regex to detect reconciliation page and value
    RECON_HINT_REGEX = re.compile(
        r'sample(?:s)?\s*of\s*bulk\s*solution',
        re.IGNORECASE
    )

    # Regex to directly extract the reconciliation quantity
    RECON_REGEX = re.compile(
        r'sample(?:s)?\s*of\s*bulk\s*solution[^\d\n]*?(\d+(?:\.\d+)?|NA|na|Na|00)',
        re.IGNORECASE
    )

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

    LLM_RECON_EXTRACTION_PROMPT = """
From the following OCR text, locate the Reconciliation table and extract the recorded "Sample of bulk solution" quantity.

Look for a row referring to "Sample of bulk solution". Examples:
- "QC Samples of bulk solution (from manufacturing vessel) (B)" -> value follows (e.g. 0.845)
- "QC Samples of bulk solution after volume makeup (from manufacturing vessel)"
- "Total sample of bulk solution after recirculation"

Extract ONLY the numerical quantity in kg.

Return STRICT JSON only:
{
  "recon_kg": <number_or_"NA">
}

If no such value exists, return: {"recon_kg": null}

OCR TEXT:
"""

    def __init__(self, api_key: str = None, api_version: str = None, azure_endpoint: str = None):
        """
        Initialize the validator with Azure OpenAI credentials.
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
        Recursively collect ALL string values from any nested structure.
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

    # ===================== REGEX CHECKS =====================

    def _has_sampling_content(self, all_text: List[str]) -> bool:
        combined = " ".join(all_text)
        return bool(self.SAMPLING_HINT_REGEX.search(combined))

    def _has_recon_content(self, all_text: List[str]) -> bool:
        combined = " ".join(all_text)
        return bool(self.RECON_HINT_REGEX.search(combined))

    def _extract_formulas_regex(self, all_text: List[str]) -> List[Tuple[str, str, str]]:
        formulas = []
        for text in all_text:
            for match in self.FORMULA_REGEX.finditer(text):
                ml_str = match.group(1)
                density_str = match.group(2)
                kg_str = match.group(3)
                formulas.append((ml_str, density_str, kg_str))
        return formulas

    def _is_na_or_zero(self, value: Any) -> bool:
        if value is None:
            return True
        v = str(value).strip().upper()
        return v in ("NA", "N/A", "00", "0", "0.0", "", "NONE", "NULL")

    # ===================== LLM FALLBACK =====================

    def _extract_formulas_with_llm(self, all_text: List[str]) -> List[Tuple[str, str, str]]:
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
                if density:
                    extracted.append((ml, density, kg))
            return extracted
        except Exception:
            return []

    def _extract_recon_with_llm(self, all_text: List[str]) -> Optional[str]:
        combined = "\n".join(all_text)[:4000]
        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a data extraction specialist. Extract the requested quantity exactly as it appears."},
                    {"role": "user", "content": self.LLM_RECON_EXTRACTION_PROMPT + combined},
                ],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            recon_val = result.get("recon_kg")
            if recon_val is not None:
                return str(recon_val)
            return None
        except Exception:
            return None

    def _extract_recon_regex(self, all_text: List[str]) -> Optional[str]:
        for text in all_text:
            match = self.RECON_REGEX.search(text)
            if match:
                return match.group(1)
        return None

    # ===================== MAIN VALIDATION =====================

    def run_validation(self, json_path_or_obj: Union[str, list, dict]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Main entry point for hybrid check 43 logic.
        """
        all_pages = self._load_pages(json_path_or_obj)
        if not all_pages:
            return [], []

        results: List[Dict[str, Any]] = []
        return_debug: List[Dict[str, Any]] = []

        total_calc_kg = 0.0
        any_sampling_found = False
        sampling_error = False
        sampling_details = []

        # Pass 1: Compute total calculated sampling kg from all pages
        for idx, page in enumerate(all_pages):
            page_no = page.get("page", idx + 1)
            all_text = self._collect_all_text(page)
            
            if self._has_sampling_content(all_text):
                formulas = self._extract_formulas_regex(all_text)
                method = "regex"
                if not formulas:
                    formulas = self._extract_formulas_with_llm(all_text)
                    method = "llm_fallback"
                
                if formulas:
                    any_sampling_found = True
                    for ml_str, density_str, kg_str in formulas:
                        if self._is_na_or_zero(ml_str) or self._is_na_or_zero(kg_str):
                            sampling_details.append({"page": page_no, "status": "NA/zero", "method": method})
                        else:
                            try:
                                ml = float(ml_str)
                                density = float(density_str)
                                calc_kg = round((ml * density) / 1000, 3)
                                total_calc_kg += calc_kg
                                sampling_details.append({
                                    "page": page_no, 
                                    "calc_kg": calc_kg, 
                                    "method": method, 
                                    "raw": (ml_str, density_str, kg_str)
                                })
                            except (ValueError, TypeError):
                                sampling_error = True
                                sampling_details.append({
                                    "page": page_no, 
                                    "status": "parse_error", 
                                    "method": method, 
                                    "raw": (ml_str, density_str, kg_str)
                                })
                else:
                    sampling_error = True
                    sampling_details.append({"page": page_no, "status": "hint_but_no_formula", "method": method})

        total_calc_kg = round(total_calc_kg, 3)

        # Pass 2: Verify reconciliation pages
        for idx, page in enumerate(all_pages):
            page_no = page.get("page", idx + 1)
            all_text = self._collect_all_text(page)
            
            anomaly = 0
            recon_method = "skipped"
            recon_val = None
            diff = None
            skip_reason = ""
            
            if self._has_recon_content(all_text):
                recon_val = self._extract_recon_regex(all_text)
                recon_method = "regex"
                
                if not recon_val:
                    recon_val = self._extract_recon_with_llm(all_text)
                    recon_method = "llm_fallback"
                
                if recon_val is not None and recon_val != "NA":
                    try:
                        r_kg = float(recon_val)
                        diff = round(abs(r_kg - total_calc_kg), 6)
                        if diff > self.TOLERANCE:
                            anomaly = 1
                            skip_reason = f"Mismatch: recon {r_kg} vs calc {total_calc_kg} (diff {diff:.4f})"
                        else:
                            skip_reason = f"Match: recon {r_kg} vs calc {total_calc_kg}"
                    except (ValueError, TypeError):
                        anomaly = 1
                        skip_reason = f"Could not parse recon_kg '{recon_val}'"
                elif recon_val == "NA" or str(recon_val).strip() == "":
                    skip_reason = "Recon value is NA"
                else:
                    anomaly = 1
                    skip_reason = "Recon hint found but no valid recon_kg extracted"
                    
                # Ensure sampling calculation was successfully found without errors
                if anomaly == 0 and recon_val is not None and recon_val != "NA":
                    if not any_sampling_found:
                        anomaly = 1
                        skip_reason = "Recon value found but NO sampling calculations found in document"
                    elif sampling_error:
                        anomaly = 1
                        skip_reason = "Recon value found but sampling calculations had parse errors or missing data"

            else:
                skip_reason = "No recon hint detected"

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
                "recon_method": recon_method,
                "recon_val": recon_val,
                "total_calc_kg": total_calc_kg,
                "diff": diff,
                "reason": skip_reason,
                "sampling_details": sampling_details if recon_val is not None else []
            })

        return results, return_debug


# =====================================================
# ---------------- MAIN RUNNER ------------------------
# =====================================================
def print_debug_report(return_debug: List[Dict[str, Any]]) -> None:
    """Print a nicely formatted debug report snippet."""
    separator = "=" * 80
    thin_sep = "-" * 80

    recon_pages = [d for d in return_debug if d.get("recon_method") != "skipped"]
    skipped_pages = [d for d in return_debug if d.get("recon_method") == "skipped"]
    anomaly_pages = [d for d in return_debug if d.get("anomaly_status") == 1]

    print(f"\n{separator}")
    print(f"  CHECK 43 — SAMPLE QUANTITY RECONCILIATION REPORT")
    print(f"{separator}")
    print(f"  Total pages processed : {len(return_debug)}")
    print(f"  Reconciliation pages  : {len(recon_pages)}")
    print(f"  Skipped (no recon)    : {len(skipped_pages)}")
    print(f"  Anomalies detected    : {len(anomaly_pages)}")
    print(f"{separator}\n")

    if not recon_pages:
        print("  No reconciliation tables detected in this BMR.\n")
        return

    for entry in recon_pages:
        page_no = entry["page_no"]
        status = "❌ ANOMALY" if entry["anomaly_status"] == 1 else "✅ PASS"
        diff = entry.get("diff", "N/A")
        recon_val = entry.get("recon_val", "N/A")
        total_calc = entry.get("total_calc_kg", "N/A")
        reason = entry.get("reason", "")
        method = entry.get("recon_method", "")
        
        print(f"  Page {str(page_no):>4s}  |  {status}  |  Method: {method}")
        print(f"           Recon Value : {recon_val} kg")
        print(f"           Calculated  : {total_calc} kg")
        print(f"           Difference  : {diff} kg")
        if reason:
            print(f"           Reason      : {reason}")
        
        details = entry.get("sampling_details", [])
        if details:
            print(f"           -> Sampling Details:")
            for det in details:
                p = det.get('page')
                m = det.get('method')
                ckg = det.get('calc_kg', det.get('status', ''))
                raw = det.get('raw', ('', '', ''))
                print(f"              Page {str(p):>3s} ({m}): {raw[0]}mL * {raw[1]}g/mL / 1000 = {ckg}kg (recorded: {raw[2]}kg)")
        
        print(f"  {thin_sep}")

    if anomaly_pages:
        print(f"\n  ANOMALY SUMMARY:")
        for entry in anomaly_pages:
            print(f"    • Page {entry['page_no']}: {entry.get('reason', 'Anomaly detected')}")
    else:
        print(f"\n  ✅ No anomalies detected across all reconciliation pages.")
    print()

if __name__ == "__main__":
    from extract_data import get_filled_master_json
    import sys
    
    if len(sys.argv) > 1:
        complete_data_path = sys.argv[1]
    else:
        complete_data_path = "/home/softsensor/Desktop/Amneal/all_result_76_20feb.json"
    
    print("Running Check 43 - Sample Quantity Verification (Optimized: Hybrid)...")
    
    try:
        filled_master_json = get_filled_master_json(complete_data_path)
        print(f"Loaded {len(filled_master_json)} pages from filled_master_json")
        
        validator = SampleQuantityVerificationValidator()
        results, return_debug = validator.run_validation(filled_master_json)
        
        # Print formatted debug report
        print_debug_report(return_debug)
        
        print("\n" + "=" * 80)
        print("  SIMPLIFIED RESULTS (JSON)")
        print("=" * 80)
        print(json.dumps(results, indent=2))

    except ImportError:
        print("Error: extract_data module not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
