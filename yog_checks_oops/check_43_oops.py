import json
import os
import re
from typing import Union, List, Dict, Any
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
GPT5_2_ENDPOINT = os.environ.get("AZURE_GPT5_ENDPOINT", "https://amneal-gpt-5.cognitiveservices.azure.com")
GPT5_2_API_KEY = os.environ.get("AZURE_GPT5_API_KEY", "")
GPT5_2_API_VERSION = os.environ.get("AZURE_GPT5_API_VERSION", "2025-04-01-preview")

# Initialize the client expected by the class
client_gpt5_1 = AzureOpenAI(
    api_key=GPT5_2_API_KEY,
    api_version=GPT5_2_API_VERSION,
    azure_endpoint=GPT5_2_ENDPOINT
)


class SampleQuantityVerificationValidator:
    """
    Check 43: AI tool should verify actual sample quantity with standard sample quantity and reconcile.
    ---
    Description:
    1. Extract sampling calculations from Manufacturing Process → Sampling sections
    2. Extract reconciliation values from Manufacturing Process → Reconciliation tables
    3. Verify that computed calc_kg matches recorded recon_kg within tolerance (±0.001 kg)
    4. Flag anomalies ONLY on reconciliation pages where mismatches are detected
    ---
    Type: LLM-based
    ---
    Attachment Required: NO
    ---
    Author: Mehul Kasliwal (30 Jan 2026)
    """

    # ===================== CONSTANTS =====================

    SECTION_NAME = "Document Compliance"
    CHECK_NAME = "verified actual sample quantity with standard sample quantity and reconcile the sample quantity"
    CHUNK_SIZE_PAGES = 4

    SYSTEM_PROMPT = """
You are a BMR cross-check assistant.

Your responsibility is to VALIDATE CONSISTENCY between:
(A) Sampling quantities CALCULATED from Manufacturing Process → Sampling sections
and
(B) Sampling quantities RECORDED in Manufacturing Process → Reconciliation under "Sample of bulk solution".

IMPORTANT SCOPE RULES:
- You MUST NOT rely on page numbers, page indices, or page references.
- You MUST link values ONLY by semantic meaning and labels.
- Sampling calculations may appear on one page, while reconciliation values may appear on a different page.
- Treat all OCR pages as a continuous document.

CRITICAL ANOMALY FLAGGING RULE:
- Flag anomaly_status = 1 ONLY on pages that contain the RECONCILIATION table with "Sample of bulk solution" entry
- DO NOT flag anomaly on pages containing only the source Sampling calculations
- Even if there's a mismatch, the anomaly should be flagged ONLY on the reconciliation page, NOT on the sampling page

TASK OBJECTIVE:
Verify that the sampling quantity computed from Manufacturing Process → Sampling
matches the corresponding "Sample of bulk solution" quantity recorded in the Reconciliation table.

────────────────────────────────────
SAMPLING CALCULATION (SOURCE A)
────────────────────────────────────
From any Manufacturing Process → Sampling section(s):

1) Identify sampling expressions using the canonical formula:
   Sample_kg = (Sample_mL × Density_g_per_mL) / 1000

2) Accept:
   - × / x / X
   - /1000 or ÷1000
   - Units: mL/ml/ML, g/mL, g per mL
   - OCR noise, spacing errors, line breaks

3) Compute calc_kg for each valid sampling line.
   - Round to 3 decimals.
   - If multiple sampling lines belong to the same phase (bulk solution), SUM them.

────────────────────────────────────
RECONCILIATION VALUE (SOURCE B)
────────────────────────────────────
From Manufacturing Process → Reconciliation table(s):

1) Locate rows semantically referring to "Sample of bulk solution".
   Accept variations such as:
   - "QC Samples of bulk solution after volume makeup (from manufacturing vessel)"
   - "Total sample of bulk solution after recirculation"
   - Any equivalent wording clearly referring to bulk solution sampling

2) Extract the Qty. (kg) value as recon_kg.

────────────────────────────────────
CROSS-CHECK LOGIC
────────────────────────────────────
Compare:
- Total computed calc_kg (from Sampling)
WITH
- Recorded recon_kg (from Reconciliation)

Tolerance:
|recon_kg − calc_kg| ≤ 0.001 → MATCH

────────────────────────────────────
ANOMALY CONDITIONS
────────────────────────────────────
Set anomaly_status = 1 ONLY on pages containing Reconciliation table if ANY of the following occur:

- Absolute difference between calc_kg and recon_kg exceeds ±0.001 kg
- Sampling mL or density is missing or invalid (cannot compute) AND this page has reconciliation table
- Reconciliation "Sample of bulk solution" kg exists but NO corresponding computed sampling kg is found

IMPORTANT: Pages containing ONLY sampling calculations (without reconciliation table) should ALWAYS have anomaly_status = 0.

Otherwise:
anomaly_status = 0

────────────────────────────────────
OUTPUT CONTRACT (STRICT)
────────────────────────────────────
Return ONE JSON array with ONE object PER OCR PAGE, preserving input order:

{
  "page_no": <integer>,
  "section_name": "Document Compliance",
  "check_name": "verified actual sample quantity with standard sample quantity and reconcile the sample quantity",
  "anomaly_status": 0 or 1
}

Rules:
- anomaly_status = 1 ONLY for pages that CONTAIN the Reconciliation table with "Sample of bulk solution" AND have a detected anomaly
- Pages with ONLY sampling calculations MUST have anomaly_status = 0
- All other pages MUST have anomaly_status = 0
- No prose, no explanations, no extra keys
"""

    TASK_DESC = """
From the OCR markdown, perform cross-document verification:

STEP 1: Scan ALL pages to locate Manufacturing Process → Sampling sections
- Extract sampling calculations: <mL> × <density> g/mL / 1000 = <written_kg> Kg
- Compute calc_kg = (mL × density) / 1000, round to 3 decimals
- If multiple sampling entries exist, sum them for total calc_kg

STEP 2: Scan ALL pages to locate Manufacturing Process → Reconciliation table
- Find "Sample of bulk solution" row (or equivalent wording)
- Extract recon_kg value

STEP 3: Compare calc_kg (from any sampling page) with recon_kg (from reconciliation page)
- Check if |recon_kg − calc_kg| ≤ 0.001

STEP 4: Generate output array with ONE element per page:
- For pages containing ONLY sampling calculations: anomaly_status = 0 (always)
- For pages containing Reconciliation table with "Sample of bulk solution":
  * anomaly_status = 1 if mismatch detected (|recon_kg − calc_kg| > 0.001)
  * anomaly_status = 1 if sampling data is missing/invalid
  * anomaly_status = 0 if values match within tolerance
- For all other pages: anomaly_status = 0

CRITICAL: Only flag anomalies on the reconciliation page where verification happens, NOT on the source sampling pages.

Return the JSON array as described in the system prompt.
"""

    # Regex for stripping code fences
    _CODE_FENCE_RE = re.compile(r"^\\s*```(?:json)?\\s*(.*?)\\s*```$", re.DOTALL)

    def __init__(self):
        """
        Initialize the validator.
        Now uses global client_gpt5_1 initialized at module level.
        """
        pass

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

    def _default_rows_for_slice(
        self,
        start_idx: int,
        count: int,
        anomaly: int = 0,
    ) -> List[Dict[str, Any]]:
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

    def _strip_code_fences(self, txt: str) -> str:
        """Strip markdown code fences from text."""
        m = self._CODE_FENCE_RE.match(txt.strip())
        return m.group(1) if m else txt

    def _try_parse_json_anyshape(self, txt: str) -> Union[List[Dict[str, Any]], None]:
        """Try to parse JSON from various formats."""
        s = self._strip_code_fences(txt).strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for k in ("data", "result", "output", "results"):
                    if isinstance(obj.get(k), list):
                        return obj[k]
        except Exception:
            pass

        start = s.find("[")
        while start != -1:
            depth = 0
            for i in range(start, len(s)):
                if s[i] == "[":
                    depth += 1
                elif s[i] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i+1]
                        try:
                            arr = json.loads(candidate)
                            if isinstance(arr, list):
                                return arr
                        except Exception:
                            break
            start = s.find("[", start + 1)
        return None

    def _normalize_batch_array(
        self,
        arr: List[Dict[str, Any]],
        global_start_index: int,
        expected_len: int,
    ) -> List[Dict[str, Any]]:
        """Normalize the batch array to expected format and length."""
        arr = list(arr) if isinstance(arr, list) else []

        if len(arr) > expected_len:
            arr = arr[:expected_len]
        if len(arr) < expected_len:
            arr += [{"anomaly_status": 0}] * (expected_len - len(arr))

        normalized = []
        for idx in range(expected_len):
            item = arr[idx] if isinstance(arr[idx], dict) else {}
            try:
                anomaly_status = int(item.get("anomaly_status", 0))
            except Exception:
                anomaly_status = 0
            normalized.append({
                "page_no": global_start_index + idx + 1,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": 1 if anomaly_status else 0,
            })
        return normalized

    def _call_model_once(self, system_prompt: str, user_prompt: str, use_json_format: bool) -> Union[List[Dict[str, Any]], None]:
        """Make a single model call."""
        kwargs = dict(
            model="gpt-5-mini",  # Changed to match check_38.py
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if use_json_format:
            kwargs["response_format"] = {"type": "json_object"}

        # Use global client_gpt5_1
        resp = client_gpt5_1.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        return self._try_parse_json_anyshape(content)

    def _run_batch_through_model(
        self,
        pages_slice: List[Dict[str, Any]],
        global_start_index: int,
    ) -> List[Dict[str, Any]]:
        """Process a batch of pages through the model."""
        blocks = []
        for offset, p in enumerate(pages_slice, start=0):
            absolute_page_no = global_start_index + offset + 1
            blocks.append(self._page_block(absolute_page_no, p))
        markdown_content = "\n\n".join(blocks)

        base_user_prompt = (
            "OCR MARKDOWN (preserve page boundaries below)\n"
            f"{markdown_content}\n\n"
            "Your Task:\n"
            f"{self.TASK_DESC}"
            "\n\nSTRICT OUTPUT: Return ONLY a JSON array (or an object with a single array field like 'data'). "
            "No prose, no extra keys beyond the required ones."
        )

        arr = self._call_model_once(self.SYSTEM_PROMPT, base_user_prompt, use_json_format=True)

        if arr is None:
            nudged_prompt = base_user_prompt + "\nReturn ONLY a JSON array of objects."
            arr = self._call_model_once(self.SYSTEM_PROMPT, nudged_prompt, use_json_format=False)

        if arr is None:
            return self._default_rows_for_slice(global_start_index, len(pages_slice), anomaly=0)

        return self._normalize_batch_array(arr, global_start_index, expected_len=len(pages_slice))

    # ===================== VALIDATION =====================

    def run_validation(self, json_path_or_obj: Union[str, list, dict]) -> List[Dict[str, Any]]:
        """
        Main entry point for sample quantity verification/reconciliation validation.
        
        Emits one row per page with keys: page_no, section_name, check_name, anomaly_status (0/1)
        
        IMPORTANT: Anomalies are flagged ONLY on reconciliation pages, NOT on sampling calculation pages.

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

            batch_out = self._run_batch_through_model(
                pages_slice=slice_pages,
                global_start_index=start,
            )
            results.extend(batch_out)

        # Final safety
        if len(results) != total_pages:
            if len(results) > total_pages:
                results = results[:total_pages]
            else:
                results += self._default_rows_for_slice(len(results), total_pages - len(results), anomaly=0)

        for i in range(total_pages):
            results[i]["page_no"] = i + 1
            results[i]["section_name"] = self.SECTION_NAME
            results[i]["check_name"] = self.CHECK_NAME
            results[i]["anomaly_status"] = 1 if int(results[i].get("anomaly_status", 0)) else 0

        return results


# =====================================================
# ---------------- MAIN RUNNER ------------------------
# =====================================================
if __name__ == "__main__":
    from extract_data import get_filled_master_json
    
    # Load OCR data from the complete_data JSON file
    complete_data_path = "/home/softsensor/Desktop/Amneal/all_result_76_20feb.json"
    
    print("Running Check 43 - Sample Quantity Verification...")
    
    # Extract the filled_master_json from the complete data
    # Note: Ensure extract_data.py is in the python path or same directory
    try:
        filled_master_json = get_filled_master_json(complete_data_path)
        print(f"Loaded {len(filled_master_json)} pages from filled_master_json")
        
        # Run the sample quantity verification check using the extracted page data
        validator = SampleQuantityVerificationValidator()
        results = validator.run_validation(filled_master_json)
        print(json.dumps(results, indent=2))
    except ImportError:
        print("Error: extract_data module not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
