import json
import os
import re
from typing import Union, List, Dict, Any
from openai import AzureOpenAI  # or OpenAI

# Setup your Azure OpenAI client
client = AzureOpenAI(
    api_key=os.environ.get("AZURE_GPT5_MINI_API_KEY", ""),
    api_version=os.environ.get("AZURE_GPT5_MINI_API_VERSION", "2025-04-01-preview"),
    azure_endpoint=os.environ.get("AZURE_GPT5_MINI_ENDPOINT", "https://amneal-gpt-5-mini.cognitiveservices.azure.com")
)

CHUNK_SIZE_PAGES = 4

SYSTEM_PROMPT_CALC = """
You are a document QA assistant for Batch Manufacturing Records (BMRs).
Your job: from OCR'd pages, locate entries under Manufacturing Process → Sampling (and any equivalent phrasing) and verify sampling quantities and conversions to kilograms.

What to extract and validate:

Total sample (In Kg) style rows, e.g.
"Total sample (In Kg): 300 mL × (Bulk density 0.99 g/mL) / 1000 = 0.297 Kg"

Canonical formula (must use):
Sample_kg = (Sample_mL × Density_g_per_mL) / 1000

Round calc_kg to 3 decimals for comparison.
Tolerance: ±0.001 kg (|written_kg − calc_kg| ≤ 0.001 → OK).

Output contract (STRICT):
Return a single JSON array; each element is:
{
  "page_no": <integer>,
  "section_name": "Document Compliance",
  "check_name": "check sample quantity and calculate the total sample quantity",
  "anomaly_status": <0 or 1>
}

One element per page in the OCR input (in order).
anomaly_status = 1 if any sampling-related anomaly exists on that page; otherwise 0.
"""

TASK_DESC_CALC = """
From the OCR markdown, scan every page for Manufacturing Process → Sampling content.
For each page with "Total sample (In Kg)" style row:
- Extract numbers, compute calc_kg = (mL × density) / 1000
- Round to 3 decimals
- Flag anomaly if |written_kg − calc_kg| > 0.001 or required numbers are missing/invalid
Return the JSON array—one element per page—with anomaly_status set accordingly.
"""

def _load_pages(json_like: Union[str, list, dict]) -> List[Dict[str, Any]]:
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

def _page_block(i: int, p: Dict[str, Any]) -> str:
    md = p.get("markdown_page", "") or ""
    return f"--- PAGE {i} START ---\n{md}\n--- PAGE {i} END ---"

def _default_rows_for_slice(start_idx: int, count: int, anomaly: int = 0) -> List[Dict[str, Any]]:
    return [
        {
            "page_no": start_idx + i + 1,
            "section_name": "Document Compliance",
            "check_name": "check sample quantity and calculate the total sample quantity",
            "anomaly_status": 1 if anomaly else 0,
        }
        for i in range(count)
    ]

def calculate_sample_quantity_chunked(json_path_or_obj: Union[str, list, dict]) -> List[Dict[str, Any]]:
    CHECK_NAME = "check sample quantity and calculate the total sample quantity"
    all_pages = _load_pages(json_path_or_obj)
    if not all_pages:
        return []

    results: List[Dict[str, Any]] = []
    total_pages = len(all_pages)
    chunk = max(1, int(CHUNK_SIZE_PAGES))

    for start in range(0, total_pages, chunk):
        end = min(start + chunk, total_pages)
        slice_pages = all_pages[start:end]
        
        # Build markdown content
        blocks = []
        for offset, p in enumerate(slice_pages):
            absolute_page_no = start + offset + 1
            blocks.append(_page_block(absolute_page_no, p))
        markdown_content = "\n\n".join(blocks)

        # Call model
        response = client.chat.completions.create(
            model="gpt-5",  # or your deployment name
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CALC},
                {"role": "user", "content": f"OCR MARKDOWN:\n{markdown_content}\n\nTask:\n{TASK_DESC_CALC}"},
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
                "section_name": "Document Compliance",
                "check_name": CHECK_NAME,
                "anomaly_status": 1 if item.get("anomaly_status", 0) else 0,
            })

    return results

if __name__ == "__main__":
    from extract_data import get_filled_master_json
    
    # Load OCR data from the complete_data JSON file
    complete_data_path = "/home/softsensor/Desktop/Amneal/yog_checks/complete_data_61 1.json"
    
    print("Running Check 42 - Sample Quantity Calculation...")
    
    # Extract the filled_master_json from the complete data
    filled_master_json = get_filled_master_json(complete_data_path)
    print(f"Loaded {len(filled_master_json)} pages from filled_master_json")
    
    # Run the sample quantity check using the extracted page data
    results = calculate_sample_quantity_chunked(filled_master_json)
    print(json.dumps(results, indent=2))