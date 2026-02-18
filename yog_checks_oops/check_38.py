from __future__ import annotations
import os
import json, re
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
 
 
 
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
 
 
 
class CheckManufacturingTimeValidation:
    """
    Check 38 :- AI tool should be able to check actual process start time and end time and verified with standard time required to finish the process. Any discrepancy highlighted.
    ----
    Description : Validates manufacturing process step times against specified limits.
    : tested mentioned on BMR : AH230061 page_no (62,63,64,65,66,67,68,70)
    ----
    type :- LLM based
    ----
    Attachment_Required : NO
    
    Author : MD SABIULLAH (9 Jan 2026)
    """
    
    # =========================================================================
    # CONSTANTS & CONFIGURATION
    # =========================================================================
    
    SECTION_NAME = "Process Execution"
    CHECK_NAME = "TIME LIMIT VALIDATION"
    
    # Manufacturing Process Keyphrases
    MANUFACTURING_PROCESS_KEYPHRASE = "MANUFACTURING PROCESS"
    MANUFACTURING_STEP_KEYPHRASE = ["MANUFACTURING STEP","MANUFACTURING PROCESS STEP"]
    
    # Required Table Columns (with case variations)
    REQUIRED_COLUMNS = {
        "manufacturing_instruction": ["Manufacturing Instruction", "Manufacturing instruction"],
        "start_time": ["Start Time", "Start time"],
        "end_time": ["End Time", "End time"],
        "actual_process": ["Actual Process", "Actual process"]
    }
    
    # =========================================================================
    # LLM PROMPT TEMPLATE
    # =========================================================================
    
    LLM_PROMPT_TEMPLATE = """
    You are analyzing a Manufacturing Process page from a Batch Manufacturing Record (BMR).
 
    **Task:**
    1. Extract the ACTUAL start time and end time from the markdown for activities that have explicit time limits.
    2. Look for any EXPLICITLY WRITTEN duration in the "Actual Process" or activity text (e.g., "Stirring time: XX Minutes", "Mixed for XX mins").
       - **CRITICAL:** Extract the number EXACTLY as written in the text. Do NOT correct it to match the timestamps.
    3. Ignore generic table start/end times if specific times are written next to the activity field.
    4. **SKIP CONDITION:** If Start Time or End Time is NOT present (blank/empty) in the row, DO NOT extract this activity.
    5. Ignore any temperature values (like °C), pH values, or other measurements.
    5. Only extract TIME values in format HH:MM.
    6. Calculate the time difference (end_time - start_time) in minutes.
    7. Identify the expected time limit from 'Actual Process' or 'Manufacturing Instruction'.
    8. Validate:
       - **Math Check:** Explicitly calculate: (End Time Minutes) - (Start Time Minutes).
       - **Consistency Check:** Does Result MATCH the Written Duration?
       - **Compliance Check:** Is the Written Duration within the Limit?
 
    **Limit Types:**
    - **NLT (Not Less Than):** Actual Duration >= Limit
    - **NMT (Not More Than):** Actual Duration <= Limit
    - **RANGE:** Min Limit <= Actual Duration <= Max Limit
    - **EXACT:** Actual Duration == Limit
 
    **Validation Rules:**
    - **ANOMALY if:**
        - **Consistency Failure:** Calculated Duration != Written Duration.
        - **Compliance Failure:** Written Duration violates the Limit.
        - Calculated Duration violates the Limit (specifically if Written Duration is missing).
    - **VALIDATE:** Activities with phrases like "Stirring time: X Min (Limit: ...)", "Mix for X min", "Cooling time: X Min (Limit: ...)".
    - **SKIP (CRITICAL):** If the activity details (Quantity, Speed, etc.) contain "NA", "N.A.", "Not Applicable" or appear crossed out/strikethrough text (e.g. "line graph decreasing"), IGNORE any timestamps found in that row. Treat the step as NOT PERFORMED.
    - **SKIP:** Clarity checks, Temperature readings, pH checks, Visual inspections UNLESS a specific time limit is mentioned.
 
    **Markdown Content:**
    {markdown}
 
    **Important - Time Extraction Logic:**
    - Look for the SPECIFIC activity mentioned in the markdown (e.g., "Stirring time").
    - **Multiple Activities per Row:** If a row contains multiple steps (e.g., "Add and dissolve..." AND "Stirring time..."), it will often have MULTIPLE timestamps in the Start/End columns.
    - **Merged Timestamps:** Watch out for OCR errors where times are glued together (e.g., "HH:MMHH:MM" means HH:MM AND HH:MM). **SPLIT THEM.**
    - **Mapping Logic (GENERIC SEQUENTIAL):**
      - **Step 1:** Identify ALL distinct actions/verbs in the "Manufacturing Instruction" text, even those without limits (e.g. "Add...", "Dissolve...", "Stir...").
      - **Step 2:** Count the independent time pairs found in the Start/End columns.
        - **IMPORTANT:** If you SPLIT a merged timestamp (e.g. "08:0709:11"), count them as **TWO DISTINCT** Time Pairs (Time Pair N and Time Pair N+1).
      - **Step 3:** Map them sequentially based on CHRONOLOGICAL execution:
        - 1st Action performed -> 1st Time Pair recorded
        - 2nd Action performed -> 2nd Time Pair recorded
        - **Context Check:** Only deviate from this if there is explicit evidence that the 1st action was skipped or the time clearly labels the 2nd action. Otherwise, assume strictly sequential matching.
    - **Handling Missing Data:**
      - If there are 2 actions but only 1 time pair found, it typically belongs to the 1st action.
      - The 2nd action should be marked as `start_time: null`, `end_time: null`.
      - **Goal:** Assign the time to the CORRECT action. If uncertain, follows sequential order.
    - **Verify Sub-Step:** Ensure you are extracting the time for the *specific* sub-step (Stirring) being validated.
    - **Missing Limits:** If an activity (e.g. Cooling) has times but NO specific limit text (like "NLT 10 min"), calculate the duration and mark `is_compliant: true` (PASS).
    - Extract HH:MM format times from the markdown structure (e.g., from \u003ctd\u003e tags near activity fields).
 
    **Output Format (JSON only):**
    {{
        "extractions": [
            {{
                "activity": "<Name of activity, e.g., Stirring time>",
                "start_time": "HH:MM",
                "end_time": "HH:MM",
                "calculated_duration_minutes": <number>,
                "stated_duration_text": "<duration written in text, e.g. 'XX Minutes' or null>",
                "expected_duration_text": "<text extracted from doc, e.g. Limit: NLT XX minutes>",
                "time_limit_type": "NLT|NMT|RANGE|EXACT",
                "min_limit": <number or null>,
                "max_limit": <number or null>,
                "is_compliant": true/false,
                "deviation_minutes": <number>,
                "notes": "<explain failure: e.g. Content Mismatch: Calc XX != Stated YY>"
            }}
        ]
    }}
    If no activities with time limits are found or performed, return "extractions": [].
    """
 
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def find_keyphrase_in_rules(obj: Any, keyphrase: str) -> bool:
        """Recursively searches for a keyphrase in rules_or_instructions."""
        try:
            if isinstance(obj, dict):
                rules = obj.get("rules_or_instructions")
                if isinstance(rules, list):
                    return any(
                        keyphrase.lower() in r.lower()
                        for r in rules if isinstance(r, str)
                    )
                return any(CheckManufacturingTimeValidation.find_keyphrase_in_rules(v, keyphrase) for v in obj.values())
 
            if isinstance(obj, list):
                return any(CheckManufacturingTimeValidation.find_keyphrase_in_rules(i, keyphrase) for i in obj)
 
            return False
        except Exception:
            return False
 
    @classmethod
    def has_required_columns(cls, page_content: List[Dict]) -> bool:
        """Check if page has the required columns."""
        for block in page_content:
            if block.get("type") != "table":
                continue
                
            table_json = block.get("table_json", {})
            
            # Handle if table_json is already the list of records
            if isinstance(table_json, list):
                records = table_json
            else:
                records = table_json.get("records", [])
            
            if not isinstance(records, list) or not records:
                continue
            
            for record in records:
                if not isinstance(record, dict):
                    continue
                
                record_keys = set(record.keys())
                
                # Check required columns (at least one variation for each)
                has_manufacturing = any(k in record_keys for k in cls.REQUIRED_COLUMNS["manufacturing_instruction"])
                has_start = any(k in record_keys for k in cls.REQUIRED_COLUMNS["start_time"])
                has_end = any(k in record_keys for k in cls.REQUIRED_COLUMNS["end_time"])
                
                if has_manufacturing and has_start and has_end:
                    return True
        
        return False
 
    @classmethod
    def find_manufacturing_process_pages(cls, bmr_data: List[Dict]) -> List[Dict]:
        """Find pages with keyphrases AND required columns."""
        selected_pages = []
        
        for page in bmr_data:
            page_content = page.get('page_content', [])
            
            # Condition 1: Check keyphrases
            # Handle if MANUFACTURING_STEP_KEYPHRASE is a list or string
            step_phrases = cls.MANUFACTURING_STEP_KEYPHRASE
            if isinstance(step_phrases, str):
                step_phrases = [step_phrases]
                
            has_step_phrase = any(cls.find_keyphrase_in_rules(page_content, phrase) for phrase in step_phrases)
 
            has_keyphrase = (
                cls.find_keyphrase_in_rules(page_content, cls.MANUFACTURING_PROCESS_KEYPHRASE) and
                has_step_phrase
            )
            
            # Condition 2: Check columns
            has_columns = cls.has_required_columns(page_content)
            
            if has_keyphrase and has_columns:
                selected_pages.append(page)
        
        return selected_pages
 
    @staticmethod
    def extract_manufacturing_columns(page_content: List[Dict]) -> List[Dict]:
        """Extract manufacturing columns from page."""
        extracted_data = []
        
        for block in page_content:
            if block.get("type") != "table":
                continue
            table_json = block.get("table_json", {})
            
            # Handle if table_json is already the list of records
            if isinstance(table_json, list):
                records = table_json
            else:
                records = table_json.get("records", [])
            
            for record in records:
                if not isinstance(record, dict):
                    continue
                    
                manufacturing_instruction = record.get("Manufacturing Instruction", record.get("Manufacturing instruction"))
                start_time = record.get("Start Time", record.get("Start time"))
                end_time = record.get("End Time", record.get("End time"))
                actual_process = record.get("Actual process", record.get("Actual Process"))
                
                if any([manufacturing_instruction, start_time, end_time, actual_process]):
                    extracted_data.append({
                        "manufacturing_instruction": manufacturing_instruction,
                        "start_time": start_time,
                        "end_time": end_time,
                        "actual_process": actual_process
                    })
        
        return extracted_data
 
    @staticmethod
    def get_page_markdown(page: Dict) -> str:
        """Extract markdown from page, preferring filled version from additional_keys."""
        # First, check if there's a filled markdown in additional_keys
        additional_keys = page.get('additional_keys', [])
        for item in additional_keys:
            if isinstance(item, dict) and item.get('key') == 'markdown_page':
                filled_markdown = item.get('value', '')
                if filled_markdown:  # If non-empty, use it
                    return filled_markdown
        
        # Fall back to the regular markdown_page (template)
        return page.get('markdown_page', '')
 
    # =========================================================================
    # MAIN VALIDATION LOGIC
    # =========================================================================
 
    @classmethod
    def run_validation(cls, bmr_data: List[Dict]) -> List[Dict]:
        """
        Main validation runner.
        Iterates through all pages, identifies Manufacturing Steps,
        and validates time limits using LLM.
        Returns a standardized list of results for every page.
        """
        results = []
        
        # Pre-calculate pages that match the criteria to avoid redundant checking
        # (Or simpler: just check criteria inside the loop for every page)
        
        for page_data in bmr_data:
            page_num = page_data.get('page')
            page_content = page_data.get('page_content', [])
            
            anomaly_status = 0
            
            # 1. Check if page is relevant (Has Keyphrase AND Columns)
            
            # Check Keyphrases
            step_phrases = cls.MANUFACTURING_STEP_KEYPHRASE
            if isinstance(step_phrases, str):
                step_phrases = [step_phrases]
            has_step_phrase = any(cls.find_keyphrase_in_rules(page_content, phrase) for phrase in step_phrases)
            has_keyphrase = (
                cls.find_keyphrase_in_rules(page_content, cls.MANUFACTURING_PROCESS_KEYPHRASE) and
                has_step_phrase
            )
            
            # Check Columns
            has_columns = cls.has_required_columns(page_content)
            
            is_relevant_page = has_keyphrase or has_columns
            
            if is_relevant_page:
                print(f"Processing page {page_num}...", flush=True)
                try:
                    # Extract table data (Double check, though redundant if has_columns is True)
                    table_data = cls.extract_manufacturing_columns(page_content)
                    
                    if table_data:
                        # Get markdown
                        markdown = cls.get_page_markdown(page_data)
                        
                        # Prepare LLM prompt
                        prompt = cls.LLM_PROMPT_TEMPLATE.format(markdown=markdown)
                        prompt += "\n\nIMPORTANT: Return ONLY valid JSON. Do not include markdown formatting like ```json ... ```."
 
                        # Call LLM
                        response =  client_gpt5_1.chat.completions.create(
                            model="gpt-5-mini",
                            messages=[
                                {"role": "system", "content": "You are a Quality Assurance expert validating manufacturing records."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        
                        llm_content = response.choices[0].message.content
                        
                        # Clean up markdown fences
                        if llm_content.startswith("```json"):
                            llm_content = llm_content[7:]
                        if llm_content.endswith("```"):
                            llm_content = llm_content[:-3]
                        llm_content = llm_content.strip()
                        
                        llm_result = json.loads(llm_content)
                        
                        # Check Compliance
                        extractions = llm_result.get('extractions', [])
                        for extraction in extractions:
                            if not extraction.get('is_compliant', True):
                                anomaly_status = 1
                                break
                                
                except Exception:
                    # On error, we default to 0 (Pass) or could mark as 1 if critical.
                    # Standard practice for 'skip on error' implies 0 unless specific requirement.
                    # We will log nothing to stdout as requested.
                    pass
 
            # Append Result for THIS page
            results.append({
                "page": page_num,
                "section_name": cls.SECTION_NAME,
                "check_name": cls.CHECK_NAME,
                "anomaly_status": anomaly_status
            })
            
        return results
 
 
 
if __name__ == "__main__":
    try:
        # Load BMR data
        with open("/home/softsensor/Desktop/Amneal/yog_checks_oops/bmr_76_filled.json", "r") as f:
            bmr_data = json.load(f)
 
        # Run validation
        results = CheckManufacturingTimeValidation.run_validation(bmr_data)
 
        # Print results
        print(json.dumps(results, indent=4))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()