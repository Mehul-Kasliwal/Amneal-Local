# NOTE: This function is commented out as it's no longer required.
# The combine_selected_pages_26 function now directly uses markdown text.
# 
# def md_to_clean_text_26(md_string: str) -> str:
#     """
#     Convert markdown → cleaned plain text (in memory).
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


def combine_selected_pages_26(selected_pages: List[Dict]) -> str:
    """
    Concatenate cleaned markdown from selected pages into a single
    LLM-friendly string.
    """
    parts = []
    for obj in selected_pages:
        page_no = obj.get("page_no", "")
        markdown_text = obj.get("markdown", "")

        # NOTE: md_to_clean_text_26 call removed as it's no longer required.
        # Directly using markdown_text instead.

        block = (
            "-------------------------\n"
            f"PAGE NUMBER: {page_no}\n"
            "-------------------------\n\n"
            f"{markdown_text}\n"
        )
        parts.append(block)

    return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Page Selection Logic
# ----------------------------------------------------------------------

def find_keyphrase_in_rules_26(obj: Any, KEY_PHRASE: str ) -> bool:
    """
    Recursively search for KEY_PHRASE in rules_or_instructions fields.
    """
    if isinstance(obj, dict):
        if "rules_or_instructions" in obj:
            rules = obj["rules_or_instructions"]
            if isinstance(rules, list):
                for rule in rules:
                    if isinstance(rule, str) and KEY_PHRASE.lower() in rule.lower():
                        return True

        # Search nested fields
        return any(
            find_keyphrase_in_rules_26(val, KEY_PHRASE) for val in obj.values()
        )

    elif isinstance(obj, list):
        return any(
            find_keyphrase_in_rules_26(item, KEY_PHRASE) for item in obj
        )

    return False


def select_26(document_pages: List[Dict]) -> List[Dict]:
    """
    Select only pages that contain potency calculation rules.
    """
    selected = []

    for page_obj in document_pages:
        page_no = page_obj.get("page", "")
        content = page_obj.get("page_content", [])
        markdown_page = page_obj.get("markdown_page", "")

        if find_keyphrase_in_rules_26(content, KEY_PHRASE="POTENCY CALCULATION FOR API"):
            selected.append({
                "page_no": page_no,
                "markdown": markdown_page
            })

    return selected


# ----------------------------------------------------------------------
# LLM Analysis
# ----------------------------------------------------------------------

def analyze_bmr_with_llm_26(combined_text: str) -> str:
    """
    Send merged potency text to Azure OpenAI for analysis.
    Returns JSON string.
    """

    # analysis_prompt = """
    # You are analyzing OCR-extracted text of a Batch Manufacturing Record (BMR) 
    # corresponding to Material Dispensing Section.
    #
    # Your objective is to:
    # - Identify which ingredient or raw material the calculation belongs to.
    # - Determine whether the potency formula is based on Option 1 (Single Lot) or Option 2 (Double Lot / Two Different Lots).
    # - Perform all necessary calculations and compare results with the values written on the scanned document.
    #
    # ------------------------------------------------------------
    # CALCULATION LOGIC AND RULES
    # ------------------------------------------------------------
    #
    # OPTION 1 — SINGLE LOT FORMULA
    # Use this when only one lot is used (the other lot will be marked as “NA”). 
    # Formula depends on whether LOD (Loss on Drying) is considered.
    #
    # C ia a Constant = 5.26 g for most raw materials ( but for Potassiam Chloride IP;  C = 0.37 g)
    #
    # Without LOD:
    # A (Actual Quantity) (g) = (C g / 1.0 L) × (Batch Size in L) × (100 / % Assay)
    #
    # With LOD:
    # A (Actual Quantity) (g) = (C g / 1.0 L) × (Batch Size in L) × (100 / % Assay) × (100 / (100 - LOD))
    #
    # Total Quantity A(g) or A/1000 (in Kg)
    #
    #
    # ------------------------------------------------------------
    #
    # OPTION 2 — DOUBLE LOT FORMULA
    # Used when two different lots are combined (Lot 1 and Lot 2). 
    # For each raw material, one of these lots may still be “NA” depending on usage.
    #
    # Lot 1 Calculation:
    #
    # Without LOD:
    # L1 = A1 (Actual Quantity of Lot 1) (g) × (1.0 L / C g) × (% Assay of Lot 1 / 100)
    #
    # With LOD:
    # L1 = A1 (Actual Quantity of Lot 1) (g) × (1.0 L / C g) × (% Assay of Lot 1 / 100) × ((100 - LOD of Lot 1) / 100)
    #
    # Lot 2 Calculation:
    # L2 = Theoretical Batch Size (L) - L1
    #
    # Then calculate “A2” (for Lot 2):
    #
    # Without LOD:
    # A2 (estimated quantity of lot 2) (g) = (C g / 1.0 L) × (L2) × (100 / % Assay of Lot 2)
    #
    # With LOD:
    # A2 (estimated quantity of lot 2) (g) = (C g / 1.0 L) × (L2) × (100 / % Assay of Lot 2) × (100 / (100 - LOD of Lot 2))
    #
    # Total Quantity Combined A(g) = A1(g) + A2(g)    
    # Total Quantity Combined A(kg) = A(g) / 1000
    #
    #
    # ------------------------------------------------------------
    # IMPORTANT NOTES
    # ------------------------------------------------------------
    # - Each raw material will use only one option (Single Lot or Double Lot). 
    # - If OPTION 2: DOUBLE Lot FORMULA is filled then Lot1, and Lot2 calculation under OPTION 2 should be strictly used in all calculations. 
    # - If a lot is not used, its field will contain “NA”.
    # - The scanned copy may represent numerator and denominator with horizontal lines, so interpret fractions carefully.
    # - Perform calculations accurately according to the formulas above.
    # - Compare your calculated output with the printed/handwritten result visible in the scanned document.
    #
    # ------------------------------------------------------------
    # REQUIRED OUTPUT FORMAT
    # ------------------------------------------------------------
    # For each raw material found in the document, return the following structured information:
    #
    # 1. Name of the Raw Material Used
    # 2. A.R. No:
    #    - A1 (for Lot 1)
    #    - A2 (for Lot 2, or NA if not used)
    # 3. % Assay:
    #    - A1 value and A2 value (A2 = NA if not used)
    # 4. LOD (if available):
    #    - A1 value and A2 value (A2 = NA if not used)
    # 5. Lot Selected for Calculation:
    #    - Option 1 (Single Lot) or Option 2 (Double Lot)
    # 6. Actual Formula Used:
    #    - Show the full mathematical expression with substituted values and the calculated result.
    # 7. Match Verification:
    #    - Compare your calculated “A” value with the printed/handwritten value on the scanned page.
    # 8. Anomaly Flag : 
    #    - 1 if there is a mismatch between calculated and printed value on a page.
    #    - 0 if the values match.
    # 9. Page Number :
    #     - Main page number analysed for the verification of this section mentioned above markdown 
    #
    # ------------------------------------------------------------
    # EXAMPLE RESPONSE
    # ------------------------------------------------------------
    # Raw Material: Sodium Chloride IP
    # A.R. No: A1 = 23007409, A2 = NA
    # % Assay: A1 = 99.7, A2 = NA
    # LOD: A1 = 0.8, A2 = NA
    # Calculation Option: Option 1 (Single Lot)
    # Formula Used: (5.26 / 1.0) × 4000 × (100 / 99.7) × (100 / (100 - 0.8))
    # Calculated Value: 21.273 kg
    # Printed Value: 21.273 kg
    # Anomaly: 0
    # Pages : 9
    #
    #
    # Return ONLY valid JSON with this schema:
    #
    # {
    #   "materials": [
    #     {
    #       "raw_material_name": "string",
    #       "ar_no": {"A1": "string", "A2": "string or NA"},
    #       "assay_percent": {"A1": "float", "A2": "float or NA"},
    #       "lod_percent": {"A1": "float", "A2": "float or NA"},
    #       "calculation_option": 1,
    #       "formula_used": "string",
    #       "calculated_value": "float",
    #       "printed_value": "float",
    #       "anomaly_flag": 0,
    #       "page_no": int
    #     }
    #   ]
    # }
    #
    # Repeat for all raw materials detected.
    # """
    analysis_prompt = get_check_prompt("check_26")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a QA documentation expert specializing in pharmaceutical BMR interpretation, "
                "potency calculations, batch record verification, ALCOA+ principles, and GMP compliance."
            )
        },
        {
            "role": "user",
            "content": f"""
    ### TASK INSTRUCTION
    {analysis_prompt}
    
    ------------------------------
    ### OCR EXTRACTED BMR TEXT
    {combined_text}
    ------------------------------
    """
            }
        ]

    response = azure_openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        # max_tokens=8000,
        # temperature=0.1,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content.strip()


# ----------------------------------------------------------------------
# Result Merging
# ----------------------------------------------------------------------

def final_rm_results_combine_26(base_results: List[Dict], llm_json_str: str) -> List[Dict]:
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


# ----------------------------------------------------------------------
# Entry Point for RM Checking
# ----------------------------------------------------------------------


def rm_formula_checking_and_estimation_26(filtered_json: List[Dict]) -> List[Dict]:
    """
    Main execution pipeline:
    - build base structure
    - select potency pages
    - combine text
    - LLM analysis
    - merge anomaly flags
    """

    base_results = [
        {
            "page_no": idx,
            "section_name": get_check_process("check_26"),
            "check_name": get_check_name("check_26"),
            "anomaly_status": 0
        }
        for idx in range(1, len(filtered_json) + 1)
    ]

    selected_pages = select_26(filtered_json)
    combined_text = combine_selected_pages_26(selected_pages)

    llm_json_str = analyze_bmr_with_llm_26(combined_text)

    final_results = final_rm_results_combine_26(base_results, llm_json_str)
    return final_results
