import json
import re
import ast
import os
import operator
from typing import List, Dict, Any, Optional, Tuple
from openai import AzureOpenAI

# Configuration
GPT5_2_ENDPOINT = os.environ.get("AZURE_GPT5_ENDPOINT", "https://amneal-gpt-5.cognitiveservices.azure.com")
GPT5_2_API_KEY = os.environ.get("AZURE_GPT5_API_KEY", "")
GPT5_2_API_VERSION = os.environ.get("AZURE_GPT5_API_VERSION", "2025-04-01-preview")

# Initialize the client expected by the class
client_gpt5_1 = AzureOpenAI(
    api_key=GPT5_2_API_KEY,
    api_version=GPT5_2_API_VERSION,
    azure_endpoint=GPT5_2_ENDPOINT
)

class BMR_Yield_Reconciliation:
    """
    Check 50: Yield Value Verification
    ---
    Description:
    1. Identify reconciliation tables in BMR pages using keywords (Reconciliation, Yield).
    2. Structure the table data and send to Azure OpenAI (GPT-5.2).
    3. LLM extracts symbol-value mappings, Yield reported value, and formula expression.
    4. Python computes the yield using ast-based safe expression evaluation (deterministic).
    5. Flag anomaly if the difference exceeds the tolerance (0.1).
    ---
    Type: Hybrid (LLM extraction + Python computation)
    ---
    Attachment Required: NO
    ---
    Author: Mehul Kasliwal (13th February 2026)
    """

    # =====================================================
    # CONFIG
    # =====================================================
    AZURE_MODEL = "gpt-5.2"
    SECTION_NAME = "Manufacturing Process"
    CHECK_NAME = "Yield Value Verification"

    # Margin of error for numeric comparison (absolute tolerance)
    YIELD_TOLERANCE = 0.01

    # Safe operators for ast-based expression evaluation
    SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def __init__(self):
        self.client = client_gpt5_1

        self.prompt_template = """
You are an expert in pharmaceutical batch record reconciliation.

The following JSON is extracted from a BMR reconciliation table:

--- BEGIN JSON ---
{text_json}
--- END JSON ---

Your ONLY task is to EXTRACT (not compute) the following information:

1. Identify all stages that have a symbol (single uppercase letter: A, B, C, D, ...).
2. For each symbol, extract its reported numeric value from the Qty field.
   - If Qty is empty, blank, "00", "NA", "N/A", or missing, use 0.
   - Strip units like "kg", "%", "ml" etc. and return only the number.
3. Find the Yield stage (any stage whose name contains "Yield").
4. Extract the Yield's reported numeric value (strip the % sign).
5. Extract the Yield's formula and normalize it into a standard math expression:
   - Use only: +, -, *, /, ( )
   - Replace "X", "×", "x" with "*"
   - Remove "=", "[", "]", "%" 
   - CRITICAL BMR CONVENTION: In BMR formulas, "(G+J/A)" means "((G+J)/A)", NOT "G+(J/A)".
     The "/variable" at the end of a parenthesized group divides the ENTIRE group.
     Always rewrite to make this explicit with proper parentheses.
   - The final expression should evaluate to a percentage (0-100 range).
   - Example: "[(B+D) / A] X 100" → "((B+D)/A)*100"
   - Example: "(G+J/A) X100" → "((G+J)/A)*100"

Return STRICT JSON only:
{{
  "symbols": {{"A": <number>, "B": <number>, ...}},
  "yield_name": "<name of the yield stage>",
  "yield_reported_value": <reported numeric value as a number>,
  "yield_expression": "<normalized math expression using symbol letters>"
}}

Rules:
- Do NOT compute the yield value — only extract and normalize.
- All symbol values must be numbers (int or float), never strings.
- If no Yield stage is found, return: {{"symbols": {{}}, "yield_name": null, "yield_reported_value": null, "yield_expression": null}}
- Return ONLY the JSON, no explanation text.
"""

    # =====================================================
    # SECTION DETECTION
    # =====================================================

    def _find_reconciliation_section(self, page_content: Any, key_phrase: str) -> bool:
        """
        Search for reconciliation/yield section on the page by checking:
        1. 'rules_or_instructions' fields
        2. Table record Stage values (for keywords like YIELD, RECONCILIATION)
        3. Markdown content
        """
        phrase_lower = key_phrase.lower()

        # Check 1: rules_or_instructions (original approach)
        if self._search_rules_or_instructions(page_content, phrase_lower):
            return True

        # Check 2: Table Stage values — look for YIELD or RECONCILIATION in stage names
        if isinstance(page_content, list):
            for block in page_content:
                if isinstance(block, dict) and block.get("type") == "table":
                    table_json = block.get("table_json", {})
                    records = None
                    if isinstance(table_json, list):
                        records = table_json
                    elif isinstance(table_json, dict):
                        records = table_json.get("records")
                    if records:
                        for row in records:
                            stage = str(row.get("Stage", "")).lower()
                            if "yield" in stage or "reconciliation" in stage:
                                return True

        return False

    def _search_rules_or_instructions(self, obj: Any, phrase_lower: str) -> bool:
        """Recursively search for keyphrase in rules_or_instructions fields."""
        if isinstance(obj, dict):
            if "rules_or_instructions" in obj:
                rules = obj["rules_or_instructions"]
                if isinstance(rules, list):
                    for rule in rules:
                        if isinstance(rule, str) and phrase_lower in rule.lower():
                            return True
            return any(self._search_rules_or_instructions(v, phrase_lower) for v in obj.values())

        if isinstance(obj, list):
            return any(self._search_rules_or_instructions(i, phrase_lower) for i in obj)

        return False

    # =====================================================
    # TABLE FILTERING
    # =====================================================

    def _looks_like_reconciliation_table(self, records: List[Dict]) -> bool:
        """
        Heuristic check for reconciliation tables.
        """
        for row in records:
            stage = str(row.get("Stage", "")).upper()
            if "YIELD" in stage or "RECONCILIATION" in stage:
                return True
        return False

    def _get_reconciliation_records(self, page_content: List[Dict]) -> List[Dict]:
        """
        Extracts reconciliation table records only.
        """
        for block in page_content:
            if block.get("type") == "table":
                table_json = block.get("table_json", {})
                if isinstance(table_json, list):
                    records = table_json
                else:
                    records = table_json.get("records")
                if records and self._looks_like_reconciliation_table(records):
                    return records
        return []

    # =====================================================
    # LLM INPUT PREPARATION
    # =====================================================

    def _prepare_llm_input(self, records: List[Dict]) -> str:
        """
        Prepares clean structured JSON input for LLM.
        Uses case-insensitive keyword matching instead of exact keys.
        """
        def find_value(record: Dict, keywords: List[str]) -> str:
            """
            Finds value in record where key contains any keyword (case-insensitive)
            """
            for key, value in record.items():
                key_lower = key.lower()
                for kw in keywords:
                    if kw in key_lower:
                        return value
            return ""

        cleaned = []

        for r in records:
            cleaned.append({
                "Stage": find_value(r, ["stage"]),
                "Qty": find_value(r, ["qty", "quantity", "bags", "units"]),
                "Remarks": find_value(r, ["remark", "comment"]),
                "Page No": find_value(r, ["page"])
            })

        return json.dumps(cleaned, indent=2)

    # =====================================================
    # LLM CALL (extraction only)
    # =====================================================

    def _extract_with_llm(self, page_json: str) -> Dict:
        """Call LLM to extract symbols, expression and reported value (no computation)."""
        prompt = self.prompt_template.format(text_json=page_json)
        response = self.client.chat.completions.create(
            model=self.AZURE_MODEL,
            messages=[
                {"role": "system", "content": "You are a meticulous data extraction specialist. Extract data exactly as instructed. Do NOT perform any calculations."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    # =====================================================
    # SAFE EXPRESSION EVALUATION (Python-based)
    # =====================================================

    def _eval_node(self, node) -> float:
        """Recursively evaluate an AST node. Only arithmetic ops allowed."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_func = self.SAFE_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_node(node.operand)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self._eval_node(node.operand)
        else:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    def _safe_eval_expression(self, expression: str, symbols: Dict[str, float]) -> Optional[float]:
        """
        Safely evaluate a math expression by substituting symbol values
        and computing using Python's ast module (no eval/exec).
        
        Args:
            expression: Normalized math expression (e.g., "((B+D)/A)*100")
            symbols: Dict mapping symbol letters to numeric values (e.g., {"A": 5500, "B": 0.845})
        
        Returns:
            Computed float result, or None if evaluation fails.
        """
        try:
            expr = expression.strip()

            # Normalize multiplication symbols
            expr = re.sub(r'[xX×]', '*', expr)
            # Remove square brackets (treat as parentheses)
            expr = expr.replace('[', '(').replace(']', ')')
            # Remove whitespace
            expr = expr.replace(' ', '')

            # Substitute symbol values (sort by length desc to avoid partial matches)
            for sym in sorted(symbols.keys(), key=len, reverse=True):
                val = symbols[sym]
                expr = expr.replace(sym, str(float(val)))

            # Validate: only allow digits, operators, parentheses, and decimal points
            if not re.match(r'^[\d\.\+\-\*/\(\)]+$', expr):
                return None

            # Parse into AST and evaluate safely
            tree = ast.parse(expr, mode='eval')
            result = self._eval_node(tree.body)
            return round(result, 4)

        except Exception:
            return None

    # =====================================================
    # RESULT ANALYSIS
    # =====================================================

    def _determine_yield_status(self, llm_output: Dict) -> Tuple[int, Optional[float]]:
        """
        Determine anomaly status using Python-computed yield value.
        
        Returns:
            (anomaly_status, python_computed_value) tuple
            anomaly_status: 0 if yield matches (or no yield found), 1 if mismatch
            python_computed_value: the value computed by Python, or None
        """
        # If no yield stage was found by the LLM
        if llm_output.get("yield_name") is None:
            return 0, None

        expression = llm_output.get("yield_expression")
        symbols = llm_output.get("symbols", {})
        reported = llm_output.get("yield_reported_value")

        # If we don't have expression or reported value, can't verify
        if not expression or reported is None:
            return 0, None

        # Compute in Python
        computed = self._safe_eval_expression(expression, symbols)

        if computed is None:
            # Fallback: couldn't parse expression
            return 0, None

        # Compare with tolerance
        try:
            diff = abs(float(reported) - computed)
            if diff <= self.YIELD_TOLERANCE:
                return 0, computed  # Within tolerance
            else:
                return 1, computed  # Mismatch
        except (ValueError, TypeError):
            return 0, computed

    # =====================================================
    # MAIN ENTRY
    # =====================================================

    def run(self, document_pages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Yield checks across all pages.
        Returns:
            (results, return_debug) tuple where:
            - results: simplified output list
            - return_debug: detailed debug information list
        """
        results = []
        return_debug = []

        for idx, page in enumerate(document_pages, 1):
            page_no = page.get("page")
            anomaly = 0
            page_content = page.get("page_content", [])

            try:
                if not self._find_reconciliation_section(page_content, "RECONCILIATION"):
                    # Debug info — page has no reconciliation section
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": "No reconciliation section found on page"
                    })
                    results.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                records = self._get_reconciliation_records(page_content)
                if not records:
                    # Debug info — reconciliation section exists but no table records
                    return_debug.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0,
                        "skip_reason": "Reconciliation section found but no valid table records"
                    })
                    results.append({
                        "page_no": page_no,
                        "section_name": self.SECTION_NAME,
                        "check_name": self.CHECK_NAME,
                        "anomaly_status": 0
                    })
                    continue

                # Prepare input and call LLM (extraction only)
                llm_input = self._prepare_llm_input(records)
                llm_output = self._extract_with_llm(llm_input)

                # Determine anomaly status using Python computation
                anomaly, python_computed = self._determine_yield_status(llm_output)

                # Debug info — full analysis details
                debug_entry = {
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": anomaly,
                    "llm_input": llm_input,
                    "llm_output": llm_output,
                    "python_computed_value": python_computed,
                    "yield_reported_value": llm_output.get("yield_reported_value"),
                    "symbols": llm_output.get("symbols", {}),
                    "expression": llm_output.get("yield_expression"),
                }
                if anomaly == 1:
                    debug_entry["anomaly_detail"] = (
                        f"Yield mismatch — Reported: {llm_output.get('yield_reported_value')}, "
                        f"Python Computed: {python_computed}, "
                        f"Diff: {abs(float(llm_output.get('yield_reported_value', 0)) - (python_computed or 0)):.4f}, "
                        f"Expression: {llm_output.get('yield_expression')}"
                    )
                else:
                    debug_entry["skip_reason"] = "Yield value matches (within tolerance) or no yield found"
                return_debug.append(debug_entry)

            except Exception as e:
                # Debug info — error on this page
                return_debug.append({
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": 0,
                    "error": str(e)
                })
                anomaly = 0

            results.append({
                "page_no": page_no,
                "section_name": self.SECTION_NAME,
                "check_name": self.CHECK_NAME,
                "anomaly_status": anomaly
            })

        return results, return_debug

if __name__ == "__main__":
    import sys, os
    from pprint import pprint as pp
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "yog_checks_oops"))
    from extract_data import get_filled_master_json
    
    # Load OCR data from the complete_data JSON file
    complete_data_path = "/home/softsensor/Desktop/Amneal/challenge_bmr/05jan_AH250076_50Checks 1.json"
    
    print("Running Check 50 - BMR Yield Value Verification (Python computation)...")
    
    try:
        filled_master_json = get_filled_master_json(complete_data_path)
        print(f"Loaded {len(filled_master_json)} pages from filled_master_json")
        
        validator = BMR_Yield_Reconciliation()
        results, return_debug = validator.run(filled_master_json)
        
        print("\n=== RESULTS ===")
        print(json.dumps(results, indent=2))
        
        print("\n=== DEBUG INFO ===")
        print(json.dumps(return_debug, indent=2, default=str))
        
    except ImportError:
        print("Error: extract_data module not found. Please ensure it is in the same directory.")
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()