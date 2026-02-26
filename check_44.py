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

# Initialize the client
client_gpt5_1 = AzureOpenAI(
    api_key=GPT5_2_API_KEY,
    api_version=GPT5_2_API_VERSION,
    azure_endpoint=GPT5_2_ENDPOINT
)


class BMR_Reconciliation:
    """
    Check 44: Reconciliation Formula Check
    ---
    Description:
    1. Identify reconciliation tables in BMR pages using keywords (Reconciliation, Yield).
    2. Structure the table data and send to Azure OpenAI for extraction.
    3. LLM extracts symbol-value mappings and formula expressions for reconciliation stages.
    4. Python computes every expression using ast-based safe evaluation (deterministic).
    5. Compares Python-computed values against reported values for reconciliation stages.
    6. Flag anomaly if any reconciliation formula has a mismatch beyond tolerance.
    ---
    Type: Hybrid (LLM extraction + Python computation)
    ---
    Attachment Required: NO
    ---
    Author: Mehul Kasliwal (17th February 2026)
    """

    # =====================================================
    # CONFIG
    # =====================================================
    AZURE_MODEL = "gpt-5-mini"
    SECTION_NAME = "Manufacturing Process"
    CHECK_NAME = "Reconciliation Formula Check"

    # Margin of error for numeric comparison (absolute tolerance)
    TOLERANCE = 0.01

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
3. For EACH stage that contains a formula/expression (e.g., "E = A - (B + C + D)"):
   - Extract and normalize the expression into standard math notation: +, -, *, /, ( )
   - Replace "X", "×", "x" with "*"
   - Remove "=", "[", "]", "%"
   - CRITICAL BMR CONVENTION: In BMR formulas, "(G+J/A)" means "((G+J)/A)", NOT "G+(J/A)".
     The "/variable" at the end of a parenthesized group divides the ENTIRE group.
     Always rewrite to make this explicit with proper parentheses.
   - CRITICAL LaTeX HANDLING: If a stage name contains LaTeX like \\frac{{numerator}}{{denominator}},
     convert it to (numerator)/(denominator) FIRST. BUT if the numerator already contains "*100"
     and the denominator is just "100", this is redundant — the "*100/100" cancels out.
     In such cases, DROP the "/100" denominator entirely. Example:
     "\\frac{{[(B+C+E+F+G+H)/A] \\times 100}}{{100}}" → the formula is "((B+C+E+F+G+H)/A)*100",
     NOT "(((B+C+E+F+G+H)/A)*100)/100". The result should be a percentage (0-100 range).
   - Example: "E = A - (B + C + D)" → expression is "A-(B+C+D)"
   - Example: "[(B+D) / A] X 100" → expression is "((B+D)/A)*100"
   - Example: "(G+J/A) X100" → expression is "((G+J)/A)*100"
   - Example: "\\frac{{[(B+C+E+F+G+H)/A] \\times 100}}{{100}}" → expression is "((B+C+E+F+G+H)/A)*100"
4. Mark whether each stage contains "reconciliation" in its name (is_reconciliation: true/false).

Return STRICT JSON only:
{{
  "symbols": {{"A": <number>, "B": <number>, ...}},
  "stages": [
    {{
      "name": "<stage name>",
      "symbol": "<symbol letter or null>",
      "reported_value": <number or null>,
      "expression": "<normalized math expression using symbol letters, or null if no formula>",
      "is_reconciliation": true | false
    }}
  ]
}}

Rules:
- Do NOT compute any values — only extract and normalize.
- All symbol values in "symbols" must be numbers (int or float), never strings.
- Only include stages that have a symbol assigned.
- "is_reconciliation" should be true if the stage name contains the word "reconciliation" (case-insensitive).
- If no reconciliation stages are found, return: {{"symbols": {{}}, "stages": []}}
- Return ONLY the JSON, no explanation text.
"""

    # =====================================================
    # SECTION DETECTION
    # =====================================================

    def _find_reconciliation_section(self, page_content: Any, key_phrase: str) -> bool:
        """
        Search for reconciliation/yield section on the page by checking:
        1. 'rules_or_instructions' fields
        2. Table record Stage values
        """
        phrase_lower = key_phrase.lower()

        if self._search_rules_or_instructions(page_content, phrase_lower):
            return True

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
        """Heuristic check for reconciliation tables."""
        for row in records:
            stage = str(row.get("Stage", "")).upper()
            if "YIELD" in stage or "RECONCILIATION" in stage:
                return True
        return False

    def _get_reconciliation_records(self, page_content: List[Dict]) -> List[Dict]:
        """Extracts reconciliation table records only."""
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
        """Prepares clean structured JSON input for LLM."""
        def find_value(record: Dict, keywords: List[str]) -> str:
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
        """Call LLM to extract symbols, expressions and reported values (no computation)."""
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
    # RESULT ANALYSIS (reconciliation stages only)
    # =====================================================

    def _verify_reconciliation_stages(self, llm_output: Dict) -> Tuple[int, List[Dict]]:
        """
        Verify ONLY reconciliation stages using Python computation.
        Unlike check_40 which checks ALL stages, this check focuses on
        stages whose name contains "reconciliation".

        Returns:
            (anomaly_status, verification_details) tuple
        """
        symbols = llm_output.get("symbols", {})
        stages = llm_output.get("stages", [])

        if not stages or not symbols:
            return 0, []

        verification_details = []
        any_mismatch = False

        for stage in stages:
            expression = stage.get("expression")
            reported = stage.get("reported_value")
            name = stage.get("name", "Unknown")
            symbol = stage.get("symbol")
            is_reconciliation = stage.get("is_reconciliation", False)

            detail = {
                "name": name,
                "symbol": symbol,
                "reported_value": reported,
                "expression": expression,
                "is_reconciliation": is_reconciliation,
                "python_computed": None,
                "match": None,
                "diff": None,
            }

            # Only verify reconciliation stages
            if not is_reconciliation:
                detail["skip_reason"] = "Not a reconciliation stage"
                detail["match"] = True
                verification_details.append(detail)
                continue

            # Skip stages without expressions
            if not expression:
                detail["match"] = True
                detail["skip_reason"] = "No expression to verify"
                verification_details.append(detail)
                continue

            # Skip if no reported value to compare against
            if reported is None:
                detail["match"] = True
                detail["skip_reason"] = "No reported value to compare"
                verification_details.append(detail)
                continue

            # Compute in Python
            computed = self._safe_eval_expression(expression, symbols)
            detail["python_computed"] = computed

            if computed is None:
                detail["skip_reason"] = "Could not evaluate expression"
                detail["match"] = True
                verification_details.append(detail)
                continue

            # Compare with tolerance
            try:
                diff = abs(float(reported) - computed)
                detail["diff"] = round(diff, 4)
                if diff <= self.TOLERANCE:
                    detail["match"] = True
                else:
                    detail["match"] = False
                    any_mismatch = True
            except (ValueError, TypeError):
                detail["match"] = True

            verification_details.append(detail)

        anomaly = 1 if any_mismatch else 0
        return anomaly, verification_details

    # =====================================================
    # MAIN ENTRY
    # =====================================================

    def run(self, document_pages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Reconciliation Formula checks across all pages.
        Returns:
            (results, return_debug) tuple
        """
        results = []
        return_debug = []

        for idx, page in enumerate(document_pages, 1):
            page_no = page.get("page")
            anomaly = 0
            page_content = page.get("page_content", [])

            try:
                if not self._find_reconciliation_section(page_content, "RECONCILIATION"):
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

                # Verify reconciliation stages using Python computation
                anomaly, verification_details = self._verify_reconciliation_stages(llm_output)

                # Debug info
                debug_entry = {
                    "page_no": page_no,
                    "section_name": self.SECTION_NAME,
                    "check_name": self.CHECK_NAME,
                    "anomaly_status": anomaly,
                    "llm_input": llm_input,
                    "llm_output": llm_output,
                    "verification_details": verification_details,
                }

                mismatches = [d for d in verification_details if d.get("match") is False]
                if mismatches:
                    mismatch_strs = []
                    for m in mismatches:
                        mismatch_strs.append(
                            f"{m['name']} ({m['symbol']}): "
                            f"reported={m['reported_value']}, "
                            f"computed={m['python_computed']}, "
                            f"diff={m['diff']}"
                        )
                    debug_entry["anomaly_detail"] = "; ".join(mismatch_strs)
                else:
                    debug_entry["skip_reason"] = "All reconciliation stages match (within tolerance) or none found"
                return_debug.append(debug_entry)

            except Exception as e:
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

    # complete_data_path = "/home/softsensor/Desktop/Amneal/all_result_76_20feb.json"
    complete_data_path = "/home/softsensor/Desktop/Amneal/New_BMRs/Emulsion_line_AH240074_filled_master_data.json"

    print("Running Check 44 - Reconciliation Formula Check (Python computation)...")

    try:
        # This JSON already contains the filled_master_json directly (list of page dicts)
        with open(complete_data_path, "r", encoding="utf-8") as f:
            filled_master_json = json.load(f)
        print(f"Loaded {len(filled_master_json)} pages from filled_master_json")

        validator = BMR_Reconciliation()
        results, return_debug = validator.run(filled_master_json)

        # Show only LLM-processed pages
        print("\n=== LLM-PROCESSED PAGES ===")
        for d in return_debug:
            if "llm_output" in d:
                print(f"\nPage {d['page_no']} — anomaly={d['anomaly_status']}")
                for v in d.get("verification_details", []):
                    if v.get("is_reconciliation") and v.get("expression"):
                        status = "✓" if v["match"] else "✗"
                        print(f"  {status} {v['name'][:70]} ({v['symbol']}): "
                              f"reported={v['reported_value']}, "
                              f"computed={v['python_computed']}, "
                              f"expr={v['expression']}, "
                              f"diff={v['diff']}")
                if "anomaly_detail" in d:
                    print(f"  ANOMALY: {d['anomaly_detail']}")

        print("\n=== FULL RESULTS ===")
        print(json.dumps(results, indent=2))

    except ImportError:
        print("Error: extract_data module not found.")
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()