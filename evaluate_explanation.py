# evaluate_hybrid.py
import argparse
import json
import re
import signal
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Union

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

_transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

# Symbols used in the problem set
_x, _A, _B, _J, _G, _K = sympy.symbols('x A B J G K')
_FREE_SYMBOLS = {_x, _A, _B, _J, _G, _K}


def latex_to_sympy(latex: str) -> Optional[sympy.Expr]:
    """Convert a LaTeX math expression to a sympy expression."""
    s = latex.strip()
    # Remove display math delimiters
    s = s.replace('$', '').replace('\\(', '').replace('\\)', '')
    # Remove \left \right
    s = s.replace('\\left', '').replace('\\right', '')
    # \sqrt{x} -> sqrt(x)  (before frac to avoid conflicts)
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
    # \operatorname{xxx} -> xxx
    s = re.sub(r'\\operatorname\{([^{}]+)\}', r'\1', s)
    # Named functions (before removing backslashes!)
    for fn in ['log', 'ln', 'exp', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
               'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'atanh']:
        s = s.replace(f'\\{fn}', fn)
    # \pi -> pi
    s = s.replace('\\pi', 'pi')
    # Operators
    s = s.replace('\\times', '*').replace('\\div', '/').replace('\\cdot', '*')
    s = s.replace('\\propto', '')
    # Remove \text{...}
    s = re.sub(r'\\text\{([^{}]+)\}', r'\1', s)
    # Handle \frac with brace matching 
    while '\\frac' in s:
        idx = s.find('\\frac')
        if idx == -1:
            break
        start = idx + 5  # len('\\frac')
        if start >= len(s) or s[start] != '{':
            break
        depth = 0
        i = start
        while i < len(s):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            break
        numerator = s[start+1:i]
        denom_start = i + 1
        if denom_start >= len(s) or s[denom_start] != '{':
            break
        depth = 0
        j = denom_start
        while j < len(s):
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            break
        denominator = s[denom_start+1:j]
        s = s[:idx] + f'(({numerator})/({denominator}))' + s[j+1:]
    # remove remaining backslashes
    s = s.replace('\\', '')
    # Brace's to parens
    s = s.replace('{', '(').replace('}', ')')
    # Power
    s = s.replace('^', '**')
    # Remove trailing whitespace
    s = s.strip()
    # Remove a leading variable assignment like "y = " or "f(x) = "
    s = re.sub(r'^[a-zA-Z](?:\([^)]*\))?\s*=\s*', '', s)
    # Remove trailing \, dx or dx (integral differential notation)
    s = re.sub(r'[,\\]?,?\s*dx\s*$', '', s)
    # Remove absolute value bars |...| -> abs(...)
    s = re.sub(r'\|([^|]+)\|', r'abs(\1)', s)
    if not s:
        return None
    try:
        old_handler = signal.signal(signal.SIGALRM, _eval_alarm_handler)
        signal.alarm(2)
        try:
            expr = parse_expr(s, transformations=_transformations, local_dict={
                'x': _x, 'A': _A, 'B': _B, 'J': _J, 'G': _G, 'K': _K,
                'pi': sympy.pi, 'e': sympy.E,
            })
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return expr
    except Exception:
        signal.alarm(0)
        return None


def extract_final_equation(steps: list) -> Optional[str]:
    """Extract the model's final equation from the last step.

    Returns the raw equation string from the last step, preferring the
    equation field over the explanation field.
    """
    if not steps:
        return None
    last = steps[-1]
    eq = last.get("equation", "").strip()
    if eq:
        return eq
    return last.get("explanation", "").strip() or None


def extract_final_answer_numeric(steps: list) -> Optional[float]:
    """Extract a numeric final answer from steps (for backward compat)."""
    if not steps:
        return None
    explicit_patterns = [
        r'(?:final\s+)?(?:answer|result)\s*(?:is|:=|:)\s*(-?\d+\.?\d*)',
        r'(?:therefore|thus|hence|so|finally)\s*,?\s*(?:the\s+)?(?:answer\s+is\s+)?(-?\d+\.?\d*)',
        r'=\s*(-?\d+\.?\d*)\s*$',
    ]
    for step in reversed(steps):
        text = step.get("explanation", "") + " " + step.get("equation", "")
        for pattern in explicit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    last_text = steps[-1].get("explanation", "") + " " + steps[-1].get("equation", "")
    numbers = re.findall(r'-?\d+\.?\d*', last_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None


def _test_points_for_symbols(symbols: set) -> list:
    """Generate test point dicts for free symbols in an expression."""
    if not symbols:
        return [{}]
    test_vals = {
        _x: [1.0, 2.0],
        _A: [1.0], _B: [1.0], _J: [1.0], _G: [1.0], _K: [1.0],
    }
    from itertools import product
    lists = []
    for sym in sorted(symbols, key=str):
        lists.append(test_vals.get(sym, [1.0]))
    return [dict(zip([s for s in sorted(symbols, key=str)], vals))
            for vals in product(*lists)]


class _EvalTimeout(Exception):
    pass

def _eval_alarm_handler(signum, frame):
    raise _EvalTimeout()


def _sympy_evalf_timeout(expr, timeout=2):
    """Evaluate a sympy expression to float with a timeout."""
    old_handler = signal.signal(signal.SIGALRM, _eval_alarm_handler)
    signal.alarm(timeout)
    try:
        val = float(expr.evalf())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return val


def check_answer_match(model_equation: Optional[str], expected_answer, tolerance: float = 0.01) -> bool:
    """Check if model's final equation matches the expected answer.

    Uses sympy symbolic comparison. Falls back to numeric evaluation at
    test points when both expressions have free variables.
    """
    if model_equation is None:
        return False

    # Parse expected
    expected_str = str(expected_answer).strip()
    expected_expr = latex_to_sympy(expected_str)
    if isinstance(expected_expr, tuple):
        expected_expr = None
    if expected_expr is None:
        # Can't parse expected — fall back to old numeric check
        try:
            expected_val = float(expected_str)
        except ValueError:
            return False
        model_num = extract_final_answer_numeric([{"explanation": "", "equation": model_equation}])
        if model_num is None:
            return False
        return abs(model_num - expected_val) <= max(tolerance, 0.1 * abs(expected_val))

    # Parse model equation: try the RHS of "LHS = RHS", or the whole thing
    model_str = model_equation.strip()
    # Skip very long equations (>500 chars) to avoid sympy timing out issue
    if len(model_str) > 500:
        return False
    model_expr = None
    # First try latex_to_sympy directly on the raw equation
    model_expr = latex_to_sympy(model_str)
    if isinstance(model_expr, tuple):
        model_expr = None
    if model_expr is None and '=' in model_str:
        parts = model_str.split('=')
        for p in reversed(parts):
            model_expr = latex_to_sympy(p.strip())
            if isinstance(model_expr, tuple):
                model_expr = None
            if model_expr is not None and hasattr(model_expr, 'free_symbols') and isinstance(model_expr, sympy.Expr):
                break
        else:
            model_expr = None

    if model_expr is None or not hasattr(model_expr, 'free_symbols') or not isinstance(model_expr, sympy.Expr):
        # Can't parse model output
        return False

    # Both are pure numbers (no free symbols)
    exp_syms = expected_expr.free_symbols - {sympy.pi, sympy.E}
    mod_syms = model_expr.free_symbols - {sympy.pi, sympy.E}
    if not exp_syms and not mod_syms:
        try:
            ev = _sympy_evalf_timeout(expected_expr)
            mv = _sympy_evalf_timeout(model_expr)
        except (_EvalTimeout, Exception):
            return False
        diff = abs(mv - ev)
        if abs(ev) > 1e10 or abs(mv) > 1e10:
            if abs(ev) > 0:
                return diff / abs(ev) < 0.001
            return diff == 0
        elif abs(ev) > 100:
            return diff <= 0.05 * abs(ev)
        elif abs(ev) > 1:
            return diff <= 0.02 * abs(ev)
        elif abs(ev) > 0.01:
            return diff <= 0.005
        else:
            return diff <= 0.001

    # Same free variables
    all_syms = (exp_syms | mod_syms) & _FREE_SYMBOLS
    test_points = _test_points_for_symbols(all_syms)
    # Limit test points to avoid timeouts
    if len(test_points) > 4:
        test_points = test_points[:4]
    matches = 0
    evaluated = 0
    for pt in test_points:
        try:
            ev = _sympy_evalf_timeout(expected_expr.subs(pt), timeout=2)
            mv = _sympy_evalf_timeout(model_expr.subs(pt), timeout=2)
        except (_EvalTimeout, Exception):
            continue
        evaluated += 1
        diff = abs(mv - ev)
        # For very large values, check relative error
        if abs(ev) > 1e10 or abs(mv) > 1e10:
            if abs(ev) > 0:
                ok = diff / abs(ev) < 0.001
            else:
                ok = diff == 0
        elif abs(ev) > 100:
            ok = diff <= 0.05 * abs(ev)
        elif abs(ev) > 1:
            ok = diff <= 0.02 * abs(ev)
        elif abs(ev) > 0.01:
            ok = diff <= 0.005
        else:
            ok = diff <= 0.001
        if ok:
            matches += 1
    if evaluated > 0:
        return matches == evaluated
    #   if all test points failed, try symbolic equality
    if exp_syms == mod_syms:
        try:
            old_handler = signal.signal(signal.SIGALRM, _eval_alarm_handler)
            signal.alarm(3)
            try:
                diff = sympy.simplify(expected_expr - model_expr)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            if diff == 0:
                return True
        except Exception:
            signal.alarm(0)
    return False


def clean_equation(eq_str: str) -> str:
    """Clean an equation string for sympy parsing (used for step-level checks)."""
    s = eq_str.strip()
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    s = s.replace('\\times', '*').replace('\\div', '/').replace('\\cdot', '*')
    s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
    s = re.sub(r'\\text\{([^{}]+)\}', r'\1', s)
    s = s.replace('\\propto', '')
    s = s.replace('\\', '').replace('{', '(').replace('}', ')')
    s = s.replace('^', '**').replace('$', '')
    return s.strip()


def try_eval(expr_str: str) -> Optional[float]:
    try:
        old_handler = signal.signal(signal.SIGALRM, _eval_alarm_handler)
        signal.alarm(3)
        try:
            expr = parse_expr(expr_str, transformations=_transformations, evaluate=True)
            val = float(expr.evalf())
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return val if not (abs(val) > 1e12) else None
    except Exception:
        signal.alarm(0)
        return None


def validate_chain_deterministic(steps: list, question: str) -> dict:
    """Checks arithmetic and number provenance."""
    available = set()
    for m in re.finditer(r'-?\d+\.?\d*', question):
        try: available.add(float(m.group()))
        except: pass
        
    arith_errors = 0
    phantom_errors = 0
    
    for step in steps:
        combined = step.get("explanation", "") + " " + step.get("equation", "")
        cleaned = clean_equation(combined)
        
        if '=' in cleaned:
            parts = cleaned.split('=')
            vals = [try_eval(p.strip()) for p in parts]
            numbers = [v for v in vals if v is not None]
            if len(numbers) >= 2:
                if not all(abs(numbers[0] - n) <= 0.01 for n in numbers[1:]):
                    arith_errors += 1
                    
        step_nums = set(float(m.group()) for m in re.finditer(r'-?\d+\.?\d*', combined))
        for n in step_nums:
            if not any(abs(n - av) <= 0.01 for av in available):
                phantom_errors += 1
                available.add(n)
        for n in step_nums:
            available.add(n)
            
    return {
        "arith_errors": arith_errors,
        "phantom_errors": phantom_errors,
        "total_steps": len(steps)
    }

def llm_judge(steps: list, question: str, expected_answer: str, sympy_report: dict, judge_url: str) -> dict:
    """Uses llama.cpp server to check semantic logic."""
    
    steps_text = "\n".join([f"{s.get('step', '?')}. {s.get('explanation', '')} (Equation: {s.get('equation', '')})" for s in steps])

    prompt = f"""You are a strict mathematical evaluator. Evaluate the student's step-by-step LOGIC and SEMANTIC correctness. 
Ignore basic arithmetic errors (the automated checker already found them). 
Determine if the student chose the correct mathematical operations to solve the specific problem asked.

Question: {question}
Expected Final Answer: {expected_answer}

Student's Steps:
{steps_text}

Automated Checker Findings:
- Arithmetic errors detected: {sympy_report['arith_errors']}
- Unsourced numbers detected: {sympy_report['phantom_errors']}

Output a JSON object with:
{{
  "semantic_errors": [list of step numbers where the logic/operation was wrong],
  "hallucination_flag": boolean (true if the model fabricated logic or context not present in the problem),
  "overall_reasoning_score": 1-5 (1=entirely flawed logic, 5=perfect logical progression)
}}
"""
    try:
        resp = requests.post(
            f"{judge_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"} # Enforce JSON output if supported
            },
            timeout=300
        )
        content = resp.json()["choices"][0]["message"]["content"]
        print(f"    [DEBUG Judge Raw Output]: {content[:200]}")
        # Fallback regex extraction just in case the model wraps JSON in markdown
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(content)
    except Exception as e:
        print(f"  [Judge Error] {e}")
        return {"error": str(e), "semantic_errors": [], "hallucination_flag": True, "overall_reasoning_score": 1}

def evaluate_file(input_file: Path, judge_url: str, tolerance: float = 0.01, skip_judge: bool = False):
    df = pd.read_csv(input_file)
    results = []
    
    for i, (_, item) in enumerate(df.iterrows()):
        q = item["question"]
        ans = str(item["expected_answer"])
        print(f"\n[{i+1}/{len(df)}] ID {item['id']}")
        
        for condition in ["anchored", "unanchored"]:
            steps_str = item[f"{condition}_steps"]
            if pd.isna(steps_str) or not steps_str:
                model_answer_num = None
                model_answer_eq = None
                answer_correct = False
                sympy_rep = {"arith_errors": 0, "phantom_errors": 0, "total_steps": 0}
                judge_rep = {"semantic_errors": [], "hallucination_flag": True, "overall_reasoning_score": 1}
            else:
                steps = json.loads(steps_str)
                
                # Extract final equation (full expression) and numeric answer
                model_answer_eq = extract_final_equation(steps)
                model_answer_num = extract_final_answer_numeric(steps)
                answer_correct = check_answer_match(model_answer_eq, item["expected_answer"], tolerance)
                
                sympy_rep = validate_chain_deterministic(steps, q)
                
                if skip_judge:
                    judge_rep = {"semantic_errors": [], "hallucination_flag": False, "overall_reasoning_score": None}
                else:
                    judge_rep = llm_judge(steps, q, ans, sympy_rep, judge_url)
                print(f"    [Judge Result] Semantic Errors: {len(judge_rep.get('semantic_errors', []))}, Hallucination: {judge_rep.get('hallucination_flag')}")
            
            display_answer = model_answer_eq[:60] if model_answer_eq else str(model_answer_num)
            print(f"  [{condition}] Item {item['id']}: answer={display_answer}, correct={answer_correct}")
            
            results.append({
                "id": item["id"],
                "condition": condition,
                "expected_answer": item["expected_answer"],
                "model_answer_numeric": model_answer_num,
                "model_answer_equation": model_answer_eq,
                "answer_correct": answer_correct,
                "total_steps": sympy_rep["total_steps"],
                "sympy_arith_errors": sympy_rep["arith_errors"],
                "sympy_phantom_errors": sympy_rep["phantom_errors"],
                "judge_semantic_errors": len(judge_rep.get("semantic_errors", [])),
                "judge_hallucination": judge_rep.get("hallucination_flag", True),
                "judge_score": judge_rep.get("overall_reasoning_score", 1)
            })
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to generated_steps.csv")
    parser.add_argument("--judge-url", default="http://localhost:11434", help="URL for llama.cpp server (e.g., http://localhost:11434)")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Numerical comparison tolerance")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge evaluation (deterministic SymPy only)")
    args = parser.parse_args()
    
    print(f"Running Hybrid Evaluation (SymPy + LLM Judge) on {args.input}...")
    eval_results = evaluate_file(Path(args.input), args.judge_url, args.tolerance, args.skip_judge)
    
    # Split by condition
    anchored_res = [r for r in eval_results if r["condition"] == "anchored"]
    unanchored_res = [r for r in eval_results if r["condition"] == "unanchored"]
    
    total = len(anchored_res)
    anchored_correct = sum(1 for r in anchored_res if r["answer_correct"])
    unanchored_correct = sum(1 for r in unanchored_res if r["answer_correct"])
    hallucinated = sum(1 for r in unanchored_res if r["judge_hallucination"] and not r["answer_correct"])
    
    anchored_pct = (anchored_correct / total * 100) if total > 0 else 0
    unanchored_pct = (unanchored_correct / total * 100) if total > 0 else 0
    hallucination_pct = (hallucinated / total * 100) if total > 0 else 0
    improvement = anchored_pct - unanchored_pct
    
    print("\nFinal answer accuracy")
    print(f"With answer (anchored):    {anchored_correct}/{total} = {anchored_pct:.2f}%")
    print(f"Without answer (unanchored): {unanchored_correct}/{total} = {unanchored_pct:.2f}%")
    print(f"Hallucinations without answer: {hallucinated}/{total} = {hallucination_pct:.2f}%")
    print(f"Accuracy improvement from anchoring: +{improvement:.2f}%")

    print("\nSymPy and LLM Judge Metrics:")
    a_arith = sum(r["sympy_arith_errors"] for r in anchored_res) / max(1, sum(r["total_steps"] for r in anchored_res))
    u_arith = sum(r["sympy_arith_errors"] for r in unanchored_res) / max(1, sum(r["total_steps"] for r in unanchored_res))
    print(f"Arithmetic Error Rate: Anchored={a_arith:.2f} vs Unanchored={u_arith:.2f}")
    
    a_phantom = sum(r["sympy_phantom_errors"] for r in anchored_res) / max(1, sum(r["total_steps"] for r in anchored_res))
    u_phantom = sum(r["sympy_phantom_errors"] for r in unanchored_res) / max(1, sum(r["total_steps"] for r in unanchored_res))
    print(f"Phantom Number Rate:   Anchored={a_phantom:.2f} vs Unanchored={u_phantom:.2f}")
    
    a_sem = sum(r["judge_semantic_errors"] for r in anchored_res) / max(1, sum(r["total_steps"] for r in anchored_res))
    u_sem = sum(r["judge_semantic_errors"] for r in unanchored_res) / max(1, sum(r["total_steps"] for r in unanchored_res))
    print(f"Semantic Error Rate:   Anchored={a_sem:.2f} vs Unanchored={u_sem:.2f}")

    # Save detailed CSV via pandas
    df_out = pd.DataFrame(eval_results)
    out_csv = Path(args.input).with_name(Path(args.input).stem + ".ASyMOB.eval.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nDetailed results saved to: {out_csv}")
