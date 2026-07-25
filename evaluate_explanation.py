# evaluate_hybrid.py
import argparse
import json
import re
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
def extract_final_answer(steps: list) -> Optional[float]:
    """Extract the model's final numerical answer from steps."""
    if not steps:
        return None
    
    # Patterns that explicitly declare an answer
    explicit_patterns = [
        r'(?:final\s+)?(?:answer|result)\s*(?:is|:=|:)\s*(-?\d+\.?\d*)',
        r'(?:therefore|thus|hence|so|finally)\s*,?\s*(?:the\s+)?(?:answer\s+is\s+)?(-?\d+\.?\d*)',
        r'=\s*(-?\d+\.?\d*)\s*$',
    ]
    
    # Search all steps back to front for explicit answer
    for step in reversed(steps):
        text = step.get("explanation", "") + " " + step.get("equation", "")
        for pattern in explicit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    
    # Fallback: last number from last step
    last_text = steps[-1].get("explanation", "") + " " + steps[-1].get("equation", "")
    numbers = re.findall(r'-?\d+\.?\d*', last_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


def check_answer_match(model_answer: Optional[float], expected_answer, tolerance: float = 0.01) -> bool:
    """Check if model's answer matches expected."""
    if model_answer is None:
        return False
    
    # Parse expected
    if isinstance(expected_answer, str):
        try:
            expected = float(expected_answer)
        except ValueError:
            return False
    else:
        expected = float(expected_answer)
    
    diff = abs(model_answer - expected)
    if abs(expected) > 100:
        return diff <= 0.05 * abs(expected)
    elif abs(expected) > 1:
        return diff <= 0.02 * abs(expected)
    else:
        return diff <= 0.1

_transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

def clean_equation(eq_str: str) -> str:
    s = eq_str.strip()
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    s = s.replace('\\times', '*').replace('\\div', '/').replace('\\cdot', '*')
    s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s) # remove leading variables like "t = "
    s = re.sub(r'\\text\{([^{}]+)\}', r'\1', s) # handle \text{}
    s = s.replace('\\propto', '~~') # proportional symbol ruins sympy
    s = s.replace('\\', '').replace('{', '(').replace('}', ')')
    s = s.replace('^', '**').replace('$', '')
    return s.strip()

def try_eval(expr_str: str) -> Optional[float]:
    try:
        expr = parse_expr(expr_str, transformations=_transformations, evaluate=True)
        val = float(expr.evalf())
        return val if not (abs(val) > 1e12) else None
    except Exception:
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
        
        # --- FIXED ARITHMETIC CHECK ---
        if '=' in cleaned:
            parts = cleaned.split('=')
            # Evaluate every part of the equation
            vals = [try_eval(p.strip()) for p in parts]
            # Filter out the ones that failed (e.g., variables like 'd' or 't')
            numbers = [v for v in vals if v is not None]
            
            # If there are at least 2 actual numbers, they MUST be equal
            # e.g., "80 * 3 = 240" -> [240.0, 240.0] (OK)
            # e.g., "80 * 3 = 250" -> [240.0, 250.0] (ERROR)
            if len(numbers) >= 2:
                if not all(abs(numbers[0] - n) <= 0.01 for n in numbers[1:]):
                    arith_errors += 1
                    
        # Check provenance
        step_nums = set(float(m.group()) for m in re.finditer(r'-?\d+\.?\d*', combined))
        for n in step_nums:
            if not any(abs(n - av) <= 0.01 for av in available):
                phantom_errors += 1
                available.add(n) # prevent duplicate flagging
                
        # Add outputs to available
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
        # llama.cpp server uses OpenAI-compatible /v1/chat/completions endpoint
        resp = requests.post(
            f"{judge_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"} # Enforce JSON output if supported
            },
            timeout=1200
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

def evaluate_file(input_file: Path, judge_url: str, tolerance: float = 0.01):
    # Read via pandas
    df = pd.read_csv(input_file)
    results = []
    
    for _, item in df.iterrows():
        q = item["question"]
        ans = str(item["expected_answer"])
        
        for condition in ["anchored", "unanchored"]:
            # Deserialize steps from CSV string back to Python list
            steps_str = item[f"{condition}_steps"]
            if pd.isna(steps_str) or not steps_str:
                # Empty steps = no answer produced
                model_answer = None
                answer_correct = False
                sympy_rep = {"arith_errors": 0, "phantom_errors": 0, "total_steps": 0}
                judge_rep = {"semantic_errors": [], "hallucination_flag": True, "overall_reasoning_score": 1}
            else:
                steps = json.loads(steps_str)
                
                # Extract and check final answer
                model_answer = extract_final_answer(steps)
                answer_correct = check_answer_match(model_answer, item["expected_answer"], tolerance)
                
                # Run Sympy
                sympy_rep = validate_chain_deterministic(steps, q)
                
                # Run Llama.cpp Judge
                judge_rep = llm_judge(steps, q, ans, sympy_rep, judge_url)
                print(f"    [Judge Result] Semantic Errors: {len(judge_rep.get('semantic_errors', []))}, Hallucination: {judge_rep.get('hallucination_flag')}")
            print(f"  [{condition}] Item {item['id']}: answer={model_answer}, correct={answer_correct}")
            
            results.append({
                "id": item["id"],
                "condition": condition,
                "expected_answer": item["expected_answer"],
                "model_answer": model_answer,
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
    args = parser.parse_args()
    
    print(f"Running Hybrid Evaluation (SymPy + LLM Judge) on {args.input}...")
    eval_results = evaluate_file(Path(args.input), args.judge_url, args.tolerance)
    
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
    out_csv = Path(args.input).with_suffix(".eval.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nDetailed results saved to: {out_csv}")