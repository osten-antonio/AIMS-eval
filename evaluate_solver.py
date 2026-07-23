import argparse
import json
import re
import requests
import pandas as pd
from pathlib import Path


def extract_number(text: str) -> float | None:
    """Extract the numerical value from text, handling various formats."""
    if not text:
        return None
    
    # Remove common prefixes/suffixes
    text = text.strip()
    text = text.replace("$", "").replace("€", "").replace("£", "")
    text = text.replace("%", "").replace("°", "")
    text = text.replace(",", "")  # Remove thousand separators
    
    # Handle time format (e.g., "9:50 PM" -> 950)
    time_match = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)?$", text, re.IGNORECASE)
    if time_match:
        hours = int(time_match.group(1))
        minutes = int(time_match.group(2))
        period = time_match.group(3)
        if period and period.upper() == "PM" and hours != 12:
            hours += 12
        elif period and period.upper() == "AM" and hours == 12:
            hours = 0
        return float(hours * 100 + minutes)
    

    
    # Handle fractions like "5/10"
    frac_match = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", text)
    if frac_match:
        num = float(frac_match.group(1))
        den = float(frac_match.group(2))
        if den != 0:
            return num / den
    
    # Handle "x = 5" or "solution = 5" format
    eq_matches = re.findall(r"(?:answer|result|x|y|value)\s*[=:]\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
    if eq_matches:
        return float(eq_matches[-1])

    # Find all numbers and return the last one (likely the final answer)
    matches = re.findall(r"-?\d+\.?\d*", text)
    if matches:
        return float(matches[-1])
    
    
    return None


def compare_answers(expected, solver_answer: str, tolerance: float = 0.01) -> bool:
    """Compare expected answer with solver answer, handling verbose responses."""
    expected_num = float(expected)
    solver_num = extract_number(solver_answer)
    if solver_num is None:
        return False
    
    # Use relative tolerance for larger numbers, absolute for small
    diff = abs(expected_num - solver_num)
    if abs(expected_num) > 100:
        # For large numbers, allow 5% relative error
        return diff <= 0.05 * abs(expected_num)
    elif abs(expected_num) > 1:
        # For medium numbers, allow 1% relative error
        return diff <= 0.01 * abs(expected_num)
    else:
        # For small numbers (decimals), use absolute tolerance
        return diff <= 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate solver API against QA pairs")
    parser.add_argument("--model", required=True, help="Model name for output file naming")
    parser.add_argument("--data", default="data/qa_pairs.json", help="Path to QA pairs JSON")
    parser.add_argument("--url", default="http://localhost:8000/solver/solve", help="Solver API URL")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Numerical comparison tolerance")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(data_path) as f:
        qa_pairs = json.load(f)

    results = []
    correct = 0
    total = len(qa_pairs)

    for pair in qa_pairs:
        try:
            resp = requests.post(args.url, json={"question": pair["latex"], "model": args.model}, timeout=30000)
            resp.raise_for_status()
            solver_data = resp.json()
            solver_answer = solver_data.get("answer", "")
        except Exception as e:
            solver_answer = f"ERROR: {e}"

        is_correct = compare_answers(pair["final_answer"], solver_answer, args.tolerance)
        if is_correct:
            correct += 1

        results.append({
            "qa_pair": pair,
            "solver_answer": solver_answer,
            "expected_answer": pair["final_answer"],
            "is_correct": is_correct,
        })

    flat_results = []
    for r in results:
        row = {
            "id": r["qa_pair"].get("id"),
            "text": r["qa_pair"].get("text"),
            "latex": r["qa_pair"].get("latex"),
            "final_answer": r["qa_pair"].get("final_answer"),
            "category": r["qa_pair"].get("category"),
            "solver_answer": r["solver_answer"],
            "expected_answer": r["expected_answer"],
            "is_correct": r["is_correct"],
        }
        flat_results.append(row)

    model_slug = args.model.replace("/", "_")
    output_file = output_dir / f"output_{model_slug}.csv"
    pd.DataFrame(flat_results).to_csv(output_file, index=False)

    accuracy = correct / total if total > 0 else 0
    summary = {
        "model": args.model,
        "total_questions": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(accuracy * 100, 2),
    }

    summary_file = output_dir / f"summary_{model_slug}.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)

    print(f"Accuracy: {summary['accuracy']}% ({correct}/{total})")
    print(f"Results saved to {output_file}")
    print(f"Summary saved to {summary_file}")