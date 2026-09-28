# generate_steps.py
import argparse
import json
import requests
import pandas as pd
from pathlib import Path

BASE_URL = "http://localhost:8000"

CATEGORY_MAP = {
    "arithmetic": "General",
    "algebra": "Algebra",
    "geometry": "General",
    "percentage": "General",
    "conversion": "General",
    "probability": "Statistics",
    "sequence": "General",
    "system": "Algebra",
    "statistics": "Statistics",
    "word_problem": "General",
}


def generate_steps(qa_pairs: list, limit: int, model: str):
    """Generates anchored and unanchored steps and saves to CSV via pandas."""
    results = []
    
    if limit:
        qa_pairs = qa_pairs[:limit]

    for i, pair in enumerate(qa_pairs):
        question = pair['Challenge']
        expected_answer = str(pair["Answer in Latex"])
        category = "General"        
        print(f"[{i+1}/{len(qa_pairs)}] Generating for: {question[:50]}...")

        try:
            resp_anchored = requests.post(
                f"{BASE_URL}/explanation/steps/",
                json={"question": question, "answer": expected_answer, "category": category, "forced": True, "model": model},
                timeout=300
            )
            anchored_data = resp_anchored.json()
        except Exception as e:
            print(f"  [Error] Anchored failed: {e}")
            anchored_data = {"error": str(e), "steps": []}

        try:
            resp_unanchored = requests.post(
                f"{BASE_URL}/explanation/steps/",
                json={"question": question, "answer": "", "category": category, "forced": True, "model": model},
                timeout=300
            )
            unanchored_data = resp_unanchored.json()
        except Exception as e:
            print(f"  [Error] Unanchored failed: {e}")
            unanchored_data = {"error": str(e), "steps": []}

        # Serialize steps to JSON strings so they fit in pandas CSV cells
        results.append({
            "id": pair.get("ID", i),
            "question": question,
            "expected_answer": expected_answer,
            "anchored_steps": json.dumps(anchored_data.get("steps", [])),
            "unanchored_steps": json.dumps(unanchored_data.get("steps", []))
        })

    # Create output directory and save via pandas
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f"explanations_ASyMOB_{model.replace('/', '_').replace(':', '_')}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nSaved raw generations to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ASyMOB_200_unique.json")
    parser.add_argument("--model", required=True, help="Model name for output file naming")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    
    with open(args.data) as f:
        qa_pairs = json.load(f)
          
    generate_steps(qa_pairs, args.limit, args.model)
