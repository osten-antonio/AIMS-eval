import argparse
import json
import time
import requests
import pandas as pd
from pathlib import Path


def solve(question: str, url: str, model: str, timeout: int = 30000) -> dict:
    resp = requests.post(url, json={"question": question, "model": model}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_stats(url: str) -> dict:
    stats_url = url.rsplit("/", 1)[0] + "/cache-stats"
    resp = requests.get(stats_url, timeout=5000)
    resp.raise_for_status()
    return resp.json()


def reset_stats(url: str):
    stats_url = url.rsplit("/", 1)[0] + "/cache-reset"
    requests.post(stats_url, timeout=5000).raise_for_status()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate solver cache: cold call vs cache hit timing")
    parser.add_argument("--model", default="gemma-4-12b-qat:latest", help="Model name")
    parser.add_argument("--data", default="data/qa_pairs.json", help="Path to QA pairs JSON")
    parser.add_argument("--url", default="http://localhost:8000/solver/solve", help="Solver API URL")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(data_path) as f:
        qa_pairs = json.load(f)


    # Reset stats before starting
    reset_stats(args.url)

    results = []

    print(f"Testing {len(qa_pairs)} questions with cache evaluation...")
    print(f"URL: {args.url}  Model: {args.model}")
    print("-" * 70)

    for i, pair in enumerate(qa_pairs):
        question = pair["text"]

        # cold call
        t0 = time.perf_counter()
        try:
            resp_data = solve(question, args.url, args.model)
            cold_answer = resp_data.get("answer", "")
        except Exception as e:
            cold_answer = f"ERROR: {e}"
        cold_ms = (time.perf_counter() - t0) * 1000

        # cache hit
        t1 = time.perf_counter()
        try:
            resp_data = solve(question, args.url, args.model)
            hit_answer = resp_data.get("answer", "")
        except Exception as e:
            hit_answer = f"ERROR: {e}"
        hit_ms = (time.perf_counter() - t1) * 1000

        speedup = cold_ms / hit_ms if hit_ms > 0 else float("inf")

        result = {
            "index": i,
            "question": question,
            "category": pair.get("category", "unknown"),
            "expected": pair.get("final_answer"),
            "cold_answer": cold_answer,
            "hit_answer": hit_answer,
            "cold_ms": round(cold_ms, 2),
            "hit_ms": round(hit_ms, 2),
            "speedup": round(speedup, 2),
        }
        results.append(result)

        print(
            f"[{i + 1}/{len(qa_pairs)}] {question[:40]:<40} "
            f"Cold: {cold_ms:>8.1f}ms  Hit: {hit_ms:>8.1f}ms  "
            f"Speedup: {speedup:.1f}x"
        )

    # Fetch server-side stats
    try:
        server_stats = get_stats(args.url)
    except Exception:
        server_stats = {"error": "Could not fetch server stats"}

    # Aggregate
    cold_times = [r["cold_ms"] for r in results if not r["cold_answer"].startswith("ERROR")]
    hit_times = [r["hit_ms"] for r in results if not r["hit_answer"].startswith("ERROR")]
    speedups = [r["speedup"] for r in results if not r["cold_answer"].startswith("ERROR") and not r["hit_answer"].startswith("ERROR")]

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    summary = {
        "model": args.model,
        "total_questions": len(qa_pairs),
        "avg_cold_ms": round(avg(cold_times), 2),
        "avg_hit_ms": round(avg(hit_times), 2),
        "avg_speedup": round(avg(speedups), 2),
        "min_cold_ms": round(min(cold_times), 2) if cold_times else 0,
        "max_cold_ms": round(max(cold_times), 2) if cold_times else 0,
        "min_hit_ms": round(min(hit_times), 2) if hit_times else 0,
        "max_hit_ms": round(max(hit_times), 2) if hit_times else 0,
        "server_stats": server_stats,
    }

    print("-" * 70)
    print(f"Avg cold call:  {summary['avg_cold_ms']:.1f}ms")
    print(f"Avg cache hit:  {summary['avg_hit_ms']:.1f}ms")
    print(f"Avg speedup:    {summary['avg_speedup']:.1f}x")
    print(f"Server stats:   {json.dumps(server_stats, indent=2)}")

    model_slug = args.model.replace("/", "_")
    output_file = output_dir / f"cache_output_{model_slug}.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)

    flat_summary = {k: v for k, v in summary.items() if k != "server_stats"}
    server_stats = summary.get("server_stats", {})
    if isinstance(server_stats, dict):
        for sk, sv in server_stats.items():
            if not isinstance(sv, (list, dict)):
                flat_summary[f"server_{sk}"] = sv

    summary_file = output_dir / f"cache_summary_{model_slug}.csv"
    pd.DataFrame([flat_summary]).to_csv(summary_file, index=False)

    print(f"\nResults saved to {output_file}")
    print(f"Summary saved to {summary_file}")


