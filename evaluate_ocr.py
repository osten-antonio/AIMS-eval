import json
import base64
import os
import requests
import time
import jiwer

ENDPOINTS = [
    ("http://localhost:9000/nougat-base/inference", "nougat-base"),
    ("http://localhost:9000/nougat-latex-base/inference", "nougat-latex-base"),
    ("http://localhost:9000/pix2tex/inference", "pix2tex"),
]

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def query_endpoint(url, image_b64, max_retries=2):
    payload = {"image": image_b64}
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=3000)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("output", "")
        except Exception as e:
            if attempt == max_retries:
                return None
            time.sleep(2)
    return None

def normalize_latex(s):
    """Whitespace-insensitive normalization for exact match."""
    if not s:
        return ""
    return " ".join(s.split())

def levenshtein(s1, s2):
    """Compute Levenshtein (edit) distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def compute_metrics(reference, hypothesis):
    """Compute three metrics: exact match, normalized edit distance, raw edit distance."""
    ref_norm = normalize_latex(reference) if reference else ""
    hyp_norm = normalize_latex(hypothesis) if hypothesis else ""

    exact = ref_norm == hyp_norm

    # Levenshtein distance
    lev_dist = levenshtein(ref_norm, hyp_norm)
    ref_len = len(ref_norm) if ref_norm else 1
    normalized_edit = lev_dist / ref_len

    return {
        "exact_match": exact,
        "levenshtein": lev_dist,
        "normalized_edit_distance": round(normalized_edit, 4),
        "reference": ref_norm,
        "hypothesis": hyp_norm,
    }

if __name__ == "__main__":
    with open("data/manifest.json") as f:
        manifest = json.load(f)

    print(f"Total items: {len(manifest)}")

    results = []

    for i, item in enumerate(manifest):
        img_path = item["image_path"]
        if not os.path.exists(img_path):
            print(f"  Skip {i}: {img_path} not found")
            continue

        img_b64 = encode_image(img_path)
        ref_latex = item.get("latex", "")

        item_results = {"id": item["id"], "image_path": img_path}
        item_results.update(item)

        for url, model_name in ENDPOINTS:
            print(f"  [{i}/{len(manifest)}] {model_name} - {item.get('render_style', '?')} - {item.get('category', '?')}...")
            output = query_endpoint(url, img_b64)

            if output is None:
                print(f"    FAILED (timeout/error)")
                item_results[f"{model_name}_output"] = None
                item_results[f"{model_name}_error"] = "timeout or request failure"
                metrics = {"exact_match": False, "levenshtein": -1, "normalized_edit_distance": 1.0}
            else:
                metrics = compute_metrics(ref_latex, output)
                print(f"    exact={metrics['exact_match']} lev={metrics['levenshtein']}")

            item_results[f"{model_name}_output"] = output
            item_results[f"{model_name}_exact_match"] = metrics["exact_match"]
            item_results[f"{model_name}_levenshtein"] = metrics["levenshtein"]
            item_results[f"{model_name}_normalized_edit_distance"] = metrics["normalized_edit_distance"]

        results.append(item_results)

    # Save results
    with open("data/inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to data/inference_results.json ({len(results)} items)")

    # Print summary
    for url, model_name in ENDPOINTS:
        total = len(results)
        exact = sum(1 for r in results if r.get(f"{model_name}_exact_match"))
        avg_lev = sum(r.get(f"{model_name}_levenshtein", 0) for r in results) / max(total, 1)
        avg_ned = sum(r.get(f"{model_name}_normalized_edit_distance", 0) for r in results) / max(total, 1)

        # Per category
        eq_total = sum(1 for r in results if r.get("category") == "equation" and r.get(f"{model_name}_exact_match"))
        eq_total_count = sum(1 for r in results if r.get("category") == "equation")
        wp_total = sum(1 for r in results if r.get("category") == "word_problem" and r.get(f"{model_name}_exact_match"))
        wp_total_count = sum(1 for r in results if r.get("category") == "word_problem")

        # Per render style
        printed_total = sum(1 for r in results if r.get("render_style") == "printed" and r.get(f"{model_name}_exact_match"))
        printed_total_count = sum(1 for r in results if r.get("render_style") == "printed")
        hw_total = sum(1 for r in results if r.get("render_style") == "handwritten" and r.get(f"{model_name}_exact_match"))
        hw_total_count = sum(1 for r in results if r.get("render_style") == "handwritten")

        print(f"\n=== {model_name} ===")
        print(f"  Total: {exact}/{total} exact matches ({exact/total*100:.1f}%)")
        print(f"  Avg Levenshtein: {avg_lev:.2f}")
        print(f"  Avg Norm Edit Distance: {avg_ned:.4f}")
        print(f"  Equations: {eq_total}/{eq_total_count} ({eq_total/eq_total_count*100:.1f}%)")
        print(f"  Word problems: {wp_total}/{wp_total_count} ({wp_total/wp_total_count*100:.1f}%)")
        print(f"  Printed: {printed_total}/{printed_total_count} ({printed_total/printed_total_count*100:.1f}%)")
        print(f"  Handwritten: {hw_total}/{hw_total_count} ({hw_total/hw_total_count*100:.1f}%)")
