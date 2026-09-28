import json
import base64
import os
import argparse
import requests
import time
import jiwer
import pandas as pd
import cv2

ENDPOINTS = [
    ("http://localhost:9000/nougat-base/inference", "nougat-base"),
    ("http://localhost:9000/nougat-latex-base/inference", "nougat-latex-base"),
    ("http://localhost:9000/pix2tex/inference", "pix2tex"),
]

FASTMATH_CSV_DEFAULT = "data/FASTMATH/latex_equation_groundtruth.csv"
FASTMATH_IMAGE_ROOT_DEFAULT = "data/FASTMATH/paper"
FASTMATH_OUTPUT_DEFAULT = "results/fastmath_inference_results.csv"
FASTMATH_SUBSET_MANIFEST_DEFAULT = "data/fastmath_subset_manifest.csv"

# Canonical levels expected in the FASTMATH csv `level` column.
FASTMATH_LEVELS = ["Level1", "Level2", "Level3"]


def encode_image(path, fm=False):
    if fm:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        cropped = img[265:-1, 70:-800]
        success, buf = cv2.imencode(".png", cropped)
        if not success:
            raise ValueError(f"Could not encode cropped image: {path}")
        return base64.b64encode(buf.tobytes()).decode()
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


def compute_metrics(reference, hypothesis):
    """Compute three metrics: exact match, normalized edit distance, raw edit distance."""
    ref_norm = normalize_latex(reference) if reference else ""
    hyp_norm = normalize_latex(hypothesis) if hypothesis else ""

    exact = ref_norm == hyp_norm

    # Levenshtein distance
    output = jiwer.process_characters(reference, hypothesis)
    lev_dist = output.substitutions + output.deletions + output.insertions
    ref_len = len(ref_norm) if ref_norm else 1
    denom = max(len(ref_norm), len(hyp_norm), 1)
    normalized_edit = lev_dist / denom

    return {
        "exact_match": exact,
        "levenshtein": lev_dist,
        "normalized_edit_distance": round(normalized_edit, 4),
        "reference": ref_norm,
        "hypothesis": hyp_norm,
    }


def build_fastmath_subset(csv_path, image_root, per_level=1000,
                           drop_empty_equation=True):
    """Load FASTMATH csv and return a balanced subset: per_level rows per level."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    unknown = df[df["level"].isna()]
    if len(unknown) > 0:
        print(f"  Warning: {len(unknown)} rows with unrecognized level values "
              f"({sorted(df.loc[df['level'].isna(), 'level'].unique().tolist())}) — excluded.")

    df = df[df["level"].notna()].copy()

    if drop_empty_equation:
        empty_mask = df["equation"].str.strip() == ""
        n_empty = int(empty_mask.sum())
        if n_empty:
            print(f"  Dropping {n_empty} rows with empty equation.")
        df = df[~empty_mask].copy()

    # Resolve image paths against image_root and keep only rows whose file exists.
    df["_abs_image_path"] = df["image_path"].apply(
        lambda p: os.path.join(image_root, str(p).strip()))
    missing = df[~df["_abs_image_path"].map(os.path.exists)]
    if len(missing) > 0:
        print(f"  Warning: {len(missing)} rows point at missing image files — excluded.")

    # Report pool sizes before sampling so shortfalls are visible.
    for lvl in FASTMATH_LEVELS:
        n = int((df["level"] == lvl).sum())
        print(f"  Pool {lvl}: {n} rows (after filtering)")

    parts = []
    for lvl in FASTMATH_LEVELS:
        pool = df[df["level"] == lvl]
        # Keep only rows with existing files for the actual sample.
        pool = pool[pool["_abs_image_path"].map(os.path.exists)]
        if len(pool) < per_level:
            raise ValueError(
                f"Not enough {lvl} rows: have {len(pool)}, need {per_level}. "
                f"Lower --per-level or check filtering.")
        parts.append(pool.sample(n=per_level, random_state=5))

    subset = pd.concat(parts, ignore_index=True)
    # Shuffle the combined 3k so evaluation order isn't grouped by level.
    subset = subset.sample(frac=1.0, random_state=5).reset_index(drop=True)

    items = []
    for idx, row in subset.iterrows():
        items.append({
            "id": f"{row.get('arena', 'arena')}-{row['level']}-{idx}",
            "image_path": row["_abs_image_path"],
            "latex": row["equation"],
            "level": row["level"],
            "arena": row.get("arena", ""),
            "csv_row": row.to_dict(),
        })

    counts = subset["level"].value_counts().to_dict()
    print(f"  Subset balance: { {k: int(counts.get(k, 0)) for k in FASTMATH_LEVELS} } "
          f"(total {len(subset)} rows)")

    return items, subset


def print_fastmath_summary(results):
    for url, model_name in ENDPOINTS:
        total = len(results)
        exact = sum(1 for r in results if r.get(f"{model_name}_exact_match"))
        avg_lev = sum(r.get(f"{model_name}_levenshtein", 0) for r in results) / max(total, 1)
        avg_ned = sum(r.get(f"{model_name}_normalized_edit_distance", 0) for r in results) / max(total, 1)

        print(f"\n=== {model_name} ===")
        print(f"  Total: {exact}/{total} exact matches ({exact/total*100:.1f}%)" if total else "  Total: 0 items")
        print(f"  Avg Levenshtein: {avg_lev:.2f}")
        print(f"  Avg Norm Edit Distance: {avg_ned:.4f}")

        for lvl in FASTMATH_LEVELS:
            lvl_rows = [r for r in results if r.get("level") == lvl]
            n = len(lvl_rows)
            if n == 0:
                print(f"  {lvl}: 0 items")
                continue
            e = sum(1 for r in lvl_rows if r.get(f"{model_name}_exact_match"))
            al = sum(r.get(f"{model_name}_levenshtein", 0) for r in lvl_rows) / n
            an = sum(r.get(f"{model_name}_normalized_edit_distance", 0) for r in lvl_rows) / n
            print(f"  {lvl}: {e}/{n} exact ({e/n*100:.1f}%) | avg lev {al:.2f} | avg NED {an:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate OCR models on a balanced FASTMATH subset "
                    "(1k Level1 + 1k Level2 + 1k Level3 by default).")
    parser.add_argument("--csv", default=FASTMATH_CSV_DEFAULT,
                        help="Path to latex_equation_groundtruth.csv")
    parser.add_argument("--image-root", default=FASTMATH_IMAGE_ROOT_DEFAULT,
                        help="Base dir prepended to csv image_path "
                             "(csv paths are relative to data/FASTMATH/paper/)")
    parser.add_argument("--per-level", type=int, default=1000,
                        help="Rows sampled per level (default 1000)")
    parser.add_argument("--output", default=FASTMATH_OUTPUT_DEFAULT,
                        help="Where to write inference results CSV")
    parser.add_argument("--subset-manifest", default=FASTMATH_SUBSET_MANIFEST_DEFAULT,
                        help="Where to write the sampled 3k subset manifest CSV "
                             "(empty string to skip)")
    parser.add_argument("--legacy-manifest", default="",
                        help="Optional legacy path (e.g. data/manifest.json) to evaluate "
                             "instead of the FASTMATH csv. Empty = use FASTMATH csv.")
    args = parser.parse_args()

    if args.legacy_manifest:
        with open(args.legacy_manifest) as f:
            manifest = json.load(f)

        print(f"Total items: {len(manifest)} (legacy manifest: {args.legacy_manifest})")

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
        pd.DataFrame(results).to_csv("data/inference_results.csv", index=False)

        print(f"\nResults saved to data/inference_results.csv ({len(results)} items)")

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
    else:
        print(f"Loading FASTMATH csv: {args.csv}")
        items, subset_df = build_fastmath_subset(
            args.csv, args.image_root, per_level=args.per_level)

        if args.subset_manifest:
            subset_df.to_csv(args.subset_manifest, index=False)
            print(f"Subset manifest saved to {args.subset_manifest} ({len(subset_df)} rows)")

        print(f"Total items: {len(items)} "
              f"({args.per_level} x {len(FASTMATH_LEVELS)} levels)")

        results = []

        for i, item in enumerate(items):
            img_path = item["image_path"]
            if not os.path.exists(img_path):
                print(f"  Skip {i}: {img_path} not found")
                continue

            img_b64 = encode_image(img_path, fm=True)
            ref_latex = item.get("latex", "")

            item_results = {"id": item["id"], "image_path": img_path,
                            "level": item.get("level", ""),
                            "arena": item.get("arena", "")}
            item_results.update(item.get("csv_row", {}))
            # Keep resolved absolute image path (csv image_path is relative).
            item_results["image_path"] = img_path

            for url, model_name in ENDPOINTS:
                print(f"  [{i}/{len(items)}] {model_name} - level={item.get('level', '?')}...")
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
        pd.DataFrame(results).to_csv(args.output, index=False)

        print(f"\nResults saved to {args.output} ({len(results)} items)")

        print_fastmath_summary(results)
