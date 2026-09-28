import json
import os
import random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SRC = os.path.join(SCRIPT_DIR, "Full_ASyMOB_Dataset.json")
DST = os.path.join(SCRIPT_DIR, "ASyMOB_200_unique.json")
N = 200
SEED = 5
PER_SOURCE = 2 

def main():
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total: {len(data)}")

    # Exclude Original 
    non_orig = [x for x in data if (x.get("Variation") or "").strip().lower() != "original"]
    print(f"Non-original: {len(non_orig)}")

    # Deduplicate by Challenge text (keep first occurrence)
    seen = set()
    uniq = []
    dups = 0
    for x in non_orig:
        c = x.get("Challenge", "")
        if c in seen:
            dups += 1
            continue
        seen.add(c)
        uniq.append(x)
    print(f"Unique challenges: {len(uniq)} (dropped {dups} exact dupes)")

    # Group by Source (= derivation family)
    by_source = defaultdict(list)
    for x in uniq:
        by_source[x.get("Source")].append(x)
    print(f"Unique sources (derivations): {len(by_source)}")
    if len(by_source) * PER_SOURCE < N:
        raise SystemExit("Not enough sources to reach N with PER_SOURCE cap")

    # Sample PER_SOURCE per Source, reproducibly
    rng = random.Random(SEED)
    sampled = []
    sources = sorted(by_source.keys())
    rng.shuffle(sources)
    for s in sources:
        pool = by_source[s]
        rng.shuffle(pool)
        sampled.extend(pool[:PER_SOURCE])
    assert len(sampled) == N, len(sampled)

    # Shuffle final order so sources are interleaved
    rng.shuffle(sampled)

    # Verify
    assert len({x["Challenge"] for x in sampled}) == N, "Challenges not unique!"
    assert len({x["Source"] for x in sampled}) == len(by_source), "Should cover all 100 derivations"
    from collections import Counter
    print("Per-source counts:", Counter(x["Source"] for x in sampled).most_common(3))
    print("Variation mix:", Counter(x["Variation"] for x in sampled).most_common())
    print("Category mix:", Counter(x["Category"] for x in sampled).most_common())

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(sampled)} items -> {DST}")

if __name__ == "__main__":
    main()
