import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text_of(row):
    return str(row.get("input", row.get("prompt", "")))


def target_of(row):
    for key in ("target", "output", "response"):
        if key in row:
            value = row[key]
            if isinstance(value, str):
                return value
            return json.dumps(value, ensure_ascii=False)
    return ""


from audit_training_pipeline_v06 import ngrams, jaccard


def numeric_id(row):
    rid = str(row.get("id", ""))
    m = re.search(r"(\d+)$", rid)
    return int(m.group(1)) if m else -1


def generation_bucket(n):
    if n <= 550:
        return "v0.1-v0.2"
    if n <= 670:
        return "v0.3"
    if n <= 820:
        return "v0.4"
    if n <= 1060:
        return "v0.5"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("reports/v06_dataset_cleaning_report.json"),
    )
    ap.add_argument("--near-threshold", type=float, default=0.82)
    args = ap.parse_args()

    rows = load_jsonl(args.dataset)

    suspicious_terms = {
        "masculine_persona": [
            "kar raha hoon",
            "karunga",
            "samajh gaya",
            "dekh raha hoon",
        ],
        "pseudo_marker": [
            "<|keren_",
            "<|keran_",
        ],
    }

    flags = defaultdict(list)

    normalized = []
    for row in rows:
        rid = str(row.get("id", ""))
        prompt = text_of(row)
        target = target_of(row)
        combined = f"{prompt}\n{target}".lower()

        for category, terms in suspicious_terms.items():
            for term in terms:
                if term in combined:
                    flags[rid].append(f"{category}:{term}")

        normalized.append((rid, ngrams(prompt)))

    near_pairs = []
    for i in range(len(normalized)):
        id_a, a = normalized[i]

        for j in range(i + 1, len(normalized)):
            id_b, b = normalized[j]

            score = jaccard(a, b)
            if score >= args.near_threshold:
                near_pairs.append(
                    {
                        "a": id_a,
                        "b": id_b,
                        "score": round(score, 4),
                    }
                )
                flags[id_a].append(f"near_duplicate:{id_b}:{score:.3f}")
                flags[id_b].append(f"near_duplicate:{id_a}:{score:.3f}")

    generation_counts = Counter()
    flagged_generation_counts = Counter()

    for row in rows:
        rid = str(row.get("id", ""))
        bucket = generation_bucket(numeric_id(row))
        generation_counts[bucket] += 1
        if flags[rid]:
            flagged_generation_counts[bucket] += 1

    report = {
        "dataset": str(args.dataset),
        "records": len(rows),
        "near_duplicate_threshold": args.near_threshold,
        "near_duplicate_pairs": len(near_pairs),
        "generation_counts": dict(generation_counts),
        "flagged_generation_counts": dict(flagged_generation_counts),
        "flagged_records": sum(1 for v in flags.values() if v),
        "flags": dict(flags),
        "near_pairs": sorted(
            near_pairs,
            key=lambda x: x["score"],
            reverse=True,
        ),
        "policy": {
            "delete_automatically": False,
            "holdout_must_not_be_added": True,
            "manual_review_required": True,
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Records: {len(rows)}")
    print(f"Near duplicate pairs: {len(near_pairs)}")
    print(f"Flagged records: {sum(1 for v in flags.values() if v)}")
    print("Generation distribution:", dict(generation_counts))
    print("Flagged by generation:", dict(flagged_generation_counts))
    print(f"Report: {args.report}")
    print("NO RECORDS DELETED")


if __name__ == "__main__":
    main()
