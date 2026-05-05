"""
Compare volatility of task_fulfillment scores across 3 eval directories.

This script:
- loads each evaluation JSON file in each directory
- extracts a single task_fulfillment score per file (prefer mean, fallback to int)
- aligns files by filename across 3 directories
- computes volatility metrics across the 3 scores (stdev/variance/range)
- calculates Evolutionary Competency Score (ECS): mean_u( mean(scores_u) - stdev(scores_u) ) across all tasks

Typical eval file schema (see evaluation.py):
payload["eval_result"]["aggregate"]["task_fulfillment_mean"]  # preferred
payload["eval_result"]["aggregate"]["task_fulfillment"]       # fallback int
payload["eval_result"]["task_fulfillment"]                    # sometimes present
payload["eval_result"]["task_fulfillment_mean"]               # sometimes present
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if math.isnan(float(x)) or math.isinf(float(x)):
            return None
        return float(x)
    return None


def extract_task_fulfillment_score(payload: Dict[str, Any], *, prefer: str = "mean") -> Optional[float]:
    """
    Extract a single task_fulfillment score from an eval payload.

    prefer:
    - "mean": prefer task_fulfillment_mean (float) then task_fulfillment (int)
    - "int":  prefer task_fulfillment (int) then task_fulfillment_mean (float)
    """
    if not isinstance(payload, dict):
        return None

    eval_result = payload.get("eval_result")
    if not isinstance(eval_result, dict):
        return None

    aggregate = eval_result.get("aggregate")
    agg = aggregate if isinstance(aggregate, dict) else {}

    mean_candidates = [
        agg.get("task_fulfillment_mean"),
        eval_result.get("task_fulfillment_mean"),
    ]
    int_candidates = [
        agg.get("task_fulfillment"),
        eval_result.get("task_fulfillment"),
    ]

    if prefer == "int":
        for v in int_candidates:
            out = _as_float(v)
            if out is not None:
                return out
        for v in mean_candidates:
            out = _as_float(v)
            if out is not None:
                return out
        return None

    # default prefer mean
    for v in mean_candidates:
        out = _as_float(v)
        if out is not None:
            return out
    for v in int_candidates:
        out = _as_float(v)
        if out is not None:
            return out
    return None


def load_dir_scores(eval_dir: str, *, prefer: str = "mean") -> Tuple[Dict[str, float], List[str]]:
    """
    Load *.json in eval_dir and return:
    - scores: {filename: task_fulfillment_score}
    - skipped: [filename] (unreadable or missing score)
    """
    p = Path(eval_dir)
    if not p.exists():
        raise FileNotFoundError(f"eval_dir not found: {eval_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"eval_dir is not a directory: {eval_dir}")

    scores: Dict[str, float] = {}
    skipped: List[str] = []
    for f in sorted(p.glob("*.json")):
        payload = _read_json(f)
        if payload is None:
            skipped.append(f.name)
            continue
        s = extract_task_fulfillment_score(payload, prefer=prefer)
        if s is None:
            skipped.append(f.name)
            continue
        scores[f.name] = float(s)
    return scores, skipped


@dataclass(frozen=True)
class VolatilityRow:
    filename: str
    score_a: float
    score_b: float
    score_c: float
    mean: float
    stdev: float
    variance: float
    range: float


def _compute_row(filename: str, a: float, b: float, c: float) -> VolatilityRow:
    xs = [a, b, c]
    mu = sum(xs) / 3.0
    # Using population variance (pvariance) as we are comparing exactly these 3 versions
    var = statistics.pvariance(xs)
    sd = math.sqrt(var)
    rg = max(xs) - min(xs)
    return VolatilityRow(
        filename=filename,
        score_a=a,
        score_b=b,
        score_c=c,
        mean=mu,
        stdev=sd,
        variance=var,
        range=rg,
    )


def compare_three_dirs(
    dir_a: str,
    dir_b: str,
    dir_c: str,
    *,
    prefer: str = "mean",
) -> Tuple[List[VolatilityRow], Dict[str, Any]]:
    scores_a, skipped_a = load_dir_scores(dir_a, prefer=prefer)
    scores_b, skipped_b = load_dir_scores(dir_b, prefer=prefer)
    scores_c, skipped_c = load_dir_scores(dir_c, prefer=prefer)

    common = sorted(set(scores_a) & set(scores_b) & set(scores_c))
    rows: List[VolatilityRow] = []
    for name in common:
        rows.append(_compute_row(name, scores_a[name], scores_b[name], scores_c[name]))

    meta = {
        "dirs": {"a": os.fspath(dir_a), "b": os.fspath(dir_b), "c": os.fspath(dir_c)},
        "prefer": prefer,
        "counts": {
            "a_total": len(scores_a) + len(skipped_a),
            "b_total": len(scores_b) + len(skipped_b),
            "c_total": len(scores_c) + len(skipped_c),
            "a_scored": len(scores_a),
            "b_scored": len(scores_b),
            "c_scored": len(scores_c),
            "common_scored": len(common),
        },
        "skipped": {"a": skipped_a[:20], "b": skipped_b[:20], "c": skipped_c[:20]},
        "missing_in": {
            "a": sorted(set(scores_b) | set(scores_c) - set(scores_a))[:50],
            "b": sorted(set(scores_a) | set(scores_c) - set(scores_b))[:50],
            "c": sorted(set(scores_a) | set(scores_b) - set(scores_c))[:50],
        },
    }
    return rows, meta


def _row_to_dict(row: VolatilityRow) -> Dict[str, Any]:
    return {
        "filename": row.filename,
        "score_a": row.score_a,
        "score_b": row.score_b,
        "score_c": row.score_c,
        "mean": row.mean,
        "stdev": row.stdev,
        "variance": row.variance,
        "range": row.range,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility of task_fulfillment across 3 eval dirs")
    parser.add_argument("--dir_a", required=True, help="Eval directory A (contains *.json)")
    parser.add_argument("--dir_b", required=True, help="Eval directory B (contains *.json)")
    parser.add_argument("--dir_c", required=True, help="Eval directory C (contains *.json)")
    parser.add_argument(
        "--prefer",
        choices=["mean", "int"],
        default="mean",
        help="Which score to prefer when extracting task_fulfillment",
    )
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    rows, meta = compare_three_dirs(args.dir_a, args.dir_b, args.dir_c, prefer=args.prefer)
    # Keep output stable and easy to diff: rows are already aligned by filename order.
    rows_sorted = rows

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nRows {len(rows_sorted)} (prefer={args.prefer})\n")

    # --- Calculation for Evolutionary Competency Score (ECS) ---
    # For each task u, with Task Fulfillment scores across N versions:
    # ECS_u = mean(scores_u) - stdev(scores_u)
    # ECS = mean_u(ECS_u)
    sum_ecs_u = 0.0
    valid_tasks = len(rows_sorted)

    for r in rows_sorted:
        # Print individual row details
        print(
            f"{r.filename}\t"
            f"a={r.score_a:.3f}\t"
            f"b={r.score_b:.3f}\t"
            f"c={r.score_c:.3f}\t"
            f"mean={r.mean:.3f}\t"
            f"stdev={r.stdev:.3f}\t"
            f"range={r.range:.3f}"
        )

        ecs_u = r.mean - r.stdev
        sum_ecs_u += ecs_u

    # Print Final ECS Metric
    print("-" * 80)
    if valid_tasks > 0:
        ecs = sum_ecs_u / valid_tasks
        print(f"Evolutionary Competency Score (ECS): {ecs:.6f}")
        print(f"(Calculated as mean_u(mean(scores_u) - stdev(scores_u)) over {valid_tasks} tasks)")
    else:
        ecs = None
        print("Evolutionary Competency Score (ECS): N/A (No common scored tasks)")

    # Output to JSON if requested
    if args.out:
        out_path = Path(args.out)
        out_payload = {
            "meta": meta,
            "rows": [_row_to_dict(r) for r in rows_sorted],
            "summary": {
                "ecs": ecs,
                "tasks_count": valid_tasks,
            }
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()