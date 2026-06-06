#!/usr/bin/env python3
"""Compare LMTE vs path-formulation optimal CSV results and write final-result.json."""

import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_NORMALIZER = 190000.0


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_float(row, field):
    return float(row[field])


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    weight = idx - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def infer_traffic_seed_offset(lmte_rows, path_form_rows):
    """Map LMTE sample_index to path-form traffic_seed."""
    return int(path_form_rows[0]["traffic_seed"]) - int(lmte_rows[0]["sample_index"])


def compare(lmte_rows, path_form_rows, normalizer, traffic_seed_offset):
    if len(lmte_rows) != len(path_form_rows):
        raise ValueError(
            f"Row count mismatch: {len(lmte_rows)} LMTE rows vs "
            f"{len(path_form_rows)} path-form rows"
        )

    per_row = []
    signed_gaps = []
    abs_gaps = []
    relative_gaps = []
    path_form_runtimes = []

    for lmte, path_form in zip(lmte_rows, path_form_rows):
        sample_index = int(lmte["sample_index"])
        traffic_seed = int(path_form["traffic_seed"])
        expected_seed = sample_index + traffic_seed_offset
        if traffic_seed != expected_seed:
            raise ValueError(
                f"Seed mismatch at sample_index={sample_index}: "
                f"expected traffic_seed={expected_seed}, got {traffic_seed}"
            )

        lmte_obj = parse_float(lmte, "total_satisfied_demand")
        path_form_obj = parse_float(path_form, "obj_val")
        total_demand = parse_float(lmte, "total_demand")
        path_form_runtime = parse_float(path_form, "runtime")

        signed_gap = (path_form_obj - lmte_obj) / normalizer
        abs_gap = abs(path_form_obj - lmte_obj) / normalizer
        rel_gap = (path_form_obj - lmte_obj) / path_form_obj if path_form_obj else None

        per_row.append(
            {
                "sample_index": sample_index,
                "traffic_seed": traffic_seed,
                "problem": path_form.get("problem"),
                "scale_factor": float(path_form.get("scale_factor", 1.0)),
                "total_demand": total_demand,
                "lmte_total_flow": lmte_obj,
                "path_form_obj_val": path_form_obj,
                "obj_val_delta": path_form_obj - lmte_obj,
                "routed_fraction": parse_float(lmte, "routed_fraction"),
                "normalized_gap": signed_gap,
                "normalized_abs_gap": abs_gap,
                "relative_gap": rel_gap,
                "path_form_runtime": path_form_runtime,
            }
        )

        signed_gaps.append(signed_gap)
        abs_gaps.append(abs_gap)
        if rel_gap is not None:
            relative_gaps.append(rel_gap)
        path_form_runtimes.append(path_form_runtime)

    summary = {
        "num_rows": len(per_row),
        "normalizer": normalizer,
        "traffic_seed_offset": traffic_seed_offset,
        "normalized_gap_path_form_minus_lmte": {
            "max": max(signed_gaps),
            "min": min(signed_gaps),
            "mean": statistics.mean(signed_gaps),
            "median": statistics.median(signed_gaps),
            "stdev": statistics.pstdev(signed_gaps) if len(signed_gaps) > 1 else 0.0,
            "p95": percentile(signed_gaps, 95),
            "p99": percentile(signed_gaps, 99),
        },
        "normalized_abs_gap": {
            "max": max(abs_gaps),
            "min": min(abs_gaps),
            "mean": statistics.mean(abs_gaps),
            "median": statistics.median(abs_gaps),
            "stdev": statistics.pstdev(abs_gaps) if len(abs_gaps) > 1 else 0.0,
            "p95": percentile(abs_gaps, 95),
            "p99": percentile(abs_gaps, 99),
        },
        "relative_gap_path_form_minus_lmte": {
            "max": max(relative_gaps) if relative_gaps else None,
            "min": min(relative_gaps) if relative_gaps else None,
            "mean": statistics.mean(relative_gaps) if relative_gaps else None,
            "median": statistics.median(relative_gaps) if relative_gaps else None,
        },
        "obj_val_delta_path_form_minus_lmte": {
            "max": max(r["obj_val_delta"] for r in per_row),
            "min": min(r["obj_val_delta"] for r in per_row),
            "mean": statistics.mean(r["obj_val_delta"] for r in per_row),
            "median": statistics.median(r["obj_val_delta"] for r in per_row),
        },
        "runtime": {
            "path_form_mean": statistics.mean(path_form_runtimes),
            "path_form_median": statistics.median(path_form_runtimes),
        },
        "rows_with_lmte_below_path_form": sum(
            1 for r in per_row if r["obj_val_delta"] > 0
        ),
        "rows_with_lmte_equal_path_form": sum(
            1 for r in per_row if r["obj_val_delta"] == 0
        ),
        "rows_with_lmte_above_path_form": sum(
            1 for r in per_row if r["obj_val_delta"] < 0
        ),
    }

    worst = max(per_row, key=lambda r: r["normalized_gap"])
    best = min(per_row, key=lambda r: r["normalized_gap"])
    summary["worst_case_row"] = {
        "traffic_seed": worst["traffic_seed"],
        "normalized_gap": worst["normalized_gap"],
        "obj_val_delta": worst["obj_val_delta"],
    }
    summary["best_case_row"] = {
        "traffic_seed": best["traffic_seed"],
        "normalized_gap": best["normalized_gap"],
        "obj_val_delta": best["obj_val_delta"],
    }

    return {"summary": summary, "rows": per_row}


def write_per_row_csv(rows, path):
    fieldnames = [
        "sample_index",
        "traffic_seed",
        "total_demand",
        "lmte_total_flow",
        "path_form_obj_val",
        "obj_val_delta",
        "routed_fraction",
        "normalized_gap",
        "normalized_abs_gap",
        "relative_gap",
        "path_form_runtime",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Compare LMTE and path-form CSV benchmark results."
    )
    parser.add_argument(
        "--lmte-csv",
        default="results/B4_lmte/demand_routing_metrics_1780706383.csv",
        help="Path to LMTE demand_routing_metrics CSV",
    )
    parser.add_argument(
        "--path-form-csv",
        default="results/B4_lmte/path-form-total_flow-all.csv",
        help="Path to path-formulation optimal results CSV",
    )
    parser.add_argument(
        "--output",
        default="results/B4_lmte/final-result.json",
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--per-row-csv",
        default="results/B4_lmte/lmte-vs-pathform.csv",
        help="Optional per-row comparison CSV output path",
    )
    parser.add_argument(
        "--normalizer",
        type=float,
        default=DEFAULT_NORMALIZER,
        help="Gap normalization denominator (default: total B4 link capacity 190000)",
    )
    parser.add_argument(
        "--traffic-seed-offset",
        type=int,
        default=None,
        help="traffic_seed = sample_index + offset (auto-detected if omitted)",
    )
    args = parser.parse_args()

    lmte_rows = load_csv(args.lmte_csv)
    path_form_rows = load_csv(args.path_form_csv)
    offset = (
        args.traffic_seed_offset
        if args.traffic_seed_offset is not None
        else infer_traffic_seed_offset(lmte_rows, path_form_rows)
    )
    result = compare(lmte_rows, path_form_rows, args.normalizer, offset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    if args.per_row_csv:
        per_row_path = Path(args.per_row_csv)
        per_row_path.parent.mkdir(parents=True, exist_ok=True)
        write_per_row_csv(result["rows"], per_row_path)

    summary = result["summary"]
    gap = summary["normalized_gap_path_form_minus_lmte"]
    abs_gap = summary["normalized_abs_gap"]
    print(f"Wrote {output_path}")
    if args.per_row_csv:
        print(f"Wrote {args.per_row_csv}")
    print(f"Rows compared: {summary['num_rows']}")
    print(f"Traffic seed offset: {summary['traffic_seed_offset']}")
    print(
        "Normalized gap (path-form - LMTE) / "
        f"{args.normalizer}: max={gap['max']:.8f}, mean={gap['mean']:.8f}, "
        f"median={gap['median']:.8f}"
    )
    print(
        "Normalized abs gap: "
        f"max={abs_gap['max']:.8f}, mean={abs_gap['mean']:.8f}, "
        f"median={abs_gap['median']:.8f}"
    )
    print(
        f"LMTE below optimal: {summary['rows_with_lmte_below_path_form']} | "
        f"equal: {summary['rows_with_lmte_equal_path_form']} | "
        f"above: {summary['rows_with_lmte_above_path_form']}"
    )
    print(
        f"Worst seed={summary['worst_case_row']['traffic_seed']} "
        f"gap={summary['worst_case_row']['normalized_gap']:.8f} "
        f"delta={summary['worst_case_row']['obj_val_delta']:.6f}"
    )


if __name__ == "__main__":
    main()
