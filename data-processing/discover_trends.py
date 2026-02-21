#!/usr/bin/env python3
"""
Discover and report how congenital conditions relate to heart volumes/ratios.

Usage:
  python discover_trends.py                    # print trends and export condition_effects.json
  python discover_trends.py --estimate VSD ASD  # show estimated features for VSD+ASD
  python discover_trends.py --export trends.json # also save trend report to JSON
"""

import argparse
import json
import os
import sys

from condition_analysis import (
    CONDITION_COLS,
    FEATURE_COLS,
    run_pipeline,
    discover_trends,
    estimate_features,
    scaling_factors_for_viewer,
    export_for_viewer,
)


def main():
    parser = argparse.ArgumentParser(description="Discover condition–feature trends from heart_features.csv")
    parser.add_argument("--csv", default=None, help="Path to heart_features.csv")
    parser.add_argument("--estimate", nargs="*", default=None, help="Conditions to estimate (e.g. --estimate VSD ASD)")
    parser.add_argument("--export", default=None, help="Export trend report to this JSON path")
    parser.add_argument("--min-count", type=int, default=3, help="Min patients per condition to report")
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    csv_path = args.csv or os.path.join(base, "heart_features.csv")
    if not os.path.isfile(csv_path):
        print(f"Not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df, reference, condition_effects, condition_counts, condition_multipliers = run_pipeline(csv_path)
    n_normal = (df["Normal"] == 1).sum() if "Normal" in df.columns else 0
    print(f"Loaded {len(df)} patients, {n_normal} normal (reference cohort)\n")

    # Trend discovery
    trends = discover_trends(reference, condition_effects, condition_counts, min_count=args.min_count)
    print("=== Condition vs normal: notable feature changes (>5%) ===\n")
    for t in trends:
        print(f"  {t['condition']} (n={t['n']})")
        for d in t["deltas"]:
            print(f"    {d['feature']}: {d['ref']:.1f} -> {d['with_condition']:.1f}  ({d['pct_change']:+.1f}%)")
        print()

    if args.export:
        with open(args.export, "w") as f:
            json.dump(trends, f, indent=2)
        print(f"Trend report saved to {args.export}\n")

    # Export JSON for 3D viewer
    out_json = os.path.join(base, "..", "models", "condition_effects.json")
    export_for_viewer(reference, condition_multipliers, condition_counts, out_json)
    print(f"Viewer payload: {out_json}\n")

    # Optional: estimate for given conditions
    if args.estimate:
        conds = [c.strip() for c in args.estimate if c.strip() and c.strip() in condition_multipliers]
        if not conds:
            print("No valid conditions in:", args.estimate)
            print("Available:", [c for c in CONDITION_COLS if c in condition_multipliers])
        else:
            est = estimate_features(conds, reference, condition_multipliers)
            scales = scaling_factors_for_viewer(est, reference)
            print(f"=== Estimated heart for: {conds} ===")
            for col in ["Label_1_vol_ml", "Label_2_vol_ml", "Label_3_vol_ml", "Label_4_vol_ml", "Total_heart_vol", "LV_RV_ratio"]:
                if col in est:
                    print(f"  {col}: {est[col]:.1f}  (normal: {reference.get(col, '—')})")
            print("  Scaling factors for 3D (per chamber):", scales)


if __name__ == "__main__":
    main()
