"""
Backtest: compare VectorAI soft attribution vs naive condition-effects baseline
for condition identification on the 5 held-out patients.

Two ranking methods are evaluated head-to-head:

  Path A — VectorAI (live):
      z-score the patient's 9D feature vector, query condition_archetypes collection,
      rank by Euclidean distance.

  Path B — Condition-effects baseline (offline):
      Compute raw Euclidean distance from the patient's feature vector to each
      condition's mean vector (from condition_effects). No z-scoring, no DB.

Metrics per patient:
  - Recall@k  : fraction of true conditions found in top-k results
  - MRR       : mean reciprocal rank of first true condition hit
  - First hit : rank of the first true condition in the ranked list

Run:
    python backtest.py
    python backtest.py --no-vectorai   # skip VectorAI, baseline only
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from condition_analysis import run_pipeline, FEATURE_ORDER, CONDITION_COLS
from vectorai_client import VectorAIClient

# ── constants ─────────────────────────────────────────────────────────────────

HOLDOUT_PATIENTS = [1, 14, 16, 28, 57]
TOP_K_VALUES     = [1, 3, 5, 10]

# ── style ─────────────────────────────────────────────────────────────────────

BG      = "#0d0f18"
PANEL   = "#12172a"
BORDER  = "#1e2d4a"
TEXT    = "#e2e8f0"
SUBTEXT = "#7a8aaa"
BLUE    = "#4a9eff"
GREEN   = "#52d68a"
ORANGE  = "#f6ad55"
RED     = "#fc8181"
PURPLE  = "#b794f4"
GOLD    = "#f6e05e"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor":  PANEL,
    "axes.edgecolor":   BORDER, "axes.labelcolor": SUBTEXT,
    "text.color":       TEXT, "xtick.color":     SUBTEXT,
    "ytick.color":      SUBTEXT, "grid.color":      BORDER,
    "grid.linestyle":   "--", "grid.alpha":       0.6,
    "font.family":      "sans-serif", "font.size": 9,
})


# ── ground truth extraction ───────────────────────────────────────────────────

def get_true_conditions(patient_id: int, df_full: pd.DataFrame) -> list:
    """Return list of condition column names that are flagged for this patient."""
    row = df_full[df_full["Patient"] == patient_id]
    if row.empty:
        return []
    row = row.iloc[0]
    return [c for c in CONDITION_COLS if c in row.index and row[c] == 1]


# ── Path B: condition-effects baseline ───────────────────────────────────────

def baseline_ranking(
    patient_features: dict,
    condition_effects: dict,
    features: list = FEATURE_ORDER,
) -> list:
    """
    Rank conditions by raw Euclidean distance from patient feature vector
    to each condition's mean vector (from condition_effects).
    Returns list of dicts: {condition, distance, similarity}.
    """
    results = []
    pat_vec = np.array([patient_features.get(f, 0) or 0 for f in features], dtype=float)

    for cond, effects in condition_effects.items():
        cond_vec = np.array([effects.get(f, 0) or 0 for f in features], dtype=float)
        if not np.all(np.isfinite(pat_vec)) or not np.all(np.isfinite(cond_vec)):
            continue
        dist = float(np.linalg.norm(pat_vec - cond_vec))
        results.append({
            "condition":  cond,
            "distance":   round(dist, 4),
            "similarity": round(1.0 / (1.0 + dist), 4),
        })

    results.sort(key=lambda x: x["distance"])
    return results


# ── metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(ranked: list, true_conds: set, k: int) -> float:
    if not true_conds:
        return float("nan")
    top_k = {r["condition"] for r in ranked[:k]}
    return len(top_k & true_conds) / len(true_conds)


def first_hit_rank(ranked: list, true_conds: set) -> int:
    """1-indexed rank of first true condition; len(ranked)+1 if not found."""
    for i, r in enumerate(ranked):
        if r["condition"] in true_conds:
            return i + 1
    return len(ranked) + 1


def mrr(ranked: list, true_conds: set) -> float:
    rank = first_hit_rank(ranked, true_conds)
    return 1.0 / rank if rank <= len(ranked) else 0.0


def evaluate(ranked: list, true_conds: set, label: str) -> dict:
    return {
        "method":    label,
        "recall@1":  recall_at_k(ranked, true_conds, 1),
        "recall@3":  recall_at_k(ranked, true_conds, 3),
        "recall@5":  recall_at_k(ranked, true_conds, 5),
        "recall@10": recall_at_k(ranked, true_conds, 10),
        "mrr":       mrr(ranked, true_conds),
        "first_hit": first_hit_rank(ranked, true_conds),
        "top5_conds": [r["condition"] for r in ranked[:5]],
    }


# ── per-patient console output ────────────────────────────────────────────────

def print_patient(pid, true_conds, baseline_ranked, vectorai_ranked, category):
    print(f"\n{'=' * 72}")
    print(f"  Patient {pid}  [{category}]")
    print(f"  True conditions ({len(true_conds)}): {', '.join(sorted(true_conds))}")
    print(f"{'=' * 72}")

    for ranked, label in [(baseline_ranked, "Baseline (condition-effects)"),
                          (vectorai_ranked, "VectorAI (soft attribution)")]:
        if ranked is None:
            print(f"\n  {label}: N/A")
            continue

        fh  = first_hit_rank(ranked, true_conds)
        mrr_val = mrr(ranked, true_conds)
        print(f"\n  {label}   (MRR={mrr_val:.3f}, first hit=#{fh})")
        print(f"  {'Rank':<5} {'Condition':<32} {'Score':>7}  Match")
        print(f"  {'-'*55}")
        for i, r in enumerate(ranked[:10], 1):
            match = "<-- TRUE" if r["condition"] in true_conds else ""
            print(f"  {i:<5} {r['condition']:<32} {r['similarity']:>7.4f}  {match}")


# ── aggregate summary ─────────────────────────────────────────────────────────

def print_aggregate(all_results):
    print(f"\n{'=' * 72}")
    print("  AGGREGATE METRICS  (across 5 holdout patients)")
    print(f"{'=' * 72}")

    for method in ["Baseline", "VectorAI"]:
        rows = [r for r in all_results if r["method"].startswith(method) and r is not None]
        if not rows:
            continue

        print(f"\n  {method}:")
        for k in [1, 3, 5, 10]:
            vals = [r[f"recall@{k}"] for r in rows if not np.isnan(r[f"recall@{k}"])]
            avg  = np.mean(vals) if vals else float("nan")
            print(f"    Recall@{k:>2} : {avg:.3f}  ({[round(v,2) for v in vals]})")

        mrr_vals = [r["mrr"] for r in rows]
        fh_vals  = [r["first_hit"] for r in rows]
        print(f"    MRR      : {np.mean(mrr_vals):.3f}  ({[round(v,3) for v in mrr_vals]})")
        print(f"    First hit: avg rank {np.mean(fh_vals):.1f}")


# ── visualization ─────────────────────────────────────────────────────────────

def fig_backtest(patient_rows, all_results):
    n_patients = len(patient_rows)
    fig = plt.figure(figsize=(18, 4 + 4 * n_patients))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Backtest: VectorAI vs Condition-Effects Baseline",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.99)

    gs = GridSpec(n_patients + 1, 2, figure=fig,
                  hspace=0.65, wspace=0.35,
                  left=0.07, right=0.97, top=0.96, bottom=0.03)

    # Per-patient side-by-side top-10 bar charts
    for row_idx, (pid, true_conds, base_r, vai_r) in enumerate(patient_rows):
        ax_b = fig.add_subplot(gs[row_idx, 0])
        ax_v = fig.add_subplot(gs[row_idx, 1])

        category = [r for r in all_results if str(pid) in r["method"]]
        cat_str  = category[0]["method"].split("|")[1].strip() if category else ""

        _bar_panel(ax_b, base_r or [], true_conds,
                   f"Pat {pid} [{cat_str}] — Baseline (condition-effects)", pid)
        _bar_panel(ax_v, vai_r  or [], true_conds,
                   f"Pat {pid} [{cat_str}] — VectorAI soft attribution",  pid)

    # Bottom row: aggregate recall@k comparison
    ax_agg = fig.add_subplot(gs[n_patients, :])
    _aggregate_panel(ax_agg, all_results)

    plt.savefig("backtest_results.png", dpi=130, bbox_inches="tight", facecolor=BG)
    print("\n  Figure saved -> backtest_results.png")
    return fig


def _bar_panel(ax, ranked, true_conds, title, pid):
    if not ranked:
        ax.text(0.5, 0.5, "Not available", ha="center", va="center",
                transform=ax.transAxes, color=SUBTEXT)
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold")
        _panel_box(ax)
        return

    top = ranked[:10]
    labels = [r["condition"] for r in top]
    scores = [r["similarity"] for r in top]
    colors = [GREEN if l in true_conds else BLUE for l in labels]

    y_pos = list(range(len(labels) - 1, -1, -1))
    ax.barh(y_pos, scores, color=colors, height=0.6, alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Similarity score")
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold")
    ax.grid(True, axis="x")

    for bar, score, label in zip(ax.patches, scores, labels):
        marker = " <--" if label in true_conds else ""
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}{marker}", va="center", fontsize=7,
                color=GREEN if label in true_conds else SUBTEXT)

    _panel_box(ax)


def _aggregate_panel(ax, all_results):
    methods   = ["Baseline", "VectorAI"]
    k_vals    = [1, 3, 5, 10]
    x         = np.arange(len(k_vals))
    width     = 0.35
    colors    = [BLUE, GREEN]

    for i, method in enumerate(methods):
        rows = [r for r in all_results if r["method"].startswith(method)]
        avgs = []
        for k in k_vals:
            vals = [r[f"recall@{k}"] for r in rows if not np.isnan(r.get(f"recall@{k}", float("nan")))]
            avgs.append(np.mean(vals) if vals else 0)

        bars = ax.bar(x + i * width, avgs, width, label=method, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=8, color=TEXT)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"Recall@{k}" for k in k_vals], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Average recall across 5 holdout patients")
    ax.set_title("Aggregate Recall@k  —  Baseline vs VectorAI",
                 color=TEXT, fontsize=10, fontweight="bold")
    ax.legend(facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    ax.grid(True, axis="y")
    _panel_box(ax)


def _panel_box(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_facecolor(PANEL)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-vectorai", action="store_true")
    parser.add_argument("--no-show",     action="store_true")
    args = parser.parse_args()

    print("Loading data and computing condition effects...")
    df, reference, condition_effects, condition_counts, _ = run_pipeline("heart_features.csv")
    print(f"  {len(df)} patients, {len(condition_effects)} conditions")

    vectorai = None
    if not args.no_vectorai:
        try:
            vectorai = VectorAIClient(host="localhost:50051")
            if not vectorai.is_available():
                print("  VectorAI not reachable — running baseline only")
                vectorai = None
            else:
                print("  VectorAI connected")
        except Exception as e:
            print(f"  VectorAI unavailable: {e}")

    patient_rows = []   # (pid, true_conds, base_ranked, vai_ranked)
    all_results  = []

    for pid in HOLDOUT_PATIENTS:
        true_conds = set(get_true_conditions(pid, df))
        pat_row    = df[df["Patient"] == pid]
        if pat_row.empty:
            print(f"  Patient {pid} not found in CSV — skipping")
            continue

        pat_features = pat_row.iloc[0].to_dict()
        category     = pat_row.iloc[0].get("Category", "?")

        # Path B — baseline
        base_ranked = baseline_ranking(pat_features, condition_effects)

        # Path A — VectorAI
        vai_ranked = None
        if vectorai:
            try:
                vai_result = vectorai.nearest_condition_archetypes(
                    pat_features, k=33, min_patients=2
                )
                vai_ranked = vai_result
            except Exception as e:
                print(f"  VectorAI query failed for patient {pid}: {e}")

        print_patient(pid, true_conds, base_ranked, vai_ranked, category)

        patient_rows.append((pid, true_conds, base_ranked, vai_ranked))

        b_eval = evaluate(base_ranked, true_conds, f"Baseline|{category}|pat{pid}")
        all_results.append(b_eval)

        if vai_ranked is not None:
            v_eval = evaluate(vai_ranked, true_conds, f"VectorAI|{category}|pat{pid}")
            all_results.append(v_eval)

    print_aggregate(all_results)

    print("\nRendering figure...")
    fig_backtest(patient_rows, all_results)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
