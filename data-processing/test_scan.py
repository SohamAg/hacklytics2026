"""
Demo: upload a held-out NIfTI segmentation and run full analysis.

Held-out patients (never seen by the model):
    Pat 1  — Normal / mild
    Pat 14 — VSD + ASD / moderate
    Pat 16 — VSD + DORV + Dextrocardia + SingleVentricle / severe
    Pat 28 — VSD + ASD + ArterialSwitch + InvertedVentricles / severe
    Pat 57 — TGA + BilateralSVC + Dextrocardia + Fontan / severe

Usage:
    python test_scan.py                          # defaults to Pat 1
    python test_scan.py --patient 16
    python test_scan.py --nii path/to/file.nii.gz
    python test_scan.py --patient 57 --no-show
"""

import argparse
import json
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

from backend_pipeline import HeartBackend, extract_features_from_nii

# Ground truth for held-out patients (from hvsmr_clinical.csv)
HOLDOUT_INFO = {
    1:  {"label": "Normal", "category": "mild",     "conditions": ["Normal"]},
    14: {"label": "VSD + ASD", "category": "moderate", "conditions": ["VSD", "ASD"]},
    16: {"label": "VSD + DORV + Dextrocardia + SingleVentricle", "category": "severe",
         "conditions": ["VSD", "DORV", "Dextrocardia", "Mesocardia", "InvertedVentricles",
                        "InvertedAtria", "SingleVentricle", "DILV", "Glenn"]},
    28: {"label": "VSD + ASD + ArterialSwitch + InvertedVentricles", "category": "severe",
         "conditions": ["VSD", "ASD", "ArterialSwitch", "InvertedVentricles",
                        "SuperoinferiorVentricles"]},
    57: {"label": "TGA + BilateralSVC + Dextrocardia + Fontan", "category": "severe",
         "conditions": ["ASD", "DLoopTGA", "BilateralSVC", "Dextrocardia",
                        "InvertedVentricles", "InvertedAtria", "LLoopTGA",
                        "SingleVentricle", "CommonAtrium", "Fontan"]},
}

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
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  SUBTEXT,
    "text.color":       TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "grid.color":       BORDER,
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "sans-serif",
    "font.size":        9,
})

CATEGORY_PALETTE = {
    "mild": BLUE, "moderate": ORANGE, "severe": RED,
    "normal": GREEN, "single": PURPLE, "complex": "#f687b3",
}
STRUCTURE_ORDER = ["LV", "RV", "LA", "RA", "Aorta", "PA", "SVC", "IVC"]


def cat_color(cat):
    c = str(cat).lower()
    for k, v in CATEGORY_PALETTE.items():
        if k in c:
            return v
    return SUBTEXT


def panel_box(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_facecolor(PANEL)


def sim_bar_color(score):
    if score >= 0.70:  return GREEN
    elif score >= 0.50: return BLUE
    elif score >= 0.33: return ORANGE
    return RED


# ── Figure 1: scan overview ───────────────────────────────────────────────────

def fig_scan_overview(result, patient_id, ground_truth):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)

    gt_label = ground_truth.get("label", "Unknown") if ground_truth else "Unknown"
    cat      = ground_truth.get("category", "?") if ground_truth else "?"
    fig.suptitle(
        f"Real Scan Analysis  —  Patient {patient_id}  |  Ground Truth: {gt_label}  [{cat}]",
        color=GOLD, fontsize=13, fontweight="bold", y=0.98,
    )

    gs = GridSpec(3, 3, figure=fig,
                  hspace=0.55, wspace=0.40,
                  left=0.06, right=0.97, top=0.93, bottom=0.05)

    _plot_anomaly        (fig.add_subplot(gs[0, 0]), result, ground_truth)
    _plot_features       (fig.add_subplot(gs[0, 1]), result)
    _plot_nearest        (fig.add_subplot(gs[0, 2]), result)
    _plot_pca            (fig.add_subplot(gs[1, :2]), result)
    _plot_structural     (fig.add_subplot(gs[1, 2]), result)
    _plot_soft_attr      (fig.add_subplot(gs[2, :2]), result)
    _plot_multi_res      (fig.add_subplot(gs[2, 2]), result)

    outfile = f"scan_pat{patient_id}_analysis.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved analysis figure -> {outfile}")
    return fig


def _plot_anomaly(ax, result, ground_truth):
    ax.axis("off")
    panel_box(ax)

    vs    = result.get("vector_search") or {}
    score = vs.get("anomaly_score")
    label = vs.get("anomaly_label", "unknown").upper()

    color_map = {
        "WITHIN NORMAL RANGE": GREEN, "MILD DEVIATION": BLUE,
        "MODERATE ANOMALY": ORANGE,   "SIGNIFICANT ANOMALY": RED,
    }
    color = color_map.get(label, SUBTEXT)

    ax.text(0.5, 0.88, "ANOMALY SCORE", transform=ax.transAxes,
            ha="center", fontsize=8, color=SUBTEXT, fontweight="bold")
    ax.text(0.5, 0.60, f"{score:.4f}" if score else "N/A",
            transform=ax.transAxes, ha="center", fontsize=34, color=color, fontweight="bold")
    ax.text(0.5, 0.38, f"[ {label} ]", transform=ax.transAxes,
            ha="center", fontsize=9, color=color)

    sev = result.get("severity", 0)
    ax.text(0.5, 0.20, f"Severity: {sev:.3f}", transform=ax.transAxes,
            ha="center", fontsize=9, color=SUBTEXT)

    if ground_truth:
        gt_cat = ground_truth.get("category", "?")
        ax.text(0.5, 0.07, f"GT: {gt_cat}", transform=ax.transAxes,
                ha="center", fontsize=8, color=GOLD)

    ax.set_title("Anomaly Detection", color=TEXT, fontsize=10, fontweight="bold", pad=6)
    rect = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                   boxstyle="round,pad=0.02",
                                   linewidth=1.5, edgecolor=color,
                                   facecolor=PANEL, transform=ax.transAxes)
    ax.add_patch(rect)


def _plot_features(ax, result):
    """Show extracted volumetric features as a horizontal bar chart."""
    feats  = result.get("extracted_features", {})
    names  = [f"Label_{i}" for i in range(1, 9)] + ["Total"]
    keys   = [f"Label_{i}_vol_ml" for i in range(1, 9)] + ["Total_heart_vol"]
    labels = ["LV", "RV", "LA", "RA", "Aorta", "PA", "SVC", "IVC", "TOTAL"]
    values = [feats.get(k, 0) for k in keys]
    colors = [BLUE, BLUE, ORANGE, ORANGE, RED, RED, PURPLE, PURPLE, GREEN]

    y_pos = list(range(len(labels) - 1, -1, -1))
    ax.barh(y_pos, values, color=colors, height=0.6, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Volume (mL)")
    ax.set_title("Extracted Volumes from Scan", color=TEXT, fontsize=10, fontweight="bold")
    ax.grid(True, axis="x")
    panel_box(ax)

    for y, val in zip(y_pos, values):
        ax.text(val + 0.5, y, f"{val:.1f}", va="center", fontsize=7.5, color=SUBTEXT)


def _plot_nearest(ax, result):
    vs  = result.get("vector_search") or {}
    pts = vs.get("nearest_patients", [])

    ax.axis("off")
    panel_box(ax)
    ax.set_title("Nearest Real Patients (Full Heart)", color=TEXT, fontsize=10, fontweight="bold", pad=6)

    headers = ["Patient", "Category", "Similarity"]
    col_x   = [0.05, 0.30, 0.68]
    ax.plot([0.02, 0.98], [0.84, 0.84], color=BORDER, linewidth=0.8,
            transform=ax.transAxes)
    for j, h in enumerate(headers):
        ax.text(col_x[j], 0.89, h, transform=ax.transAxes,
                fontsize=8, color=SUBTEXT, fontweight="bold")

    for i, p in enumerate(pts[:6]):
        y   = 0.75 - i * 0.12
        sim = p["similarity"]
        ax.text(col_x[0], y, f"#{p['patient_id']}", transform=ax.transAxes,
                fontsize=9, color=TEXT, fontweight="bold")
        ax.text(col_x[1], y, p["category"], transform=ax.transAxes,
                fontsize=8, color=cat_color(p["category"]))
        ax.text(col_x[2], y, f"{sim:.4f}", transform=ax.transAxes,
                fontsize=9, color=sim_bar_color(sim), fontweight="bold")

    rect = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                   boxstyle="round,pad=0.02",
                                   linewidth=1, edgecolor=BORDER,
                                   facecolor=PANEL, transform=ax.transAxes)
    ax.add_patch(rect)


def _plot_pca(ax, result):
    pca_df = pd.read_csv("../models/pca_dataset.csv")
    for cat in pca_df["Category"].unique():
        mask = pca_df["Category"] == cat
        ax.scatter(pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"],
                   c=cat_color(cat), s=45, alpha=0.75, label=cat, zorder=2)

    pc1, pc2 = result["pca"]["pc1"], result["pca"]["pc2"]
    ax.scatter(pc1, pc2, c=GOLD, s=350, marker="*", zorder=5,
               edgecolors="#ffffff", linewidths=1.5, label="This scan")
    ax.annotate(f"  ({pc1:.2f}, {pc2:.2f})", xy=(pc1, pc2),
                color=GOLD, fontsize=8.5, fontweight="bold")

    ax.set_title(f"PCA Morphology Landscape  (9D -> 2D)  |  Severity: {result['severity']:.3f}",
                 color=TEXT, fontsize=10, fontweight="bold")
    ax.set_xlabel("PC1 — Overall Size")
    ax.set_ylabel("PC2 — Shape Variation")
    ax.legend(fontsize=7, facecolor=BG, edgecolor=BORDER,
              labelcolor=TEXT, loc="upper left", markerscale=0.8)
    ax.grid(True)
    panel_box(ax)


def _plot_structural(ax, result):
    scales = result["mesh_scales"]
    pcts   = [(scales.get(s, 1.0) ** 3 - 1) * 100 for s in STRUCTURE_ORDER]
    colors = [RED if p > 5 else GREEN if p < -5 else SUBTEXT for p in pcts]

    y_pos = list(range(len(STRUCTURE_ORDER) - 1, -1, -1))
    ax.barh(y_pos, pcts, color=colors, height=0.55, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(STRUCTURE_ORDER, fontsize=9)
    ax.axvline(0, color=BLUE, linewidth=1, alpha=0.5)
    ax.set_xlabel("Vol change vs reference (%)")
    ax.set_title("Structural Deviations", color=TEXT, fontsize=10, fontweight="bold")
    ax.grid(True, axis="x")
    panel_box(ax)

    for bar, pct in zip(ax.patches, pcts):
        sign = "+" if pct >= 0 else ""
        xpos = bar.get_width() + (0.5 if pct >= 0 else -0.5)
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{sign}{pct:.0f}%", va="center",
                ha="left" if pct >= 0 else "right",
                fontsize=7.5, color=RED if pct > 5 else GREEN if pct < -5 else SUBTEXT)


def _plot_soft_attr(ax, result):
    vs         = result.get("vector_search") or {}
    archetypes = vs.get("soft_attribution", [])

    if not archetypes:
        ax.text(0.5, 0.5, "VectorAI not available", ha="center",
                va="center", transform=ax.transAxes, color=SUBTEXT)
        ax.set_title("Condition Attribution", color=TEXT, fontsize=10, fontweight="bold")
        return

    labels = [f"{a['condition']}  (n={a['n_patients']})" for a in archetypes]
    scores = [a["similarity"] for a in archetypes]
    colors = [sim_bar_color(s) for s in scores]
    y_pos  = list(range(len(labels) - 1, -1, -1))

    bars = ax.barh(y_pos, scores, color=colors, height=0.55, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Similarity to condition archetype")
    ax.axvline(0.50, color=BORDER, linewidth=0.8, linestyle=":")

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=9,
                color=sim_bar_color(score), fontweight="bold")

    ax.set_title(
        "Soft Condition Attribution  —  What condition does this scan most resemble?",
        color=TEXT, fontsize=10, fontweight="bold",
    )
    ax.grid(True, axis="x")
    panel_box(ax)


def _plot_multi_res(ax, result):
    vs     = result.get("vector_search") or {}
    mr     = vs.get("multi_resolution", {})
    c_arch = mr.get("chambers", {}).get("archetypes", [])
    v_arch = mr.get("vessels",  {}).get("archetypes", [])

    if not c_arch and not v_arch:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color=SUBTEXT)
        ax.set_title("Multi-Resolution", color=TEXT, fontsize=10, fontweight="bold")
        return

    top_c = c_arch[:3] if c_arch else []
    top_v = v_arch[:3] if v_arch else []

    conds  = list(dict.fromkeys([a["condition"] for a in top_c + top_v]))
    c_map  = {a["condition"]: a["similarity"] for a in top_c}
    v_map  = {a["condition"]: a["similarity"] for a in top_v}
    x      = np.arange(len(conds))
    w      = 0.35

    ax.bar(x - w/2, [c_map.get(c, 0) for c in conds],
           w, label="Chambers", color=ORANGE, alpha=0.85)
    ax.bar(x + w/2, [v_map.get(c, 0) for c in conds],
           w, label="Vessels",  color=PURPLE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Similarity")
    ax.set_title("Chamber vs Vessel\nAttribution", color=TEXT, fontsize=10, fontweight="bold")
    ax.legend(facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=7)
    ax.grid(True, axis="y")
    panel_box(ax)


# ── console summary ───────────────────────────────────────────────────────────

def print_scan_summary(result, patient_id, ground_truth):
    vs = result.get("vector_search") or {}

    print("\n" + "=" * 70)
    print(f"  PATIENT     : {patient_id}")
    if ground_truth:
        print(f"  GROUND TRUTH: {ground_truth['label']}  [{ground_truth['category']}]")
    print(f"  SEVERITY    : {result['severity']:.4f}")
    print(f"  PCA         : PC1={result['pca']['pc1']:.3f}  PC2={result['pca']['pc2']:.3f}")
    if result.get("obj_path"):
        print(f"  MESH OBJ    : {result['obj_path']}")
    print("=" * 70)

    score = vs.get("anomaly_score")
    label = vs.get("anomaly_label", "?").upper()
    print(f"\n  ANOMALY SCORE : {score:.4f}  [{label}]" if score else "\n  VectorAI not available")

    print("\n  -- INFERRED CONDITIONS (soft attribution) --")
    for a in vs.get("soft_attribution", [])[:5]:
        bar = "#" * int(a["similarity"] * 30)
        print(f"  {a['condition']:<30} (n={a['n_patients']:<3}) {bar}  {a['similarity']:.4f}")

    print("\n  -- NEAREST REAL PATIENTS --")
    for p in vs.get("nearest_patients", [])[:5]:
        bar = "#" * int(p["similarity"] * 30)
        print(f"  #{p['patient_id']:<4} {p['category']:<20} {bar}  {p['similarity']:.4f}")

    if ground_truth:
        print(f"\n  >> Top inferred condition: {vs.get('soft_attribution', [{}])[0].get('condition', '?')}")
        print(f"  >> Actual conditions:      {', '.join(ground_truth['conditions'][:3])}")

    print("=" * 70 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=int, default=1,
                        help="Held-out patient ID (1, 14, 16, 28, or 57)")
    parser.add_argument("--nii", type=str, default=None,
                        help="Path to a .nii.gz segmentation file (overrides --patient)")
    parser.add_argument("--no-vectorai", action="store_true")
    parser.add_argument("--no-show",     action="store_true")
    args = parser.parse_args()

    vectorai_host = None if args.no_vectorai else "localhost:50051"

    if args.nii:
        nii_path     = args.nii
        patient_id   = "custom"
        ground_truth = None
    else:
        patient_id   = args.patient
        nii_path     = f"cropped/cropped/pat{patient_id}_cropped_seg.nii.gz"
        ground_truth = HOLDOUT_INFO.get(patient_id)

    if not os.path.isfile(nii_path):
        print(f"Error: file not found: {nii_path}")
        sys.exit(1)

    print(f"Initialising HeartBackend...")
    backend = HeartBackend("heart_features.csv", vectorai_host=vectorai_host)

    print(f"Analyzing scan: {nii_path} ...")
    result = backend.analyze_scan(nii_path, output_dir="scan_outputs")

    out_json = f"scan_pat{patient_id}_output.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Raw output saved to {out_json}")

    print_scan_summary(result, patient_id, ground_truth)

    print("Rendering figure...")
    fig_scan_overview(result, patient_id, ground_truth)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
