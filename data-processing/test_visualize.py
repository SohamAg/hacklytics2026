"""
Backend output test + visualization.

Two separate figures saved to disk:
    test_vectorai.png   — anomaly score, nearest patients, soft attribution, multi-resolution
    test_morphology.png — PCA map of all patients + structural volume changes

Also prints a clean text summary and saves test_output.json.

Run from data-processing/:
    python test_visualize.py
    python test_visualize.py --conditions DORV DLoopTGA
    python test_visualize.py --no-vectorai
"""

import argparse
import json
import os
import sys

# Force UTF-8 output on Windows (cp1252 can't print box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

from backend_pipeline import HeartBackend

# ── colour constants ──────────────────────────────────────────────────────────
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
    "mild":    BLUE,
    "moderate": ORANGE,
    "severe":  RED,
    "normal":  GREEN,
    "single":  PURPLE,
    "complex": "#f687b3",
}
STRUCTURE_ORDER = ["LV", "RV", "LA", "RA", "Aorta", "PA", "SVC", "IVC"]


def cat_color(cat: str) -> str:
    c = str(cat).lower()
    for k, v in CATEGORY_PALETTE.items():
        if k in c:
            return v
    return SUBTEXT


def panel_box(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_facecolor(PANEL)


def sim_bar_color(score: float) -> str:
    """
    Score = 1/(1+euclidean_distance) in z-score space.
    1.0 = identical, ~0.5 = ~1 std away, ~0.33 = ~2 std away.
    """
    if score >= 0.70:
        return GREEN
    elif score >= 0.50:
        return BLUE
    elif score >= 0.33:
        return ORANGE
    return RED


# ── Figure 1: VectorAI results ────────────────────────────────────────────────

def fig_vectorai(result: dict):
    vs = result.get("vector_search")
    if not vs:
        print("  [VectorAI not available — skipping Figure 1]")
        return None

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(BG)
    conds = " + ".join(result["conditions"]) if result["conditions"] else "Baseline"
    fig.suptitle(f"VectorAI Search Results  —  {conds}",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.97)

    gs = GridSpec(3, 2, figure=fig,
                  hspace=0.55, wspace=0.35,
                  left=0.07, right=0.96, top=0.91, bottom=0.06)

    _plot_anomaly         (fig.add_subplot(gs[0, 0]), vs)
    _plot_nearest_patients(fig.add_subplot(gs[0, 1]), vs)
    _plot_soft_attribution(fig.add_subplot(gs[1, :]), vs)
    _plot_multi_resolution(fig.add_subplot(gs[2, :]), vs)

    plt.savefig("test_vectorai.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("  Saved → test_vectorai.png")
    return fig


def _plot_anomaly(ax, vs: dict):
    ax.axis("off")
    panel_box(ax)

    score = vs.get("anomaly_score")
    label = vs.get("anomaly_label", "unknown").upper()

    color_map = {
        "WITHIN NORMAL RANGE": GREEN,
        "MILD DEVIATION":      BLUE,
        "MODERATE ANOMALY":    ORANGE,
        "SIGNIFICANT ANOMALY": RED,
    }
    color = color_map.get(label, SUBTEXT)

    ax.text(0.5, 0.84, "ANOMALY SCORE", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=SUBTEXT, fontweight="bold")
    score_txt = f"{score:.4f}" if score is not None else "N/A"
    ax.text(0.5, 0.53, score_txt, transform=ax.transAxes,
            ha="center", va="center", fontsize=38, color=color, fontweight="bold")
    ax.text(0.5, 0.27, f"[ {label} ]", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color=color)

    desc = {
        "WITHIN NORMAL RANGE": "Closely matches a known patient",
        "MILD DEVIATION":      "Minor deviation from known anatomy",
        "MODERATE ANOMALY":    "Moderately unusual morphology",
        "SIGNIFICANT ANOMALY": "Outside normal patient range",
    }
    ax.text(0.5, 0.10, desc.get(label, ""), transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color=SUBTEXT)

    ax.set_title("Anomaly Detection", color=TEXT, fontsize=10, fontweight="bold", pad=6)
    rect = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                   boxstyle="round,pad=0.02",
                                   linewidth=1.5, edgecolor=color,
                                   facecolor=PANEL, transform=ax.transAxes)
    ax.add_patch(rect)


def _plot_nearest_patients(ax, vs: dict):
    mr     = vs.get("multi_resolution", {})
    pts    = vs.get("nearest_patients", [])
    c_pts  = mr.get("chambers", {}).get("patients", [])
    v_pts  = mr.get("vessels",  {}).get("patients", [])

    ax.axis("off")
    panel_box(ax)
    ax.set_title("Nearest Real Patients", color=TEXT, fontsize=10, fontweight="bold", pad=6)

    headers = ["Patient", "Category", "Full ♥", "Chambers", "Vessels"]
    col_x   = [0.04, 0.20, 0.55, 0.70, 0.85]
    row_h   = 0.14

    for j, h in enumerate(headers):
        ax.text(col_x[j], 0.88, h, transform=ax.transAxes,
                fontsize=8, color=SUBTEXT, fontweight="bold")
    ax.plot([0.02, 0.98], [0.84, 0.84], color=BORDER, linewidth=0.8,
            transform=ax.transAxes, zorder=0)

    full_by_id = {p["patient_id"]: p["similarity"] for p in pts}
    cham_by_id = {p["patient_id"]: p["similarity"] for p in c_pts}
    ves_by_id  = {p["patient_id"]: p["similarity"] for p in v_pts}
    all_ids    = list(dict.fromkeys(
        [p["patient_id"] for p in pts] +
        [p["patient_id"] for p in c_pts] +
        [p["patient_id"] for p in v_pts]
    ))
    pid_to_cat = {p["patient_id"]: p["category"]
                  for p in pts + c_pts + v_pts}

    for i, pid in enumerate(all_ids[:5]):
        y   = 0.75 - i * row_h
        cat = pid_to_cat.get(pid, "Unknown")
        ax.text(col_x[0], y, f"#{pid}", transform=ax.transAxes,
                fontsize=9, color=TEXT, fontweight="bold")
        ax.text(col_x[1], y, cat, transform=ax.transAxes,
                fontsize=8, color=cat_color(cat))
        for val, cx in [(full_by_id.get(pid), col_x[2]),
                        (cham_by_id.get(pid), col_x[3]),
                        (ves_by_id.get(pid),  col_x[4])]:
            if val is not None:
                ax.text(cx, y, f"{val:.3f}", transform=ax.transAxes,
                        fontsize=8.5, color=sim_bar_color(val), fontweight="bold")
            else:
                ax.text(cx, y, "—", transform=ax.transAxes,
                        fontsize=8, color=SUBTEXT)

    rect = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                   boxstyle="round,pad=0.02",
                                   linewidth=1, edgecolor=BORDER,
                                   facecolor=PANEL, transform=ax.transAxes)
    ax.add_patch(rect)


def _plot_soft_attribution(ax, vs: dict):
    archetypes = vs.get("soft_attribution", [])
    if not archetypes:
        ax.text(0.5, 0.5, "No results (all conditions have < 2 patients)",
                ha="center", va="center", transform=ax.transAxes, color=SUBTEXT)
        ax.set_title("Soft Condition Attribution", color=TEXT, fontsize=10, fontweight="bold")
        return

    labels = [f"{a['condition']}   (n={a['n_patients']})" for a in archetypes]
    scores = [a["similarity"] for a in archetypes]
    colors = [sim_bar_color(s) for s in scores]

    y_pos = list(range(len(labels) - 1, -1, -1))
    bars  = ax.barh(y_pos, scores, color=colors, height=0.55, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Similarity to condition archetype  [1/(1+Euclidean·dist in z-score space)]")
    ax.axvline(0.50, color=BORDER, linewidth=0.8, linestyle=":")
    ax.text(0.505, -0.7, "0.50", color=SUBTEXT, fontsize=7)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=9,
                color=sim_bar_color(score), fontweight="bold")

    ax.set_title("Soft Condition Attribution  —  What does this morphology most resemble?",
                 color=TEXT, fontsize=10, fontweight="bold")
    ax.grid(True, axis="x")
    panel_box(ax)


def _plot_multi_resolution(ax, vs: dict):
    mr     = vs.get("multi_resolution", {})
    c_arch = mr.get("chambers", {}).get("archetypes", [])
    v_arch = mr.get("vessels",  {}).get("archetypes", [])

    all_conds = list(dict.fromkeys(
        [a["condition"] for a in c_arch] + [a["condition"] for a in v_arch]
    ))
    if not all_conds:
        ax.text(0.5, 0.5, "No archetype data", ha="center", va="center",
                transform=ax.transAxes, color=SUBTEXT)
        ax.set_title("Multi-Resolution Attribution", color=TEXT, fontsize=10, fontweight="bold")
        return

    c_by_cond = {a["condition"]: a["similarity"] for a in c_arch}
    v_by_cond = {a["condition"]: a["similarity"] for a in v_arch}
    x     = np.arange(len(all_conds))
    width = 0.35

    ax.bar(x - width / 2, [c_by_cond.get(c, 0) for c in all_conds],
           width, label="Chambers  (LV, RV, LA, RA)",      color=ORANGE, alpha=0.85)
    ax.bar(x + width / 2, [v_by_cond.get(c, 0) for c in all_conds],
           width, label="Vessels   (Aorta, PA, SVC, IVC)", color=PURPLE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_conds, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Similarity to archetype")
    ax.set_title(
        "Multi-Resolution Attribution  —  Which conditions drive chambers vs vessels?",
        color=TEXT, fontsize=10, fontweight="bold",
    )
    ax.legend(facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    ax.grid(True, axis="y")
    panel_box(ax)

    if c_arch and v_arch:
        top_c = c_arch[0]["condition"]
        top_v = v_arch[0]["condition"]
        if top_c != top_v:
            note = f"Chambers → {top_c}   |   Vessels → {top_v}"
            ax.text(0.5, 1.01, note, transform=ax.transAxes,
                    ha="center", fontsize=8, color=SUBTEXT, style="italic")


# ── Figure 2: morphology context ──────────────────────────────────────────────

def fig_morphology(result: dict):
    fig, (ax_pca, ax_struct) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    conds = " + ".join(result["conditions"]) if result["conditions"] else "Baseline"
    fig.suptitle(f"Morphology State  —  {conds}",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.97)

    _plot_pca       (ax_pca,    result)
    _plot_structural(ax_struct, result)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("test_morphology.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("  Saved → test_morphology.png")
    return fig


def _plot_pca(ax, result: dict):
    pca_df = pd.read_csv("../models/pca_dataset.csv")
    for cat in pca_df["Category"].unique():
        mask = pca_df["Category"] == cat
        ax.scatter(pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"],
                   c=cat_color(cat), s=50, alpha=0.8, label=cat, zorder=2)

    pc1, pc2 = result["pca"]["pc1"], result["pca"]["pc2"]
    ax.scatter(pc1, pc2, c="#ffffff", s=300, marker="*", zorder=5,
               edgecolors=BLUE, linewidths=2, label="Generated")
    ax.annotate(f"  ({pc1:.2f}, {pc2:.2f})", xy=(pc1, pc2),
                color=BLUE, fontsize=8)

    sev = result["severity"]
    ax.set_title(f"Morphology Space  —  Severity: {sev:.3f}",
                 color=TEXT, fontsize=10, fontweight="bold")
    ax.set_xlabel("PC1  →  Structural Magnitude")
    ax.set_ylabel("PC2  →  Shape Variation")
    ax.legend(fontsize=7, facecolor=BG, edgecolor=BORDER,
              labelcolor=TEXT, loc="upper left", markerscale=0.8)
    ax.grid(True)
    panel_box(ax)


def _plot_structural(ax, result: dict):
    scales = result["mesh_scales"]
    pcts   = [(scales.get(s, 1.0) ** 3 - 1) * 100 for s in STRUCTURE_ORDER]
    colors = [RED if p > 5 else GREEN if p < -5 else SUBTEXT for p in pcts]

    y_pos = list(range(len(STRUCTURE_ORDER) - 1, -1, -1))
    bars  = ax.barh(y_pos, pcts, color=colors, height=0.55, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(STRUCTURE_ORDER, fontsize=10)
    ax.axvline(0, color=BLUE, linewidth=1, alpha=0.5)
    ax.set_xlabel("Volume change from baseline (%)")

    for bar, pct in zip(bars, pcts):
        sign = "+" if pct >= 0 else ""
        xpos = bar.get_width() + (0.8 if pct >= 0 else -0.8)
        ha   = "left" if pct >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{sign}{pct:.1f}%", va="center", ha=ha,
                fontsize=8.5,
                color=RED if pct > 5 else GREEN if pct < -5 else SUBTEXT,
                fontweight="bold")

    ax.set_title("Structural Changes vs Baseline",
                 color=TEXT, fontsize=10, fontweight="bold")
    ax.grid(True, axis="x")
    panel_box(ax)


# ── console summary ───────────────────────────────────────────────────────────

def print_summary(result: dict):
    vs    = result.get("vector_search")
    conds = result["conditions"] or ["Baseline"]

    print("\n" + "═" * 65)
    print(f"  CONDITIONS : {', '.join(conds)}")
    print(f"  SEVERITY   : {result['severity']:.4f}")
    print(f"  PCA        : PC1={result['pca']['pc1']:.3f}  PC2={result['pca']['pc2']:.3f}")
    print("═" * 65)

    if not vs:
        print("  VectorAI not available.\n")
        return

    score = vs.get("anomaly_score")
    label = vs.get("anomaly_label", "unknown").upper()
    print(f"\n  ANOMALY SCORE : {score:.4f}  [{label}]" if score else "\n  ANOMALY SCORE : N/A")

    print("\n  ── NEAREST REAL PATIENTS (full heart) ──────────────────")
    for p in vs.get("nearest_patients", []):
        bar = "█" * int(p["similarity"] * 30)
        print(f"  #{p['patient_id']:<4} {p['category']:<28} {bar}  sim={p['similarity']:.4f}  dist={p['distance']:.4f}")

    print("\n  ── SOFT CONDITION ATTRIBUTION (full-heart archetype) ───")
    for a in vs.get("soft_attribution", []):
        bar = "█" * int(a["similarity"] * 30)
        print(f"  {a['condition']:<28} (n={a['n_patients']:<3}) {bar}  sim={a['similarity']:.4f}  dist={a['distance']:.4f}")

    mr = vs.get("multi_resolution", {})
    print("\n  ── MULTI-RESOLUTION ARCHETYPES ─────────────────────────")
    print("  Chambers (LV, RV, LA, RA):")
    for a in mr.get("chambers", {}).get("archetypes", []):
        print(f"    {a['condition']:<28}  sim={a['similarity']:.4f}  dist={a['distance']:.4f}")
    print("  Vessels (Aorta, PA, SVC, IVC):")
    for a in mr.get("vessels", {}).get("archetypes", []):
        print(f"    {a['condition']:<28}  sim={a['similarity']:.4f}  dist={a['distance']:.4f}")

    print("═" * 65 + "\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="*", default=["VSD", "SevereDilation"])
    parser.add_argument("--no-vectorai", action="store_true")
    parser.add_argument("--no-show", action="store_true",
                        help="Save PNGs without opening interactive windows")
    args = parser.parse_args()

    vectorai_host = None if args.no_vectorai else "localhost:50051"

    print("Initialising HeartBackend...")
    backend = HeartBackend("heart_features.csv", vectorai_host=vectorai_host)
    print(f"Running simulate_heart({args.conditions})...")
    result  = backend.simulate_heart(args.conditions)

    with open("test_output.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("  Raw output saved to test_output.json")

    print_summary(result)

    print("Rendering figures...")
    fig_vectorai(result)
    fig_morphology(result)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
