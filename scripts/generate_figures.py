#!/usr/bin/env python3
"""Generate publication-ready figures for the Turkish evidentiality evaluation.

Run from the repository root with ``python scripts/generate_figures.py`` (or
run this file directly). The corrected evaluation CSV is used when available;
otherwise the exact verified summary values are used so figures can still be
reproduced without the data file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

# Verified fallback values from evaluation_results_corrected.csv /
# evaluation_summary_corrected.csv. These are percentages as proportions.
CONDITION_ACCURACY = {
    "Truth": 173 / 392,
    "False_Belief": 177 / 380,
    "Deception": 174 / 380,
}
SOURCE_ACCURACY = {
    "FB_Dir_TR": 3 / 288,
    "L_Dir_TR": 40 / 288,
    "T_Dir_TR": 196 / 288,
    "L_Ind_TR": 285 / 288,
}
CONFUSION = {"TP": 325, "FP": 377, "FN": 251, "TN": 199}
METRICS = {"Precision": 0.462963, "Recall": 0.564236,
           "F1-score": 0.508607, "Overall Accuracy": 0.454861}

# Restrained, color-blind-conscious palette suitable for print.
NAVY = "#355C7D"
TEAL = "#5B9A9A"
GOLD = "#D49A45"
RED = "#B65C5C"
INK = "#263238"
MUTED = "#64727A"
PALE = "#F2F5F7"


def _find_csv() -> Path | None:
    """Search conventional locations without depending on the current directory."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "evaluation_results_corrected.csv",
        here.parent / "evaluation_results_corrected.csv",
        Path.cwd() / "evaluation_results_corrected.csv",
        Path.cwd() / "data" / "evaluation_results_corrected.csv",
        Path("/mnt/data/evaluation_results_corrected.csv"),
    ]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and candidate.is_file():
            return candidate
        seen.add(candidate)
    return None


def _load_values() -> Tuple[dict, dict, dict, dict, str]:
    """Return condition, source, confusion, metric values and data provenance."""
    csv_path = _find_csv()
    if csv_path is None:
        return (CONDITION_ACCURACY.copy(), SOURCE_ACCURACY.copy(),
                CONFUSION.copy(), METRICS.copy(), "verified hardcoded fallback")
    try:
        df = pd.read_csv(csv_path)
        required = {"ground_truth", "prediction", "is_correct", "label_type", "source_column"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV lacks required columns: {sorted(required - set(df.columns))}")
        # Normalize common CSV boolean representations safely.
        correct = df["is_correct"].astype(str).str.strip().str.lower().map(
            {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
        )
        if correct.isna().any():
            raise ValueError("Unrecognized values in is_correct")
        df = df.copy()
        df["_correct"] = correct.astype(bool)
        conditions = {}
        for display, key in [("Truth", "Truth"), ("False_Belief", "False_Belief"), ("Deception", "Deception")]:
            subset = df[df["label_type"].astype(str).str.strip().str.casefold() == key.casefold()]
            if subset.empty:
                raise ValueError(f"No records for condition {key}")
            conditions[display] = float(subset["_correct"].mean())
        sources = {}
        for key in ("FB_Dir_TR", "L_Dir_TR", "T_Dir_TR", "L_Ind_TR"):
            subset = df[df["source_column"].astype(str).str.strip() == key]
            if subset.empty:
                raise ValueError(f"No records for source {key}")
            sources[key] = float(subset["_correct"].mean())

        # Positive class is Deceptive. Rows are actual labels; columns predictions.
        truth = df["ground_truth"].astype(str).str.strip().str.casefold()
        pred = df["prediction"].astype(str).str.strip().str.casefold()
        deceptive = {"deceptive", "false", "falsehood"}
        truthful = {"truthful", "true"}
        known = truth.isin(deceptive | truthful) & pred.isin(deceptive | truthful)
        if not known.all():
            raise ValueError("Unexpected class labels in ground_truth/prediction")
        conf = {
            "TP": int((truth.isin(deceptive) & pred.isin(deceptive)).sum()),
            "FP": int((truth.isin(truthful) & pred.isin(deceptive)).sum()),
            "FN": int((truth.isin(deceptive) & pred.isin(truthful)).sum()),
            "TN": int((truth.isin(truthful) & pred.isin(truthful)).sum()),
        }
        precision = conf["TP"] / (conf["TP"] + conf["FP"])
        recall = conf["TP"] / (conf["TP"] + conf["FN"])
        f1 = 2 * precision * recall / (precision + recall)
        metrics = {"Precision": precision, "Recall": recall, "F1-score": f1,
                   "Overall Accuracy": float(df["_correct"].mean())}
        return conditions, sources, conf, metrics, str(csv_path)
    except Exception as exc:
        print(f"WARNING: Could not use {csv_path}: {exc}; using verified fallback values.")
        return (CONDITION_ACCURACY.copy(), SOURCE_ACCURACY.copy(),
                CONFUSION.copy(), METRICS.copy(), "verified hardcoded fallback")


def _base_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.18,
                  rc={"axes.edgecolor": "#D5DDE1", "grid.color": "#E5EAED",
                      "grid.linewidth": 0.8, "axes.labelcolor": INK,
                      "xtick.color": INK, "ytick.color": INK,
                      "text.color": INK, "font.family": "DejaVu Sans"})
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "savefig.facecolor": "white",
                         "axes.titleweight": "bold", "axes.titlesize": 15,
                         "axes.labelsize": 11, "legend.frameon": False})


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bar_chart(values: dict, labels: list, title: str, ylabel: str,
               outfile: Path, color: str, decimals: int = 2,
               y_max: float = 1.0) -> None:
    vals = [values[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    bars = ax.bar(np.arange(len(labels)), vals, width=0.62, color=color,
                  edgecolor="white", linewidth=1.1, zorder=3)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, y_max)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=14)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.annotate(f"{val * 100:.{decimals}f}%",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    va="bottom", fontsize=10, fontweight="bold", color=INK)
    _save(fig, outfile)


def _figure3(conf: dict, outfile: Path) -> None:
    # Rows: actual class, columns: predicted class; positive class = Deceptive.
    raw = np.array([[conf["TP"], conf["FN"]],
                    [conf["FP"], conf["TN"]]], dtype=int)
    row_pct = raw / raw.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{n}\n{p:.2f}%" for n, p in zip(rn, rp)]
                       for rn, rp in zip(raw, row_pct)])
    fig, ax = plt.subplots(figsize=(7.3, 5.8), constrained_layout=True)
    sns.heatmap(raw, annot=labels, fmt="", cmap=sns.light_palette(NAVY, as_cmap=True),
                cbar=False, linewidths=2, linecolor="white", square=True,
                annot_kws={"fontsize": 13, "fontweight": "bold", "color": INK}, ax=ax)
    ax.set_xticklabels(["Deceptive", "Truthful"], fontsize=11)
    ax.set_yticklabels(["Deceptive", "Truthful"], fontsize=11, rotation=0)
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("Actual label", labelpad=10)
    ax.set_title("Confusion Matrix", loc="left", pad=14)
    ax.text(0.5, -0.19, "Cell labels show count and row-normalized percentage",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
    _save(fig, outfile)


def _figure4(metrics: dict, outfile: Path) -> None:
    labels = list(metrics)
    vals = [metrics[k] for k in labels]
    palette = [NAVY, TEAL, GOLD, RED]
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    bars = ax.bar(np.arange(len(labels)), vals, width=0.62, color=palette,
                  edgecolor="white", linewidth=1.1, zorder=3)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=12, ha="right")
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("Score")
    ax.set_title("Overall Evaluation Metrics", loc="left", pad=14)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:.3f}", (bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold")
    _save(fig, outfile)


def _figure5(outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.3))
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 5.3); ax.axis("off")

    def box(x, y, w, h, title, body, face, title_color=INK, body_size=10.4):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.16",
                               linewidth=1.4, edgecolor="#D0D9DE", facecolor=face)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h-0.38, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=title_color)
        ax.text(x+w/2, y+h/2-0.13, body, ha="center", va="center",
                fontsize=body_size, color=INK, linespacing=1.45)

    box(0.25, 1.33, 2.45, 2.65, "Turkish Evidentiality",
        "Direct evidence\n-Ø / direct marker\n\nIndirect / reported\n-miş / -muş", "#EEF3F6")
    box(3.45, 1.33, 2.50, 2.65, "Evaluation set",
        "288 scenarios\n× 4 source configurations\n\n1,152 prompts", "#F3F5F2")
    box(6.72, 1.33, 2.40, 2.65, "Language model",
        "Qwen 2.5\n3B-Instruct\n\nZero-shot evaluation", "#F5F1E9")
    box(9.90, 1.33, 2.05, 2.65, "Diagnostic findings",
        "Accuracy by condition\n\nSource configuration\n\nClassification metrics", "#F4EEEE", body_size=9.5)
    for x1, x2 in [(2.72, 3.37), (5.97, 6.64), (9.14, 9.82)]:
        ax.add_patch(FancyArrowPatch((x1, 2.65), (x2, 2.65), arrowstyle="-|>",
                                     mutation_scale=17, linewidth=1.7, color=MUTED))
    ax.text(6.1, 4.78, "Diagnostic evaluation framework", ha="center",
            fontsize=17, fontweight="bold", color=INK)
    ax.text(6.1, 0.62,
            "Evidence marking  →  controlled prompts  →  model predictions  →  diagnostic analysis",
            ha="center", fontsize=10, color=MUTED)
    _save(fig, outfile)


def main() -> None:
    _base_style()
    # Save in ./figures relative to the invocation directory, as requested.
    outdir = Path.cwd() / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    conditions, sources, conf, metrics, provenance = _load_values()
    print(f"Data: {provenance}")
    _bar_chart(conditions, ["Truth", "False_Belief", "Deception"],
               "Accuracy by Evaluation Condition", "Accuracy", outdir / "fig1.png", NAVY)
    _bar_chart(sources, ["FB_Dir_TR", "L_Dir_TR", "T_Dir_TR", "L_Ind_TR"],
               "Accuracy by Morphosyntactic Source Configuration", "Accuracy",
               outdir / "fig2.png", TEAL)
    _figure3(conf, outdir / "fig3.png")
    _figure4(metrics, outdir / "fig4.png")
    _figure5(outdir / "graphical_abstract.png")
    print("Generated high-resolution figures:")
    for name in ("fig1.png", "fig2.png", "fig3.png", "fig4.png", "graphical_abstract.png"):
        print(f"  {outdir / name}")


if __name__ == "__main__":
    main()
