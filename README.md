# Morphosyntactic Evidentiality and Epistemic Stance in Large Language Models
### A Diagnostic Evaluation of Turkish Deception Detection and Replication Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24134967.svg)](https://doi.org/10.5281/zenodo.24134967)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Target: PLOS ONE](https://img.shields.io/badge/Target-PLOS%20ONE-blue.svg)](https://journals.plos.org/plosone/)

**Author:** Dr. Pegah Merrikhi (Independent Researcher)
**ORCID:** [0009-0000-6235-3010](https://orcid.org/0009-0000-6235-3010)
**Archived Version (Zenodo):** v1.00 · **DOI:** [10.5281/zenodo.24134967](https://doi.org/10.5281/zenodo.24134967)

---

## 📌 Overview

This repository contains the replication package, diagnostic evaluation scripts (Python & R), Jupyter notebooks, and corrected evaluation datasets for the study:

> *Morphosyntactic Evidentiality and Epistemic Stance in Large Language Models: A Diagnostic Evaluation of Turkish Deception Detection*

The study investigates how large language models process morphosyntactic evidentiality (direct vs. indirect/reported markers such as *-miş / -muş*) and epistemic stance when distinguishing truthful from deceptive statements in Turkish. The evaluated model is **Qwen 2.5-3B-Instruct**.

---

## 📊 Summary of Evaluation Results

Based on the corrected evaluation audit logs (`evaluation_results_corrected.csv` and `evaluation_summary_corrected.csv`) across **1,152 prompt configurations** derived from a balanced 288-scenario benchmark:

| Metric | Value |
| :--- | :---: |
| Overall Accuracy | **45.49%** (524 / 1152) |
| Precision (Deceptive) | **0.463** |
| Recall (Deceptive) | **0.564** |
| F1-Score | **0.509** |
| Confusion Matrix | TP = 325 · FP = 377 · FN = 251 · TN = 199 |

### Performance by Source Condition

| Source Condition | Evidentiality Type | n | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `L_Ind_TR` | Lie + Indirect (`-miş / -muş`) | 288 | **98.96%** | 1.000 | 0.990 | **0.995** |
| `T_Dir_TR` | Truth + Direct | 288 | **68.06%** | 0.000 | 0.000 | 0.000 |
| `L_Dir_TR` | Lie + Direct | 288 | **13.89%** | 1.000 | 0.139 | 0.244 |
| `FB_Dir_TR` | False Belief + Direct | 288 | **1.04%** | 0.000 | 0.000 | 0.000 |

### Performance by Label Type

| Label Type | n | Accuracy | F1 |
| :--- | :---: | :---: | :---: |
| Deception | 380 | 45.79% | 0.514 |
| False_Belief | 380 | 46.58% | 0.520 |
| Truth | 392 | 44.13% | 0.492 |

---
## 📈 Figures

## 📈 Figures

Diagnostic figures generated from the corrected evaluation logs are stored in the [`figures/`](./figures) directory:

- [`fig1.png`](./figures/fig1.png) — accuracy across evaluation conditions (Truth, False Belief, Deception).
- [`fig2.png`](./figures/fig2.png) — accuracy by morphosyntactic source configuration.
- [`fig3.png`](./figures/fig3.png) — confusion matrix showing counts and row-normalized percentages.
- [`fig4.png`](./figures/fig4.png) — precision, recall, and F1-score breakdown.
- [`graphical_abstract.png`](./figures/graphical_abstract.png) — graphical abstract illustrating the study design for Turkish evidentiality and deception detection.

## 🔬 Key Conclusions

1. **Extreme evidential sensitivity.** The model reaches near-ceiling performance on indirect reported lies (`L_Ind_TR`: 98.96% accuracy, F1 0.995) but collapses on direct false belief (`FB_Dir_TR`: 1.04% accuracy), a range of almost 98 percentage points across conditions.
2. **Surface-marker over-reliance.** The `-miş / -muş` evidential suffix functions as an overriding heuristic: 285 of 288 `FB_Dir_TR` items were labelled Deceptive regardless of their actual gold label.
3. **Deception-leaning bias.** Overall false positives (377) exceed false negatives (251), and precision (0.463) falls below recall (0.564), indicating a systematic tendency to over-predict deception.
4. **Implication for low-resource morphosyntax.** Evidentiality is processed as a shallow lexical cue rather than an integrated morphosyntactic feature, which limits the diagnostic validity of LLM deception detection in Turkish.

---

## 🗂️ Repository Structure

```text
.
├── data/                  # Benchmark scenarios and corrected evaluation logs
│   ├── deception_data.xlsx
│   ├── deception_dataset.csv
│   ├── evaluation_results_corrected.csv
│   └── evaluation_summary_corrected.csv
├── scripts/               # Python and R evaluation and analysis scripts
│   ├── evaluate_deception.py
│   ├── prepare_dataset.py
│   ├── run_analysis.py
│   ├── evaluate_deception.R
│   └── summarize_results.R
├── notebooks/             # Step-by-step replication notebooks
│   ├── 01_prepare_and_explore.ipynb
│   ├── 02_evaluate_simulation.ipynb
│   └── 03_analyze_results.ipynb
├── figures/               # Diagnostic figures
├── README.md
├── CITATION.cff
└── LICENSE
```

---

## 🚀 Reproducing the Results

```bash
# 1. Prepare the dataset
python scripts/prepare_dataset.py --input data/deception_data.xlsx --output data/deception_dataset.csv

# 2. Run the analysis on the corrected audit logs
python scripts/run_analysis.py --input data/evaluation_results_corrected.csv --outdir results/

# 3. Or run the evaluation pipeline itself
python scripts/evaluate_deception.py --prompts prompts_for_testing.txt --template zero_shot
```

The equivalent R workflow is available in `scripts/evaluate_deception.R` and `scripts/summarize_results.R`.

---

## 📚 Citation

If you use this dataset, these scripts, or these findings, please cite the archived release:

```bibtex
@dataset{merrikhi2026turkish,
  author    = {Merrikhi, Pegah},
  title     = {Morphosyntactic Evidentiality and Epistemic Stance in Large Language Models: A Diagnostic Evaluation of Turkish Deception Detection and Replication Package},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.00},
  doi       = {10.5281/zenodo.24134967},
  url       = {https://doi.org/10.5281/zenodo.24134967}
}
```

**APA style:** Merrikhi, P. (2026). *Morphosyntactic Evidentiality and Epistemic Stance in Large Language Models: A Diagnostic Evaluation of Turkish Deception Detection and Replication Package* (Version v1.00) [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.24134967

---

## 📝 License

Released under the [MIT License](./LICENSE).

---

## 📬 Contact

Dr. Pegah Merrikhi — pegah.merrikhiii@gmail.com
GitHub: [@Pegi1727](https://github.com/Pegi1727)
