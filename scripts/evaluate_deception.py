#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_deception.py
=====================
Parse 'prompts_for_testing.txt' (1152 structured prompts), build LLM classification
prompts from templates, query an LLM (real API if requested/available, otherwise a
deterministic in-sandbox simulator), and evaluate:

    * Overall Accuracy, Precision, Recall, F1  (positive class = "Deceptive")
    * Broken down by Label_Type   (Truth / False_Belief / Deception)
    * Broken down by Source_Column (T_Dir_TR / FB_Dir_TR / L_Dir_TR / L_Ind_TR)

Outputs (all in the same directory as this script, /mnt/data):
    evaluation_results.csv     -- one row per prompt: fields + raw + predicted label
    evaluation_summary.csv     -- per-condition and overall metric table
    evaluation_summary.json    -- same metrics + run metadata as JSON

Only the Python standard library is used (no third-party dependencies).

Run:
    python3 evaluate_deception.py            # simulated run (offline)
    python3 evaluate_deception.py --real     # real LLM calls (needs API key + internet)
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from collections import OrderedDict

# --------------------------------------------------------------------------
# 1. Constants and paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(HERE, "prompts_for_testing.txt")

RESULTS_CSV = os.path.join(HERE, "evaluation_results.csv")
SUMMARY_CSV = os.path.join(HERE, "evaluation_summary.csv")
SUMMARY_JSON = os.path.join(HERE, "evaluation_summary.json")

REQUIRED_FIELDS = ["Prompt ID", "Subject_EN", "Label_Type", "Source_Column",
                   "Sentence_TR", "Mapped_Label"]
CONDITIONS = ["Label_Type", "Source_Column"]
EXPECTED_N = 1152
LABELS = ("Truthful", "Deceptive")
POSITIVE_LABEL = "Deceptive"          # class of interest for P/R/F1

# --------------------------------------------------------------------------
# 2. Prompt templates for the LLM classifier
# --------------------------------------------------------------------------
ZERO_SHOT_TEMPLATE = """You are an expert in linguistics and deception analysis.

Read the following Turkish sentence, which was produced in a context where the \
speaker's actual belief and the ground-truth location of the subject are known.

Context (English description of the true situation): {subject_en}
Condition: {label_type}
Turkish sentence to classify: "{sentence_tr}"

Decide whether the sentence is TRUTHFUL or DECEPTIVE in this context.
- Truthful: the sentence asserts what the speaker believes / what is the case.
- Deceptive: the sentence asserts the opposite of what the speaker believes.

Answer with exactly one word, either "Truthful" or "Deceptive".
Answer:"""

FEW_SHOT_TEMPLATE = """You are an expert in linguistics and deception analysis.

You classify Turkish sentences as Truthful or Deceptive, relative to a given context.

Examples:
Context: Ali is at home. | Sentence: "Ali evde."        -> Truthful
Context: Ali is at home. | Sentence: "Ali evdeymi\u015f."   -> Truthful  (the evidential -mi\u015f marks reported knowledge, consistent with the context)
Context: Ali is at home. | Sentence: "Ali evde de\u011fil."   -> Deceptive
Context: Ali is at home. | Sentence: "Ali evde de\u011filmi\u015f." -> Deceptive

Now classify this item:
Context (English): {subject_en}
Condition: {label_type}
Turkish sentence: "{sentence_tr}"

Answer with exactly one word, either "Truthful" or "Deceptive".
Answer:"""

SYSTEM_PROMPT = ("You are a careful linguistic annotator. Follow the instruction and "
                 "answer with exactly one word: Truthful or Deceptive.")

TEMPLATES = {"zero_shot": ZERO_SHOT_TEMPLATE, "few_shot": FEW_SHOT_TEMPLATE}

# --------------------------------------------------------------------------
# 3. Parsing
# --------------------------------------------------------------------------
def parse_prompts(path=PROMPTS_FILE):
    """Return a list of OrderedDicts, one per prompt record."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    parts = re.split(r"(?m)^Prompt\s+(\d+)\s*$", text)
    records = []
    for i in range(1, len(parts) - 1, 2):
        rec = OrderedDict()
        rec["Prompt ID"] = int(parts[i])
        for line in parts[i + 1].split("\n"):
            line = line.strip()
            if not line or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            rec[key.strip()] = value.strip()
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            raise ValueError("Prompt %s missing fields %s" % (rec.get("Prompt ID"), missing))
        rec["Mapped_Label"] = rec["Mapped_Label"].capitalize()
        records.append(rec)
    return records


def validate(records):
    if len(records) != EXPECTED_N:
        sys.stderr.write("[warn] expected %d records, parsed %d\n" % (EXPECTED_N, len(records)))
    ids = [r["Prompt ID"] for r in records]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("Prompt IDs are not unique/monotonic")
    for rec in records:
        if rec["Mapped_Label"] not in LABELS:
            raise ValueError("Unknown label %r in prompt %s" % (rec["Mapped_Label"], rec["Prompt ID"]))
    return True

# --------------------------------------------------------------------------
# 4. LLM querying (real API or deterministic simulator)
# --------------------------------------------------------------------------
def build_prompt(rec, template_name="zero_shot"):
    return TEMPLATES[template_name].format(
        subject_en=rec["Subject_EN"],
        label_type=rec["Label_Type"],
        sentence_tr=rec["Sentence_TR"],
    )


def _post_json(url, payload, api_key, timeout=60):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_llm_real(prompt, system_prompt=SYSTEM_PROMPT, model=None,
                   api_key=None, base_url=None, max_retries=3):
    """Call an OpenAI-compatible chat-completions endpoint. Requires internet."""
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get(
        "LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
    payload = {"model": model,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": 5}
    last = None
    for attempt in range(max_retries):
        try:
            data = _post_json(base_url, payload, api_key)
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:                                   # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError("LLM query failed after %d retries: %s" % (max_retries, last))


def simulate_llm(sentence_tr, mapped_label):
    """
    Deterministic stand-in for a real LLM call (offline mode).

    Mirrors a plausible-but-imperfect classifier: predicts the gold label most
    of the time with ~10% deterministic flips, so all metric components
    (Accuracy / Precision / Recall / F1) are non-trivial and reproducible.
    Replace this with a real API call (use_real_llm=True) to evaluate a model.
    """
    h = int(hashlib.sha256((sentence_tr + "|" + mapped_label).encode()).hexdigest(), 16)
    pred = mapped_label
    if h % 10 == 0:                 # controlled, deterministic error rate
        pred = "Deceptive" if mapped_label == "Truthful" else "Truthful"
    return pred


def classify_records(records, use_real_llm=False, template_name="zero_shot",
                     model=None, api_key=None):
    """Return list of (record, raw_llm_answer, predicted_label)."""
    out = []
    for n, rec in enumerate(records, 1):
        prompt = build_prompt(rec, template_name)
        if use_real_llm:
            raw = query_llm_real(prompt, model=model, api_key=api_key)
        else:
            raw = simulate_llm(rec["Sentence_TR"], rec["Mapped_Label"])
        pred = normalize_label(raw)
        out.append((rec, raw, pred))
        if n % 200 == 0 or n == len(records):
            print("  classified %d/%d" % (n, len(records)), flush=True)
    return out


def normalize_label(raw):
    """Map any raw LLM answer onto Truthful/Deceptive (or 'Unknown')."""
    if raw is None:
        return "Unknown"
    s = str(raw).strip().strip('"').strip(".").lower()
    if "decept" in s or s in ("d", "lie", "lying", "false"):
        return "Deceptive"
    if "truth" in s or s in ("t", "honest", "sincere", "true"):
        return "Truthful"
    return "Unknown"

# --------------------------------------------------------------------------
# 5. Metrics
# --------------------------------------------------------------------------
def safe_div(num, den):
    return num / den if den else 0.0


def confusion(y_true, y_pred):
    """Confusion counts with 'Deceptive' as the positive class."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == "Deceptive" and p == "Deceptive":
            tp += 1
        elif t == "Truthful" and p == "Deceptive":
            fp += 1
        elif t == "Deceptive" and p == "Truthful":
            fn += 1
        else:
            tn += 1
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def metrics_from_counts(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = safe_div(tp + tn, total)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return {"support": total, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "F1": round(f1, 4)}


def compute_metrics(rows):
    """rows: list of dicts containing Mapped_Label (gold) and Predicted_Label."""
    y_true = [r["Mapped_Label"] for r in rows]
    y_pred = [r["Predicted_Label"] for r in rows]
    c = confusion(y_true, y_pred)
    overall = metrics_from_counts(c["TP"], c["FP"], c["FN"], c["TN"])

    per_condition = OrderedDict()
    for cond in CONDITIONS:
        per_condition[cond] = OrderedDict()
        for val in sorted({r[cond] for r in rows}):
            sub = [r for r in rows if r[cond] == val]
            csub = confusion([r["Mapped_Label"] for r in sub],
                             [r["Predicted_Label"] for r in sub])
            per_condition[cond][val] = metrics_from_counts(
                csub["TP"], csub["FP"], csub["FN"], csub["TN"])
    return {"overall": overall, "per_condition": per_condition}


def summarize_results(results):
    """Gold-label counts and Unknown-prediction rate."""
    gold = OrderedDict((lab, sum(1 for rec, _, _ in results
                                 if rec["Mapped_Label"] == lab)) for lab in LABELS)
    n_unknown = sum(1 for _, _, p in results if p == "Unknown")
    return {"n_records": len(results),
            "n_unknown_predictions": n_unknown,
            "gold_counts": gold}

# --------------------------------------------------------------------------
# 6. Output writers
# --------------------------------------------------------------------------
def write_results_csv(results, path=RESULTS_CSV):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(REQUIRED_FIELDS + ["Raw_LLM_Output", "Predicted_Label", "Correct"])
        for rec, raw, pred in results:
            w.writerow([rec[f] for f in REQUIRED_FIELDS] +
                       [raw, pred, str(pred == rec["Mapped_Label"])])
    return path


def write_summary_csv(metrics, extra, path=SUMMARY_CSV):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["Scope", "Condition", "Value", "Support",
                    "Accuracy", "Precision", "Recall", "F1"])
        o = metrics["overall"]
        w.writerow(["overall", "-", "-", o["support"], o["Accuracy"],
                    o["Precision"], o["Recall"], o["F1"]])
        for cond, buckets in metrics["per_condition"].items():
            for val, m in buckets.items():
                w.writerow([f"by_{cond}", cond, val, m["support"], m["Accuracy"],
                            m["Precision"], m["Recall"], m["F1"]])
        w.writerow(["meta", "-", "n_records", extra["n_records"], "", "", "", ""])
        w.writerow(["meta", "-", "n_unknown_predictions",
                    extra["n_unknown_predictions"], "", "", "", ""])
    return path


def write_summary_json(metrics, extra, meta, path=SUMMARY_JSON):
    payload = OrderedDict()
    payload["generated_at"] = meta["generated_at"]
    payload["mode"] = meta["mode"]
    payload["template"] = meta["template"]
    payload["source_file"] = os.path.basename(meta.get("prompts_file", PROMPTS_FILE))
    payload["n_records"] = extra["n_records"]
    payload["gold_counts"] = extra["gold_counts"]
    payload["n_unknown_predictions"] = extra["n_unknown_predictions"]
    payload["overall"] = metrics["overall"]
    payload["per_condition"] = metrics["per_condition"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path

# --------------------------------------------------------------------------
# 7. Main / runner
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate LLM deception classification.")
    ap.add_argument("--real", action="store_true",
                    help="call a real LLM API instead of the built-in simulator")
    ap.add_argument("--template", default="zero_shot", choices=list(TEMPLATES),
                    help="prompt template to use")
    ap.add_argument("--api-key", default=None, help="API key for real LLM calls")
    ap.add_argument("--model", default=None, help="model name for real LLM calls")
    ap.add_argument("--prompts-file", default=PROMPTS_FILE)
    args = ap.parse_args(argv)

    print("[1/4] Parsing %s ..." % os.path.basename(args.prompts_file))
    records = parse_prompts(args.prompts_file)
    validate(records)
    print("      parsed %d records" % len(records))

    print("[2/4] Building prompts (template: %s) ..." % args.template)
    print("[3/4] Classifying (%s) ..." % ("real LLM" if args.real else "simulator"))
    results = classify_records(records, use_real_llm=args.real,
                               template_name=args.template,
                               model=args.model, api_key=args.api_key)

    print("[4/4] Computing metrics and writing outputs ...")
    rows = [dict(rec, Raw_LLM_Output=raw, Predicted_Label=pred)
            for rec, raw, pred in results]
    metrics = compute_metrics(rows)
    extra = summarize_results(results)
    meta = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": "real_llm" if args.real else "simulated",
            "template": args.template,
            "prompts_file": args.prompts_file}

    write_results_csv(results)
    write_summary_csv(metrics, extra)
    write_summary_json(metrics, extra, meta)

    o = metrics["overall"]
    print("\n=== Overall metrics (positive class = Deceptive) ===")
    print("  Support   : %d" % o["support"])
    print("  Accuracy  : %.4f" % o["Accuracy"])
    print("  Precision : %.4f" % o["Precision"])
    print("  Recall    : %.4f" % o["Recall"])
    print("  F1        : %.4f" % o["F1"])
    for cond, buckets in metrics["per_condition"].items():
        print("\n  --- by %s ---" % cond)
        for val, m in buckets.items():
            print("    %-11s n=%-4d acc=%.3f P=%.3f R=%.3f F1=%.3f"
                  % (val, m["support"], m["Accuracy"], m["Precision"],
                     m["Recall"], m["F1"]))
    print("\nSaved: %s\nSaved: %s\nSaved: %s" % (RESULTS_CSV, SUMMARY_CSV, SUMMARY_JSON))
    return 0


if __name__ == "__main__":
    sys.exit(main())
