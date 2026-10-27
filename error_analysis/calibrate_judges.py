"""
calibrate_judges.py
===================
Run all 6 binary judges against 50 synthetic traces, compare to Mar's
human labels, and print a confusion matrix per judge.

Metrics per judge:
  - Agreement %  (how often judge matches human)
  - TPR          (true positive rate — catches real failures)
  - TNR          (true negative rate — doesn't flag clean traces)

Usage:
    python calibrate_judges.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Resolve paths relative to this file, not cwd
_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from anthropic import Anthropic
from judge_prompts import JUDGES, JUDGE_SYSTEM

# --- Config ---
MODEL = "claude-sonnet-4-6"
TRACES_PATH = _DIR / "synthetic_traces.json"

# --- Human labels: which trace IDs fail each judge ---
# From Mar's open coding → axial coding on 50 synthetic traces
HUMAN_LABELS = {
    "emotional-and-emoji": {1, 2, 3, 5, 10, 12, 13, 39},
    "formatting": {6},
    "outlandish-claims": {8, 17, 19, 23, 38},
    "missing-suggestion": {11, 15},
    "off-rails": {29, 31, 42},
    "overempathy-overfamiliar": {34, 46},
}


def load_traces(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_judge(client: Anthropic, category: str, user_input: str, agent_output: str) -> tuple[str, str]:
    """Call the LLM judge and return ('PASS'|'FAIL', reason)."""
    judge = JUDGES[category]
    user_prompt = judge["prompt"].format(input=user_input, output=agent_output)

    response = client.messages.create(
        model=MODEL,
        max_tokens=150,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text.strip()
    verdict = "FAIL" if text.upper().startswith("FAIL") else "PASS"
    reason = text.split(":", 1)[1].strip() if ":" in text else text
    return verdict, reason


def main():
    client = Anthropic()
    traces = load_traces(TRACES_PATH)
    trace_map = {t["id"]: t for t in traces}

    all_ids = sorted(trace_map.keys())

    # Check for prompt overrides exported from the viewer
    overrides_path = _DIR / "prompt_overrides.json"
    if overrides_path.exists():
        with open(overrides_path) as f:
            overrides = json.load(f)
        for cat, prompt_text in overrides.items():
            if cat in JUDGES:
                JUDGES[cat]["prompt"] = prompt_text
                print(f"  ⚡ Override loaded for: {cat}")
        print()

    print(f"Running {len(JUDGES)} judges × {len(all_ids)} traces = {len(JUDGES) * len(all_ids)} calls\n")

    results = {}   # {category: {trace_id: "PASS"|"FAIL"}}
    reasons = {}   # {category: {trace_id: "reason string"}}

    for cat_name in JUDGES:
        print(f"  [{cat_name}] ", end="", flush=True)
        results[cat_name] = {}
        reasons[cat_name] = {}
        for tid in all_ids:
            trace = trace_map[tid]
            verdict, reason = run_judge(client, cat_name, trace["input"], trace["output"])
            results[cat_name][tid] = verdict
            reasons[cat_name][tid] = reason
            print("." if verdict == "PASS" else "F", end="", flush=True)
        print()

    # --- Confusion matrix per judge ---
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    for cat_name in JUDGES:
        human_fails = HUMAN_LABELS[cat_name]
        human_passes = set(all_ids) - human_fails

        judge_results = results[cat_name]

        # TP = human says fail AND judge says fail
        tp = sum(1 for tid in human_fails if judge_results[tid] == "FAIL")
        # FN = human says fail BUT judge says pass
        fn = sum(1 for tid in human_fails if judge_results[tid] == "PASS")
        # TN = human says pass AND judge says pass
        tn = sum(1 for tid in human_passes if judge_results[tid] == "PASS")
        # FP = human says pass BUT judge says fail
        fp = sum(1 for tid in human_passes if judge_results[tid] == "FAIL")

        total = tp + fn + tn + fp
        agreement = (tp + tn) / total * 100
        tpr = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

        print(f"\n--- {cat_name} ---")
        print(f"  Human fails: {sorted(human_fails)}")
        print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
        print(f"  Agreement: {agreement:.1f}%")
        print(f"  TPR (catches failures):    {tpr:.1f}%")
        print(f"  TNR (doesn't false-alarm): {tnr:.1f}%")

        # Show disagreements
        fn_ids = sorted(tid for tid in human_fails if judge_results[tid] == "PASS")
        fp_ids = sorted(tid for tid in human_passes if judge_results[tid] == "FAIL")
        if fn_ids:
            print(f"  MISSED (human=fail, judge=pass): {fn_ids}")
        if fp_ids:
            print(f"  FALSE ALARM (human=pass, judge=fail): {fp_ids}")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print(f"{'Judge':<28} {'Agree%':>8} {'TPR':>8} {'TNR':>8} {'FP':>5} {'FN':>5}")
    print("-" * 70)
    for cat_name in JUDGES:
        human_fails = HUMAN_LABELS[cat_name]
        human_passes = set(all_ids) - human_fails
        judge_results = results[cat_name]
        tp = sum(1 for tid in human_fails if judge_results[tid] == "FAIL")
        fn = sum(1 for tid in human_fails if judge_results[tid] == "PASS")
        tn = sum(1 for tid in human_passes if judge_results[tid] == "PASS")
        fp = sum(1 for tid in human_passes if judge_results[tid] == "FAIL")
        total = tp + fn + tn + fp
        agreement = (tp + tn) / total * 100
        tpr = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        print(f"  {cat_name:<26} {agreement:>7.1f}% {tpr:>7.1f}% {tnr:>7.1f}% {fp:>5} {fn:>5}")
    print("=" * 70)

    # --- Save to calibration log ---
    log_path = _DIR / "calibration_log.json"
    existing = []
    if log_path.exists():
        with open(log_path) as f:
            existing = json.load(f)

    run_entry = {
        "run": len(existing) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "prompts": {cat: JUDGES[cat]["prompt"] for cat in JUDGES},
        "human_labels": {cat: sorted(ids) for cat, ids in HUMAN_LABELS.items()},
        "summary": {},
        "details": {},
    }

    for cat_name in JUDGES:
        human_fails = HUMAN_LABELS[cat_name]
        human_passes = set(all_ids) - human_fails
        jr = results[cat_name]
        tp = sum(1 for tid in human_fails if jr[tid] == "FAIL")
        fn = sum(1 for tid in human_fails if jr[tid] == "PASS")
        tn = sum(1 for tid in human_passes if jr[tid] == "PASS")
        fp = sum(1 for tid in human_passes if jr[tid] == "FAIL")
        total = tp + fn + tn + fp

        fn_ids = sorted(tid for tid in human_fails if jr[tid] == "PASS")
        fp_ids = sorted(tid for tid in human_passes if jr[tid] == "FAIL")

        run_entry["summary"][cat_name] = {
            "agreement_pct": round((tp + tn) / total * 100, 1),
            "tpr_pct": round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0,
            "tnr_pct": round(tn / (tn + fp) * 100, 1) if (tn + fp) > 0 else 0,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "missed": fn_ids,
            "false_alarms": fp_ids,
        }

        run_entry["details"][cat_name] = {
            str(tid): {"verdict": jr[tid], "reason": reasons[cat_name][tid]}
            for tid in all_ids
        }

    existing.append(run_entry)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nLogged to {log_path} (run #{run_entry['run']})")


if __name__ == "__main__":
    main()
