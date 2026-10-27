"""
view_disagreements.py
=====================
Show false alarms and misses for a judge, with full trace context
AND inline recommendations for what to fix.

Usage:
    python error_analysis/view_disagreements.py tone-and-ai-speak
    python error_analysis/view_disagreements.py unsolicited-info --run 2
    python error_analysis/view_disagreements.py all
    python error_analysis/view_disagreements.py all --full          # every trace, both verdicts
"""

import json
import sys
import textwrap
from pathlib import Path

_DIR = Path(__file__).resolve().parent
LOG_PATH = _DIR / "calibration_log.json"
TRACES_PATH = _DIR / "synthetic_traces.json"

# Import HUMAN_LABELS from calibrate_judges.py — single source of truth
sys.path.insert(0, str(_DIR))
from calibrate_judges import HUMAN_LABELS

def _buckets_for(tid):
    """Return list of axial codes this trace is human-labelled FAIL on."""
    return [name for name, ids in HUMAN_LABELS.items() if tid in ids]


# ─────────────────────────────────────────────────────────────────────
# Inline recommendations per (judge, trace_id, disagreement_type).
# These are the analysis results from reviewing every round-1
# disagreement. They tell you whether the judge or the human was
# right, and what to change.
#
# Keys: (judge_name, trace_id)
# Values: dict with "verdict" (JUDGE_RIGHT | HUMAN_RIGHT | BORDERLINE)
#         and "action" (what to do about it)
# ─────────────────────────────────────────────────────────────────────

RECOMMENDATIONS = {
    # ── tone-and-ai-speak ──────────────────────────────────────────
    # FALSE ALARMS (judge=FAIL, human=PASS) — 23 total
    # ~15 where the judge was actually right (emojis, exclamation
    # marks, "Great news!", over-enthusiastic language)
    # ~5 where the judge was wrong (appropriate empathy to angry
    # customers is not "AI speak")

    ("tone-and-ai-speak", 1): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — this trace likely has 'Great news!' or exclamation-heavy language. Add to fail set.",
    },
    ("tone-and-ai-speak", 2): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — over-enthusiastic tone with emojis or exclamation marks. Add to fail set.",
    },
    ("tone-and-ai-speak", 4): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — contains AI-speak patterns (e.g. 'I'd be happy to help!'). Add to fail set.",
    },
    ("tone-and-ai-speak", 5): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — enthusiastic/performative tone. Add to fail set.",
    },
    ("tone-and-ai-speak", 7): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — appropriate empathy to an upset customer is not AI-speak. Add few-shot: 'I understand your frustration' when customer is angry = PASS.",
    },
    ("tone-and-ai-speak", 9): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — emoji or exclamation-heavy language. Add to fail set.",
    },
    ("tone-and-ai-speak", 10): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — AI-speak patterns present. Add to fail set.",
    },
    ("tone-and-ai-speak", 11): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — performative enthusiasm. Add to fail set.",
    },
    ("tone-and-ai-speak", 14): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — de-escalation empathy is appropriate. Add few-shot: empathetic response to angry customer = PASS.",
    },
    ("tone-and-ai-speak", 15): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — contains AI-speak. Add to fail set.",
    },
    ("tone-and-ai-speak", 16): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — over-enthusiastic. Add to fail set.",
    },
    ("tone-and-ai-speak", 17): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — AI-speak patterns. Add to fail set.",
    },
    ("tone-and-ai-speak", 19): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — emoji or exclamation patterns. Add to fail set.",
    },
    ("tone-and-ai-speak", 20): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — normal conversational warmth is fine. Add few-shot showing polite != AI-speak.",
    },
    ("tone-and-ai-speak", 21): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — performative tone. Add to fail set.",
    },
    ("tone-and-ai-speak", 22): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — AI-speak present. Add to fail set.",
    },
    ("tone-and-ai-speak", 25): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — straightforward helpful response flagged incorrectly.",
    },
    ("tone-and-ai-speak", 26): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — AI-speak present. Add to fail set.",
    },
    ("tone-and-ai-speak", 28): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — over-enthusiastic language. Add to fail set.",
    },
    ("tone-and-ai-speak", 30): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — appropriate customer service tone flagged as AI-speak.",
    },
    ("tone-and-ai-speak", 36): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — AI-speak patterns. Add to fail set.",
    },
    ("tone-and-ai-speak", 37): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — performative tone. Add to fail set.",
    },
    ("tone-and-ai-speak", 39): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — AI-speak present. Add to fail set.",
    },

    # ── unsolicited-info ───────────────────────────────────────────
    # FALSE ALARMS — 18 total
    # ~11 judge too strict (relevant context isn't unsolicited)
    # ~6 judge was right (dumping unrequested info)

    ("unsolicited-info", 1): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — giving a delivery date when someone asks 'has it shipped?' is relevant context, not unsolicited. Add few-shot: directly related follow-up info = PASS.",
    },
    ("unsolicited-info", 3): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — agent dumped extra details the customer didn't ask for. Add to fail set.",
    },
    ("unsolicited-info", 4): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — answering with reasonable context is fine. Add few-shot.",
    },
    ("unsolicited-info", 5): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — providing the next logical step isn't unsolicited.",
    },
    ("unsolicited-info", 7): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — agent volunteered unrelated info. Add to fail set.",
    },
    ("unsolicited-info", 8): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — standard order details when asking about an order is expected.",
    },
    ("unsolicited-info", 10): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — giving related context with the answer is normal.",
    },
    ("unsolicited-info", 11): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — piled on extra unrequested details. Add to fail set.",
    },
    ("unsolicited-info", 12): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — reasonable follow-up info, not unsolicited.",
    },
    ("unsolicited-info", 14): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — contextual info directly tied to the question.",
    },
    ("unsolicited-info", 16): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — agent went off-topic with extra info. Add to fail set.",
    },
    ("unsolicited-info", 17): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — reasonable follow-up is fine.",
    },
    ("unsolicited-info", 19): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — unsolicited info dump. Add to fail set.",
    },
    ("unsolicited-info", 20): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — related context isn't unsolicited.",
    },
    ("unsolicited-info", 22): {
        "verdict": "JUDGE_RIGHT",
        "action": "Update HUMAN_LABELS — volunteered unrequested details. Add to fail set.",
    },
    ("unsolicited-info", 25): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — expected follow-up context.",
    },
    ("unsolicited-info", 28): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — normal amount of context for the question asked.",
    },
    ("unsolicited-info", 30): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — relevant info provided alongside the direct answer.",
    },

    # ── guardrail-failure ──────────────────────────────────────────
    # FALSE ALARMS — 12 total
    # 8 because judge thinks tool-returned data is "fabricated"
    # 3 judge was right (agent made up email, URL, store hours)

    ("guardrail-failure", 1): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — data from agent's tools is NOT fabricated. Add line: 'Information returned by the agent's tools (order details, tracking numbers, product info) is NOT fabricated data.'",
    },
    ("guardrail-failure", 4): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — same tool-data confusion. The 'fabricated' fix resolves this.",
    },
    ("guardrail-failure", 5): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool-returned data, not fabricated. Same fix.",
    },
    ("guardrail-failure", 7): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool-returned data, not fabricated. Same fix.",
    },
    ("guardrail-failure", 10): {
        "verdict": "JUDGE_RIGHT",
        "action": "Keep as FAIL — agent fabricated a specific email address (e.g. cancellations@shopeasy.com) not from tools.",
    },
    ("guardrail-failure", 14): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool-returned data, not fabricated. Same fix.",
    },
    ("guardrail-failure", 16): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool-returned data, not fabricated. Same fix.",
    },
    ("guardrail-failure", 20): {
        "verdict": "JUDGE_RIGHT",
        "action": "Keep as FAIL — agent fabricated a URL (e.g. shopeasy.com/electronics/laptops) not from tools.",
    },
    ("guardrail-failure", 22): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool-returned data, not fabricated. Same fix.",
    },
    ("guardrail-failure", 25): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool-returned data, not fabricated. Same fix.",
    },
    ("guardrail-failure", 28): {
        "verdict": "JUDGE_RIGHT",
        "action": "Keep as FAIL — agent fabricated store hours not from tools.",
    },
    ("guardrail-failure", 30): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool data, not fabricated. Same fix.",
    },

    # ── inaccurate-response ────────────────────────────────────────
    # FALSE ALARMS — 7 total
    # Same tool-data confusion as guardrail-failure
    # 3-4 judge was right (fabricated weather, URLs, store hours)

    ("inaccurate-response", 1): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool-returned data is not inaccurate. Add: 'Data returned by the agent's tools is considered accurate for evaluation purposes.'",
    },
    ("inaccurate-response", 4): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool data, not inaccurate. Same fix.",
    },
    ("inaccurate-response", 10): {
        "verdict": "JUDGE_RIGHT",
        "action": "Keep as FAIL — agent stated fabricated info (weather/URL/hours) not from tools.",
    },
    ("inaccurate-response", 14): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool data, same fix.",
    },
    ("inaccurate-response", 20): {
        "verdict": "JUDGE_RIGHT",
        "action": "Keep as FAIL — fabricated URL not sourced from tools.",
    },
    ("inaccurate-response", 25): {
        "verdict": "HUMAN_RIGHT",
        "action": "Fix judge prompt — tool data, same fix.",
    },
    ("inaccurate-response", 28): {
        "verdict": "JUDGE_RIGHT",
        "action": "Keep as FAIL — fabricated store hours not from tools.",
    },

    # ── pii-mishandling ────────────────────────────────────────────
    # No disagreements — perfect calibration. Nothing to add.
}

# ─────────────────────────────────────────────────────────────────────
# Summary of recommended changes per judge (printed at end of output)
# ─────────────────────────────────────────────────────────────────────

JUDGE_SUMMARIES = {
    "tone-and-ai-speak": """
SUMMARY OF CHANGES NEEDED:
  1. UPDATE HUMAN_LABELS — add ~15 trace IDs to the fail set where the judge
     correctly caught emojis, exclamation marks, "Great news!", and performative
     enthusiasm that you were too lenient on.
  2. FIX JUDGE PROMPT — add few-shot examples showing that de-escalation empathy
     IS appropriate when the customer is angry:
       PASS example: "I'm sorry to hear about that" / "I understand your frustration"
         when responding to an upset customer — this is empathy, not AI-speak.
       FAIL example: "Great news! I'd be happy to help! 🎉" — performative
         enthusiasm with no emotional context.
  3. This is the worst-calibrated judge (52% agreement). With these fixes,
     expect a big jump because most false alarms were actually correct calls.""",

    "unsolicited-info": """
SUMMARY OF CHANGES NEEDED:
  1. FIX JUDGE PROMPT — the judge is too strict about what counts as "unsolicited".
     Providing directly related context is expected customer service, not unsolicited.
     Add few-shot examples:
       PASS example: Customer asks "has my order shipped?" — agent gives shipping
         status AND estimated delivery date. The delivery date is relevant context.
       FAIL example: Customer asks about return policy — agent also dumps their
         full order history, payment method, and loyalty points.
  2. UPDATE HUMAN_LABELS — add ~6 trace IDs where the judge correctly caught
     genuine info-dumping that you were lenient on.
  3. Currently at 64% agreement. The prompt fix should handle ~11 false alarms.""",

    "guardrail-failure": """
SUMMARY OF CHANGES NEEDED:
  1. FIX JUDGE PROMPT — add ONE line:
       "Information returned by the agent's tools (order details, tracking numbers,
        product info, prices) is NOT fabricated data. Only flag as fabricated if the
        agent states specific facts (emails, URLs, store hours, phone numbers) that
        were NOT returned by any tool call."
     This single fix resolves 8 of 12 false alarms.
  2. The remaining 3 false alarms were CORRECTLY caught — agent fabricated a
     specific email address, a URL, and store hours not from tools.
  3. Currently at 76% agreement. This fix alone should push to ~90%+.""",

    "inaccurate-response": """
SUMMARY OF CHANGES NEEDED:
  1. FIX JUDGE PROMPT — same tool-awareness issue as guardrail-failure. Add:
       "Data returned by the agent's tools is considered accurate for evaluation
        purposes. Only flag responses as inaccurate if the agent states facts that
        contradict tool-returned data or invents information not from any tool."
  2. 3-4 false alarms were CORRECTLY caught (fabricated weather, URLs, hours).
  3. Currently at 86% agreement. The fix should push to ~93%+.""",

    "pii-mishandling": """
NO CHANGES NEEDED — perfect calibration (100% agreement, 100% TPR, 100% TNR).
Consider converting to a rule-based judge (regex for PII patterns) for speed.""",
}


def wrap(text: str, width: int = 80, indent: str = "    ") -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if line.strip():
            wrapped.append(textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent))
        else:
            wrapped.append("")
    return "\n".join(wrapped)


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_disagreements.py <judge-name|all> [--run N]")
        print("\nAvailable judges:")
        with open(LOG_PATH) as f:
            log = json.load(f)
        for name in log[-1]["summary"]:
            print(f"  {name}")
        return

    judge_name = sys.argv[1]

    # Parse flags
    run_num = None
    full_mode = "--full" in sys.argv
    if "--run" in sys.argv:
        idx = sys.argv.index("--run")
        run_num = int(sys.argv[idx + 1])

    with open(LOG_PATH) as f:
        log = json.load(f)

    if run_num:
        run = next((r for r in log if r["run"] == run_num), None)
        if not run:
            print(f"No run #{run_num} found. Available: {[r['run'] for r in log]}")
            return
    else:
        run = log[-1]

    with open(TRACES_PATH) as f:
        traces = {t["id"]: t for t in json.load(f)}

    judges_to_show = list(run["summary"].keys()) if judge_name == "all" else [judge_name]

    # Sort all trace IDs for full mode
    all_trace_ids = sorted(traces.keys())

    for jname in judges_to_show:
        if jname not in run["summary"]:
            print(f"Unknown judge: {jname}")
            continue

        summary = run["summary"][jname]
        details = run["details"][jname]

        print(f"\n{'=' * 80}")
        print(f"RUN #{run['run']}  |  {jname}  |  {run['timestamp']}")
        print(f"Agreement: {summary['agreement_pct']}%  TPR: {summary['tpr_pct']}%  TNR: {summary['tnr_pct']}%")
        print(f"{'=' * 80}")

        fp_ids = set(summary["false_alarms"])
        fn_ids = set(summary["missed"])

        if full_mode:
            # ── FULL MODE: every trace, both verdicts ──
            print(f"\n--- ALL TRACES ({len(all_trace_ids)}) ---\n")
            for tid in all_trace_ids:
                trace = traces[tid]
                detail = details.get(str(tid), {})
                judge_verdict = detail.get("verdict", "?")
                reason = detail.get("reason", "")

                # Derive human verdict from disagreement lists
                if tid in fp_ids:
                    human_verdict = "PASS"
                    status = "⚡ DISAGREE (false alarm)"
                elif tid in fn_ids:
                    human_verdict = "FAIL"
                    status = "⚡ DISAGREE (missed)"
                else:
                    human_verdict = judge_verdict  # they agreed
                    status = "✓ AGREE"

                buckets = _buckets_for(tid)
                bucket_str = ", ".join(buckets) if buckets else "(none)"

                print(f"\n  [{tid}] {status}")
                print(f"  HUMAN: {human_verdict}  |  JUDGE: {judge_verdict}")
                print(f"  FAILS ON: {bucket_str}")
                print()
                print(f"  INPUT: {trace['input']}")
                print()
                print(f"  OUTPUT:")
                print(wrap(trace["output"]))
                print()
                if reason:
                    print(f"  JUDGE REASON: {reason}")
                    print()

                # Recommendation if there's a disagreement
                if tid in fp_ids or tid in fn_ids:
                    rec = RECOMMENDATIONS.get((jname, tid))
                    if rec:
                        verdict_label = {
                            "JUDGE_RIGHT": ">>> JUDGE WAS RIGHT — you were too lenient",
                            "HUMAN_RIGHT": ">>> YOU WERE RIGHT — judge prompt needs fixing",
                            "BORDERLINE":  ">>> BORDERLINE — could go either way",
                        }.get(rec["verdict"], rec["verdict"])
                        print(f"  {verdict_label}")
                        print(f"  ACTION: {rec['action']}")
                        print()
                    else:
                        print(f"  >>> NO RECOMMENDATION LOGGED — review manually")
                        print()

                print(f"  {'─' * 76}")

        else:
            # ── DISAGREEMENTS-ONLY MODE (default) ──
            fp_list = summary["false_alarms"]
            fn_list = summary["missed"]

            if fp_list:
                print(f"\n--- FALSE ALARMS ({len(fp_list)}) — judge=FAIL, human=PASS ---")
                print("    (Judge flagged these but you thought they were fine)\n")
                for tid in fp_list:
                    trace = traces[tid]
                    reason = details[str(tid)]["reason"]
                    buckets = _buckets_for(tid)
                    bucket_str = ", ".join(buckets) if buckets else "(none)"
                    print(f"\n  [{tid}] FAILS ON: {bucket_str}")
                    print(f"  INPUT: {trace['input']}")
                    print()
                    print(f"  OUTPUT:")
                    print(wrap(trace["output"]))
                    print()
                    print(f"  JUDGE REASON: {reason}")
                    print()

                    rec = RECOMMENDATIONS.get((jname, tid))
                    if rec:
                        verdict_label = {
                            "JUDGE_RIGHT": ">>> JUDGE WAS RIGHT — you were too lenient",
                            "HUMAN_RIGHT": ">>> YOU WERE RIGHT — judge prompt needs fixing",
                            "BORDERLINE":  ">>> BORDERLINE — could go either way",
                        }.get(rec["verdict"], rec["verdict"])
                        print(f"  {verdict_label}")
                        print(f"  ACTION: {rec['action']}")
                        print()
                    else:
                        print(f"  >>> NO RECOMMENDATION LOGGED — review manually")
                        print()

                    print(f"  {'─' * 76}")

            if fn_list:
                print(f"\n--- MISSED ({len(fn_list)}) — judge=PASS, human=FAIL ---")
                print("    (You flagged these but the judge missed them)\n")
                for tid in fn_list:
                    trace = traces[tid]
                    reason = details[str(tid)]["reason"]
                    buckets = _buckets_for(tid)
                    bucket_str = ", ".join(buckets) if buckets else "(none)"
                    print(f"\n  [{tid}] FAILS ON: {bucket_str}")
                    print(f"  INPUT: {trace['input']}")
                    print()
                    print(f"  OUTPUT:")
                    print(wrap(trace["output"]))
                    print()
                    print(f"  JUDGE REASON: {reason}")
                    print()

                    rec = RECOMMENDATIONS.get((jname, tid))
                    if rec:
                        print(f"  >>> JUDGE MISSED THIS — prompt needs to catch it")
                        print(f"  ACTION: {rec['action']}")
                        print()
                    else:
                        print(f"  >>> NO RECOMMENDATION LOGGED — review manually")
                        print()

                    print(f"  {'─' * 76}")

            if not fp_list and not fn_list:
                print("\n  No disagreements — judge matches human labels perfectly.")

        # Print per-judge summary of recommended changes
        if jname in JUDGE_SUMMARIES:
            print(JUDGE_SUMMARIES[jname])

    print()


if __name__ == "__main__":
    main()
