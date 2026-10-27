"""
run_traces.py
=============
Run all inputs from trace_inputs.json through the support agent.
If BRAINTRUST_API_KEY is set, traces appear in Braintrust UI.
Also saves a local traces.csv as backup.

Usage:
    python run_traces.py
"""

import json
import csv
import traceback
from agent_instrumented import support_agent


def main():
    with open("trace_inputs.json") as f:
        inputs = json.load(f)

    results = []
    for i, input_text in enumerate(inputs, 1):
        print(f"[{i:02d}/{len(inputs)}] {input_text[:70]}...")

        try:
            output = support_agent(input_text)
        except Exception as e:
            output = f"ERROR: {type(e).__name__}: {e}"
            traceback.print_exc()

        results.append({
            "id": i,
            "input": input_text,
            "output": output,
        })

        print(f"  → {output[:80]}...\n")

    # Local backup — just input/output, no pre-labelling
    with open("traces.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "input", "output"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. {len(results)} traces written to traces.csv")
    print("If BRAINTRUST_API_KEY was set, traces are also at braintrust.dev")


if __name__ == "__main__":
    main()
