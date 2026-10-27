"""
load_synthetic_traces.py
========================
Push synthetic traces into Braintrust for review in the Logs UI.
These are pre-generated input/output pairs (some with intentional failures)
designed for practising the systematic error analysis workflow.

Usage:
    python load_synthetic_traces.py
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import braintrust

_DIR = Path(__file__).resolve().parent

braintrust.init_logger(project="deepeval-agent-demo")


def main():
    with open(_DIR / "synthetic_traces.json") as f:
        traces = json.load(f)

    for trace in traces:
        span = braintrust.start_span(name="support_agent")
        span.log(
            input={"user_input": trace["input"]},
            output=trace["output"],
            metadata={"trace_id": trace["id"], "source": "synthetic"},
        )
        span.end()
        print(f"[{trace['id']:02d}/{len(traces)}] {trace['input'][:60]}...")

    braintrust.flush()
    print(f"\nDone. {len(traces)} synthetic traces pushed to Braintrust.")
    print("Open braintrust.dev → deepeval-agent-demo → Logs to review them.")


if __name__ == "__main__":
    main()
