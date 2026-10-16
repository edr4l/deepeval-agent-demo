"""
test_tool_agent_csv.py
=======================
Same pattern as test_TracingComponentsTest.py, but the goldens are
externalised to evals/goldens/shopease_tool_agent.csv instead of being
hard-coded in this file. This is the version to scale up: add rows to
the CSV, nothing in this file needs to change.

NOTE: ToolCorrectnessMetric is deliberately not used here. It requires
`tools_called` to be populated from the trace, and on this DeepEval version
that comes back None even for goldens where a tool genuinely should be (and
was) called — a tracing-integration issue, not a golden-design one (the
repo's own README flags this exact integration surface as version-shaky).
Everything runs through TaskCompletionMetric only for now, whose judge reads
expected_output and scores the full response, tool use included.

CSV columns:
    bucket           - happy_path | edge_case | adversarial | ambiguous
                        (for your own reporting/filtering; not passed to
                        DeepEval, just keeps the spec organised)
    input            - the user message
    expected_output  - free-text description of what a correct answer
                        contains (used by TaskCompletionMetric's judge)
    expected_tools   - pipe-delimited tool names, e.g. "get_order_status|get_shipping_options"
                        (kept in the CSV/goldens for your own records and for
                        when tool-correctness checking gets revisited; not
                        currently passed to any metric)

Run (hosted judge, default):
    python evals/test_tool_agent_csv.py

Run (local Ollama judge):
    EVAL_JUDGE=ollama:llama3.1 python evals/test_tool_agent_csv.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from agent_instrumented import support_agent as _support_agent
from judges import get_judge

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens", "shopease_tool_agent.csv")


def load_goldens_from_csv(path: str) -> list[Golden]:
    goldens = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tool_names = [t.strip() for t in row["expected_tools"].split("|") if t.strip()]
            goldens.append(
                Golden(
                    input=row["input"].strip(),
                    expected_output=row["expected_output"].strip() or None,
                    # Empty list (not None) when no tool call is expected —
                    # ToolCorrectnessMetric needs something to compare against,
                    # even for the "correct answer is: call nothing" cases.
                    expected_tools=[ToolCall(name=n) for n in tool_names],
                )
            )
    return goldens


# Passes each golden's expected_output onto the live trace so
# TaskCompletionMetric can score against it.
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden and golden.expected_output:
        update_current_trace(expected_output=golden.expected_output)
    return _support_agent(user_input)


task_completion = TaskCompletionMetric(threshold=0.7, model=get_judge())

dataset = EvaluationDataset(goldens=load_goldens_from_csv(CSV_PATH))

print(f"Running {len(dataset.goldens)} goldens with task-completion (sequentially)...")
# run_async=False makes evals_iterator process one golden at a time instead
# of firing all of them concurrently — this is what was hitting both the
# OpenAI TPM rate limit and, with the Ollama judge, the local throughput
# ceiling. Slower overall, but no capacity ceiling to trip over.
for golden in dataset.evals_iterator(
    metrics=[task_completion],
    async_config=AsyncConfig(run_async=False),
):
    support_agent(golden.input)
