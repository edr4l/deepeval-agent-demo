import os, sys
os.environ["OPIK_USAGE_REPORT_ENABLED"] = "false"
os.environ["DEEPEVAL_NO_INSPECT_PROMPT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ToolCorrectnessMetric, TaskCompletionMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace
from deepeval.evaluate import AsyncConfig
from deepeval.contextvars import get_current_golden
from agent_instrumented import support_agent as _support_agent

@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _support_agent(user_input)

K = 1

golden = Golden(
    input="Where is my order ORD-1042?",
    expected_tools=[ToolCall(name="get_refund_policy")],
    expected_output="Your refund has been processed and will arrive in 5-7 business days."
)

'''
golden = Golden(
    input="I bought a laptop last month and it's broken, what can I do?",
    expected_tools=[ToolCall(name="get_refund_policy")],
    expected_output="Electronics can be returned within 15 days if unopened. Since the laptop is broken and past 15 days, a refund may not be available."
)

golden = Golden(
    input="Where is my order ORD-1042?",
    expected_tools=[ToolCall(name="get_order_status")],
    expected_output="Order ORD-1042 is currently shipped with an estimated delivery date."
)
'''

scores = {"TaskCompletion": [], "ToolCorrectness": []}

for i in range(K):
    ds = EvaluationDataset(goldens=[golden])
    tc = TaskCompletionMetric(threshold=0.7, model="gpt-4o")
    tool = ToolCorrectnessMetric()

    for g in ds.evals_iterator(
        metrics=[tc, tool],
        async_config=AsyncConfig(run_async=False),
    ):
        support_agent(g.input)

    scores["TaskCompletion"].append(tc.score)
    scores["ToolCorrectness"].append(tool.score)
    print(f"Run {i+1}: TaskCompletion={tc.score}, ToolCorrectness={tool.score}")

print(f"\n--- Results over {K} runs ---")
for name, vals in scores.items():
    valid = [v for v in vals if v is not None]
    if valid:
        avg = sum(valid) / len(valid)
        spread = max(valid) - min(valid)
        print(f"{name}: scores={vals}, avg={avg:.2f}, spread={spread:.2f}")
    else:
        print(f"{name}: all None")