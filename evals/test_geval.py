import os, sys
os.environ["DEEPEVAL_NO_INSPECT_PROMPT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.tracing import observe, update_current_trace
from deepeval.evaluate import AsyncConfig
from deepeval.contextvars import get_current_golden
from rag_agent import rag_support_agent as _rag_support_agent

@observe(name="rag_support_agent")
def rag_support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _rag_support_agent(user_input)

# Your custom metric — plain English criteria
tone_metric = GEval(
    name="Professional Tone",
    criteria=(
        "The response should use a professional, neutral customer-service tone. "
        "It should not use excessive emojis, slang, or overly casual language. "
        "It should not be robotic either — warm but professional."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model="gpt-4o",
)

policy_accuracy = GEval(
    name="Policy Accuracy",
    criteria=(
        "The response must state specific policy details (days, conditions, costs) "
        "that match the expected output. Vague or hedged answers score low. "
        "Contradicting the expected output scores 0."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model="gpt-4o",
)

dataset = EvaluationDataset(goldens=[
    Golden(
        input="What is the return policy for electronics?",
        expected_output=(
            "Electronics can be returned within 15 days of delivery if unopened "
            "and in original packaging. Refunds take 5–7 business days."
        ),
    ),
    Golden(
        input="How long does express shipping take and what does it cost?",
        expected_output=(
            "Express shipping takes 1–2 business days and costs $15. "
            "Orders placed before 2 PM are dispatched the same day."
        ),
    ),
])

for golden in dataset.evals_iterator(
    metrics=[tone_metric, policy_accuracy],
    async_config=AsyncConfig(run_async=False),
):
    rag_support_agent(golden.input)