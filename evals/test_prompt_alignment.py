import os, sys
os.environ["DEEPEVAL_NO_INSPECT_PROMPT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval, FaithfulnessMetric
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

# Custom metric: does the agent obey "answer only from retrieved information"?
grounding_compliance = GEval(
    name="Grounding Compliance",
    criteria=(
        "The agent's system prompt says 'Answer only from the retrieved information.' "
        "Score 1.0 if the response contains ONLY facts that appear in the retrieval context. "
        "Score 0.5 if it adds minor filler (greetings, 'anything else?') but no new facts. "
        "Score 0.0 if it introduces facts, numbers, or policy details not in the retrieval context."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.7,
    model="gpt-4o",
)

conciseness = GEval(
    name="Conciseness",
    criteria=(
        "The agent's system prompt says 'Keep replies concise.' "
        "Score 1.0 if the response is under 3 sentences of substance. "
        "Score 0.5 if 4-5 sentences. Score 0.0 if longer than 5 sentences."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model="gpt-4o",
)

faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o", include_reason=True)

dataset = EvaluationDataset(goldens=[
    # Normal question — should pass everything
    Golden(
        input="What is the return policy for electronics?",
        expected_output="Electronics can be returned within 15 days if unopened. Refunds take 5-7 business days.",
    ),
    # Out-of-scope question — agent should NOT make up an answer
    Golden(
        input="Do you offer gift wrapping?",
        expected_output="I don't have information about gift wrapping in our policies.",
    ),
    # Ambiguous question — tests whether agent stays grounded or embellishes
    Golden(
        input="Can I return something I bought last year?",
        expected_output="Based on our policies, return windows range from 15 to 30 days depending on category.",
    ),
])

for golden in dataset.evals_iterator(
    metrics=[grounding_compliance, conciseness, faithfulness],
    async_config=AsyncConfig(run_async=False),
):
    rag_support_agent(golden.input)