import os
import sys

os.environ["DEEPEVAL_NO_INSPECT_PROMPT"] = "1"
os.environ["OPIK_USAGE_REPORT_ENABLED"] = "false"

import opik
opik.configure(use_local=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepeval.contextvars import get_current_golden
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, AnswerRelevancyMetric, \
    FaithfulnessMetric
from deepeval.tracing import observe, update_current_trace
from deepeval.evaluate import AsyncConfig

from rag_agent import rag_support_agent as _rag_support_agent

@opik.track
@observe(name="rag_support_agent")
def rag_support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _rag_support_agent(user_input)

dataset = EvaluationDataset(goldens=[
    Golden(
        input="What is the return policy for electronics?",
        expected_output=(
            "Electronics can be returned within 30 days for a full refund, "
            "no questions asked, even if opened and used."
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

precisionMetric = ContextualPrecisionMetric(threshold=0.7, model="gpt-4o")
recallMetric = ContextualRecallMetric(threshold=0.7, model="gpt-4o", include_reason=True)
relevancyMetric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o", include_reason=True)
faithfulMetric = FaithfulnessMetric(threshold=0.7, model="gpt-4o", include_reason=True)

for golden in dataset.evals_iterator(
    metrics=[precisionMetric, recallMetric, relevancyMetric, faithfulMetric],
    async_config=AsyncConfig(run_async=False),
):
    rag_support_agent(golden.input)
    opik.flush_tracker()