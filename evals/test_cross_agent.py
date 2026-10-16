"""
Sprint 4 Exercise 1 — Cross-Agent Consistency
Same policy questions asked to both agents. Do they give the same facts?
If not, there's a spec gap — the agents derive from different sources or
interpret the same source differently.
"""
import os, sys
os.environ["DEEPEVAL_NO_INSPECT_PROMPT"] = "1"
os.environ["OPIK_USAGE_REPORT_ENABLED"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluate import evaluate, AsyncConfig
from rag_agent import rag_support_agent
from agent_instrumented import support_agent

# ---------------------------------------------------------------------------
# METRIC
# ---------------------------------------------------------------------------

cross_agent_consistency = GEval(
    name="Cross-Agent Consistency",
    criteria=(
        "The expected output is the answer from Agent A. The actual output is "
        "the answer from Agent B, to the same question. "
        "Compare ONLY the factual claims: numbers, timeframes, conditions, "
        "eligibility rules. Ignore tone, formatting, greetings, emojis. "
        "Score 1.0 if every factual claim in one answer matches the other. "
        "Score 0.5 if most facts match but one or two minor details differ. "
        "Score 0.0 if key facts contradict each other — different numbers, "
        "different conditions, or one states a policy the other denies."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.7, model="gpt-4o",
)

# ---------------------------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What is the return policy for electronics?",
    "How long does standard shipping take?",
    "Can I cancel my order?",
]


def build_cases():
    cases = []
    for q in QUESTIONS:
        rag_answer = rag_support_agent(q)
        tool_answer = support_agent(q)
        print(f"\n=== Q: {q} ===")
        print(f"RAG Agent:  {rag_answer[:150]}...")
        print(f"Tool Agent: {tool_answer[:150]}...")
        cases.append(LLMTestCase(
            input=q,
            actual_output=tool_answer,
            expected_output=rag_answer,
        ))
    return cases


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Cross-Agent Consistency: RAG Agent vs Tool Agent")
    print("=" * 60)
    cases = build_cases()
    evaluate(
        test_cases=cases,
        metrics=[cross_agent_consistency],
        async_config=AsyncConfig(run_async=False),
    )
