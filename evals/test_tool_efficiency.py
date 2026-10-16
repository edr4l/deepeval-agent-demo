"""
Sprint 4 Exercise 2 — Tool Efficiency
Does the agent call tools when it shouldn't? Does it stay in scope?
Uses the RAG agent directly — no tracing dependency.
"""
import os, sys
os.environ["DEEPEVAL_NO_INSPECT_PROMPT"] = "1"
os.environ["OPIK_USAGE_REPORT_ENABLED"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluate import evaluate, AsyncConfig
from rag_agent import rag_support_agent

# ---------------------------------------------------------------------------
# METRIC
# ---------------------------------------------------------------------------

unnecessary_tool_use = GEval(
    name="Unnecessary Tool Use",
    criteria=(
        "The expected output describes whether a tool call was needed for this query. "
        "Check whether the agent's response suggests it searched its knowledge base "
        "when it didn't need to. "
        "Signs of an unnecessary search: quoting policy details in response to a "
        "greeting or off-topic question, or listing policy categories unprompted. "
        "Signs of correct behaviour: a short conversational reply to a greeting, "
        "or a decline for off-topic questions, with no policy details. "
        "Score 1.0 if the agent's response matches the expected behaviour. "
        "Score 0.0 if it searched policies when it shouldn't have, or failed "
        "to search when it should have."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.7, model="gpt-4o",
)

# ---------------------------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------------------------

QUERIES = [
    (
        "Hi there, how are you doing today?",
        "No search needed. Agent should reply conversationally without "
        "quoting any policy details or listing topics it can help with."
    ),
    (
        "What's the weather like?",
        "No search needed. Agent should decline — weather is outside its scope. "
        "It should NOT search policies or list its available topics."
    ),
    (
        "What is your return policy for clothing?",
        "Search needed. Agent should look up the clothing return policy and "
        "provide specific details (30 days, unworn, tags attached, etc)."
    ),
    (
        "Thanks, that's all I needed!",
        "No search needed. Agent should respond politely to end the conversation "
        "without searching or quoting any policies."
    ),
]


def build_cases():
    cases = []
    for query, expected in QUERIES:
        output = rag_support_agent(query)
        print(f"\n=== Q: {query} ===")
        print(f"Output: {output[:200]}...")
        cases.append(LLMTestCase(
            input=query,
            actual_output=output,
            expected_output=expected,
        ))
    return cases


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Tool Efficiency: Right tool, right time")
    print("=" * 60)
    cases = build_cases()
    evaluate(
        test_cases=cases,
        metrics=[unnecessary_tool_use],
        async_config=AsyncConfig(run_async=False),
    )
