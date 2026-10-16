"""
Sprint 4 Exercise 3 — Multi-Document Synthesis
Questions spanning multiple policy areas. Tests spec gap #2 (k=3 retrieval
ceiling) and whether the agent correctly merges facts from separate docs.
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
# METRICS
# ---------------------------------------------------------------------------

synthesis_accuracy = GEval(
    name="Synthesis Accuracy",
    criteria=(
        "The question requires combining facts from multiple policy areas. "
        "The expected output contains the correct combined answer. "
        "Compare ONLY the factual claims. Ignore tone and formatting. "
        "Score 1.0 if the response correctly covers ALL policy areas "
        "in the expected output with matching facts. "
        "Score 0.5 if it covers some areas but misses one entirely. "
        "Score 0.0 if it only covers one area or contradicts the expected output."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.7, model="gpt-4o",
)

completeness = GEval(
    name="Cross-Topic Completeness",
    criteria=(
        "The question asks about TWO distinct policy areas. "
        "Score 1.0 if the response addresses BOTH areas with specific details. "
        "Score 0.5 if it addresses both but one is vague or generic. "
        "Score 0.0 if it only addresses one area and ignores the other entirely."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    threshold=0.7, model="gpt-4o",
)

# ---------------------------------------------------------------------------
# TEST CASES — each spans two policy areas
# ---------------------------------------------------------------------------

QUERIES = [
    (
        "I want to return a clothing item and ship a replacement via express. "
        "What are the return conditions and how fast will the replacement arrive?",
        "Clothing can be returned within 30 days if unworn, unwashed, and with "
        "original tags attached. Express shipping takes 1-2 business days and "
        "costs $15. Orders placed before 2 PM are dispatched same day."
    ),
    (
        "Can I cancel my order and get a refund on the furniture I ordered, "
        "or is it too late?",
        "Orders can be cancelled within 1 hour of placement for a full refund. "
        "After 1 hour, the order cannot be cancelled. Furniture can be returned "
        "within 30 days if unassembled and in original packaging. A 15% "
        "restocking fee applies. Assembled furniture is only returnable if defective."
    ),
    (
        "I'm ordering food items for delivery to Canada. What should I know "
        "about shipping and returns?",
        "International shipping takes 10-21 business days via international "
        "courier. Customs duties and taxes are the buyer's responsibility. "
        "Food and perishable items are non-returnable, but if the order arrives "
        "damaged or spoiled, contact within 24 hours for a full refund or replacement."
    ),
]


def build_cases():
    cases = []
    for query, expected in QUERIES:
        output = rag_support_agent(query)
        print(f"\n=== Q: {query[:60]}... ===")
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
    print("Multi-Doc Synthesis: Combining facts across policy areas")
    print("=" * 60)
    cases = build_cases()

    print("\n--- Synthesis Accuracy ---")
    evaluate(
        test_cases=cases,
        metrics=[synthesis_accuracy],
        async_config=AsyncConfig(run_async=False),
    )

    print("\n--- Cross-Topic Completeness ---")
    # Rebuild cases (fresh agent calls) to avoid state issues
    cases_2 = build_cases()
    evaluate(
        test_cases=cases_2,
        metrics=[completeness],
        async_config=AsyncConfig(run_async=False),
    )
