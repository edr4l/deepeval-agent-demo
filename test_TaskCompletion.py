from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric
from agent_instrumented import support_agent
from evals.test_TaskCompletion import actual_output

support_agent("Where is my order ORD-1042?")

test_case = LLMTestCase(
    input = "Where is my order ORD-1042?",
    actual_output= actual_output
)

# Black box - did it complete? this does not concern itself with internals
evaluate(test_cases = [test_case],
         metrics= [TaskCompletionMetric(threshold=0.7, model= "gpt-4o-mini")])

