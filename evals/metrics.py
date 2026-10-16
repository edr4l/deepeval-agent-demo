"""
metrics.py
==========
Custom deterministic metric to cover a gap in DeepEval's built-in
ToolCorrectnessMetric: it requires both `expected_tools` and `tools_called`
to be non-None, but when an agent genuinely calls zero tools, DeepEval's own
tracing leaves `tools_called` as None rather than []. That makes it
impossible to assert "the correct behaviour here is calling no tool" using
the built-in metric — it just raises MissingTestCaseParamsError.

NoToolCalledMetric does the same comparison ToolCorrectnessMetric would do
for the empty-list case, but treats None as [] instead of erroring on it.
It's deterministic (no judge), so EVAL_JUDGE has no effect on it, same as
ToolCorrectnessMetric.
"""

from deepeval.metrics import BaseMetric


class NoToolCalledMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = None
        self.success = None
        self.reason = None

    def measure(self, test_case, *args, **kwargs) -> float:
        called = getattr(test_case, "tools_called", None) or []
        self.score = 1.0 if len(called) == 0 else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            "No tools were called, as expected."
            if self.success
            else f"Expected no tool call but got: {[t.name for t in called]}"
        )
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "No Tool Called"
