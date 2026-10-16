"""
judges.py
=========
Lets you swap the LLM judge used by DeepEval metrics (TaskCompletionMetric,
GEval, etc.) between a hosted model (gpt-4o) and a local Ollama model,
via one env var — no code changes needed per run.

    EVAL_JUDGE=gpt-4o            python evals/test_tool_agent_csv.py   # default
    EVAL_JUDGE=ollama:llama3.1   python evals/test_tool_agent_csv.py
    EVAL_JUDGE=ollama:mistral    python evals/test_tool_agent_csv.py

Note this only affects metrics that take a `model=` argument (an LLM judge).
ToolCorrectnessMetric has no judge — it's a deterministic comparison of tool
names — so it's unaffected by this either way.

Assumes `ollama serve` is running locally with the model already pulled
(`ollama pull llama3.1`). Ollama exposes an OpenAI-compatible endpoint at
http://localhost:11434/v1, which is what this wraps.
"""

import os

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI


class OllamaJudge(DeepEvalBaseLLM):
    """Minimal DeepEvalBaseLLM wrapper around a local Ollama model."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434/v1"):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key="ollama")  # api_key is unused by Ollama, just needs to be set
        super().__init__(model_name)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        # Simplest safe path: no async Ollama call here, just run the sync one.
        # Fine for eval runs (not latency-sensitive); revisit if this becomes
        # a bottleneck on a large golden set.
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"Ollama ({self.model_name})"


def get_judge():
    """
    Returns whatever `model=` value DeepEval metrics should use, read from
    EVAL_JUDGE. Returns a plain string for hosted models (DeepEval resolves
    those itself) or an OllamaJudge instance for "ollama:<model>".
    """
    spec = os.environ.get("EVAL_JUDGE", "gpt-4o")
    if spec.startswith("ollama:"):
        return OllamaJudge(model_name=spec.split("ollama:", 1)[1])
    return spec
