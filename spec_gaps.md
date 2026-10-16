# Spec Gaps

Real product/business decisions or missing guardrails surfaced while writing goldens
for `deepeval-agent-demo`. These are NOT encoded into golden `expected_output`
values — where a gap exists, the golden instead asserts that the agent must not
resolve the ambiguity on its own (see `evals/goldens/shopease_tool_agent.csv`).

Two kinds of gap, tracked separately because they need different owners:

- **Business decision** — nobody has decided the behavior yet; needs a product/policy call.
- **Missing guardrail** — the system should already meet a baseline (security, safety)
  regardless of whether anyone wrote it down; the absence itself is the bug.

---

## Open

### 1. Standard shipping cost below $50 — undefined
**Kind:** Business decision
**Where:** `mcp_server.py`, `get_shipping_options()` — returns "Standard (3-5 days,
free on orders over $50), Express..." with no price stated for orders under $50.
**Impact:** Agent cannot answer "how much is Standard shipping on a $30 order?"
without either refusing or inventing a number. Goldens currently forbid the agent
from stating an unspecified price.

### 2. No topic-scoping instruction in `agent_instrumented.py`
**Kind:** Missing guardrail (borderline business decision)
**Where:** `agent_instrumented.py`, `system_prompt=` — no "stay on topic" /
"decline unrelated requests" instruction, unlike `chatbot.py`'s `SYSTEM_PROMPT`
which explicitly states "Never discuss topics outside of ShopEasy customer support."
**Impact:** Two implementations of the same product have different, undocumented
behavior for off-topic requests (e.g. "what's the weather"). Adversarial goldens
against `agent_instrumented.py` need to be re-examined against this — we cannot
assert topic-refusal as a hard requirement for this file until the prompt is
updated or the behavior is confirmed intentional.

### 3. "No refund policy found" gives no next step
**Kind:** Business decision
**Where:** `agent_instrumented.py`, `get_refund_policy()` — not-found message is
exactly `"No refund policy on file for '{category}'."`, no escalation path.
**Impact:** Dead-end response for the end user (e.g. furniture query). Worth
deciding whether the agent should offer to route to customer service.

### 4. Order status values have no defined semantics
**Kind:** Business decision
**Where:** `agent_instrumented.py`, `ORDERS` dict — statuses are bare strings
("Processing", "Shipped", "Delivered") with no lifecycle/state-machine definition
anywhere in the code.
**Impact:** Cannot determine, e.g., whether an order in "Processing" is still
eligible to have its shipping method changed. Golden for the ORD-7777 +
express-shipping case explicitly forbids the agent from asserting eligibility
either way, pending a real answer.

### 5. No resistance to system-prompt disclosure / instruction override
**Kind:** Missing guardrail
**Where:** `agent_instrumented.py`, `system_prompt=` — contains no instruction
to withhold itself or resist "ignore previous instructions" framing.
**Impact:** Unlike gaps 1, 3, 4, this isn't a pending business decision — it's a
named risk category (system prompt leakage / excessive agency, OWASP Top 10 for
Agentic Applications) that any deployed agent should defend against regardless of
whether its author thought to write the instruction. Currently relying entirely
on the base model's own training, with no explicit policy in the prompt.
Golden still asserts the requirement (baseline-grounded, not spec-grounded) but
this is flagged as higher priority than the business-decision gaps above.

### 6. Return-window duration has no start-date reference point
**Kind:** Business decision
**Where:** `agent_instrumented.py`, `REFUND_POLICIES` — clothing's entry is
"Clothing can be returned within 30 days with tags attached." No policy text
(electronics, clothing, food) states what date the window counts from —
purchase, delivery, or order date are all equally plausible reads.
**Impact:** Customer-facing ambiguity — a user told only "within 30 days" has
no way to know their actual deadline. The system already tracks a delivery
date per order (`ORDERS[...]["eta"]`), so delivery date is the most likely
candidate if a reference point is added. Recommend the policy text (or the
tool's return value) be updated to explicitly name the reference date. Goldens
currently test today's actual (ambiguous) behavior and forbid the agent from
computing or implying a deadline; they'll need rewriting once this is decided.

### 7. `rag_agent.py` and `policies.txt` contradict each other on real facts
**Kind:** Missing guardrail / data integrity
**Where:** `rag_agent.py`'s `POLICY_DOCS` (the actual, live knowledge base
embedded into its vector store) vs. `policies.txt` (used only by
`evals/test_agent_synthesized.py` to synthesize goldens — not read by any
agent at runtime). These two documents disagree on:
- Express shipping cost: $15 (`POLICY_DOCS`) vs. $12.99 (`policies.txt`,
  and also `mcp_server.py`/`agent_instrumented.py`'s target)
- Electronics/clothing return-window reference date: "of delivery"
  (`POLICY_DOCS`) vs. "of purchase" (`policies.txt`)
- Damaged/defective items: general 48-hour policy in `policies.txt`, but
  `POLICY_DOCS` has no general equivalent — only a food-specific 24-hour
  clause for spoiled/damaged food
- Order cancellation: 1-hour window exists in `POLICY_DOCS`, absent from
  `policies.txt` entirely
**Impact:** Serious — this isn't an undefined gap, it's two "ground truth"
documents actively disagreeing. Any golden or synthesized test case sourced
from `policies.txt` risks being validated against facts `rag_agent.py`
cannot actually retrieve, and vice versa. Real product risk too: if a human
maintainer edits one document assuming it's the source of truth, the
divergence gets worse. Needs a decision on which document is canonical, and
the other either deleted, regenerated from it, or explicitly marked as
"synthesis seed only, not runtime truth."

### 8. A previous golden in this project asserted a fact from `policies.txt`
that `rag_agent.py` cannot actually retrieve
**Kind:** Process note, not a product gap
**Where:** `evals/goldens/shopease_tool_agent.csv`, the "It's damaged what
happens now?" row (against `agent_instrumented.py`) originally justified its
negative clause by claiming the 48-hour damaged-item policy was "reachable
by rag_agent.py's search_policies tool." That's false — `rag_agent.py` never
reads `policies.txt`; its real knowledge base is `POLICY_DOCS`, which has no
matching general policy. The golden's core assertion (this agent must not
describe a damage-claims process) is still correct, but the justification
needs correcting: this agent has no such tool, full stop — not "the tool
exists elsewhere." Logged as a reminder to verify runtime data sources
directly (grep for what actually reads a file) rather than assuming a
plausible-looking document is live.

---

## Resolved
_(none yet)_
