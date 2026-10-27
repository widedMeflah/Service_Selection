"""
Simulated user (tests and evaluation): a hidden priority and a way of answering.

mode "letter"    answers with the letter of the option that best serves its priority
mode "priority"  states its priority in a free-form sentence (needs an LLM to be interpreted)
mode "free"      describes the chosen option in a free-form sentence (needs an LLM)
confirm          answers "yes" to a confirmation (or "no" for the first `refuse_first` confirmations)
noise            answers to give first, before answering properly (e.g., ["I don't know"])
"""
from __future__ import annotations

import re


class SimulatedUser:
    def __init__(self, priority: str = "cost", mode: str = "letter", refuse_first: int = 0, noise=None):
        self.priority = priority
        self.mode = mode
        self.refuse_first = refuse_first
        self.noise = list(noise or [])
        self.asked = []

    def _options(self, message: str):
        """(letter, monthly cost, availability) of each option; an availability that the provider does not publish
        counts as 0 (the least available)."""
        av = r"with an availability (?:of ([\d.]+)%|that the provider does not publish)"
        out = []
        for pat in (rf"Option ([A-Z])\b[^\n]*?costs ([\d,]+\.\d+) \w+ per month, {av}",
                    rf"Option ([A-Z]): ([\d,]+\.\d+) \w+ per month, {av}"):
            for m in re.finditer(pat, message):
                out.append((m.group(1), float(m.group(2).replace(",", "")), float(m.group(3) or 0.0)))
            if out:
                break
        return out

    def __call__(self, message: str, awaiting: str = "decision") -> str:
        self.asked.append((awaiting, message))
        if self.noise:
            return self.noise.pop(0)
        if awaiting == "confirmation":
            if self.refuse_first > 0:
                self.refuse_first -= 1
                return "no"
            return "yes"
        if self.mode == "priority":
            return f"For me, {self.priority} matters most."
        opts = self._options(message)
        if not opts:
            return "I don't know"
        key = (lambda o: (o[1], -o[2])) if self.priority == "cost" else (lambda o: (-o[2], o[1]))
        best = min(opts, key=key)
        if self.mode == "free":
            return f"I prefer the one at {best[1]:.2f} per month."
        return best[0]


USER_PROMPT = """Role (i)
You play a user who asked a cloud consultant to deploy an application, and who answers the consultant.

Context (ii)
Considering what matters most to you, which you never state: {priority}.
Considering the conversation so far:
{history}
Considering the last message of the consultant:
{message}

Task (iii)
Answer the last message as this user would.

Instructions (iv)
Follow these instructions:
- you need this deployment, and the consultant has already checked every possibility: the options of the message
  are the only ones; when the message proposes options, choose one of them now, the one that best matches what
  matters most to you, in one short sentence written in your own words, even if none is ideal (e.g. a single
  option that raises your budget).
- when the message asks you to confirm a choice, answer yes if it is the option you chose in the conversation,
  otherwise no.
- never state what matters most to you, unless the consultant asks which criterion matters most.

Output format (v)
ANSWER: <your answer>"""

PRIORITY_TEXT = {"cost": "the lowest monthly cost among the proposed options",
                 "availability": "the highest availability among the proposed options"}


def expected_choice(proposal, priority: str):
    """The option that a user of this priority chooses, read from the proposal of the solver (not from the text
    of the message): the cheapest (cost), or the most available (availability), the other criterion breaking
    ties; an availability that is not published counts as 0."""
    if proposal is None or not proposal.options:
        return None
    av = lambda o: o.availability if getattr(o, "availability_known", True) else 0.0
    key = (lambda o: (o.cost_cents, -av(o))) if priority == "cost" else (lambda o: (-av(o), o.cost_cents))
    return min(proposal.options, key=key).letter


class StructuredUser:
    """Deterministic simulated user that reads the proposal of the solver instead of the message, so that its
    choice does not depend on how the message is phrased (by the LLM or by the predefined template)."""

    def __init__(self, priority: str, proposal_fn):
        self.priority, self.proposal_fn = priority, proposal_fn
        self.turns: list[dict] = []

    def __call__(self, message: str, awaiting: str = "decision") -> str:
        answer = "yes" if awaiting == "confirmation" else (expected_choice(self.proposal_fn(), self.priority)
                                                           or "I don't know")
        self.turns.append({"awaiting": awaiting, "answer": answer, "expected": answer})
        return answer


class LLMUser:
    """Simulated user played by an LLM (evaluation sheet), with a hidden priority. Every answer is logged with the
    answer of the deterministic user of the same priority (the expected choice), to measure the errors of the
    simulated user apart from those of the approach."""

    def __init__(self, llm, priority: str = "cost", proposal_fn=None):
        self.llm, self.priority = llm, priority
        self.oracle = SimulatedUser(priority)
        self.proposal_fn = proposal_fn
        self.turns: list[dict] = []
        self.history: list[tuple[str, str]] = []           # (message of the consultant, answer), this case only

    def __call__(self, message: str, awaiting: str = "decision") -> str:
        from ..llm_call import ask_text
        if awaiting == "confirmation":
            expected = "yes"
        elif self.proposal_fn is not None:
            expected = expected_choice(self.proposal_fn(), self.priority) or self.oracle(message, awaiting)
        else:
            expected = self.oracle(message, awaiting)
        try:
            history = "\n".join(f"Consultant: {m}\nYou: {a}" for m, a in self.history) or "(none)"
            out = ask_text(self.llm, USER_PROMPT.format(priority=PRIORITY_TEXT.get(self.priority, self.priority),
                                                        history=history, message=message))
            m = re.search(r"ANSWER:\s*(.*)", out or "", re.S)
            answer = (m.group(1) if m else out or "").strip() or "I don't know"
        except Exception as exc:                     # noqa: BLE001 - logged; the question is asked again
            answer = "I don't know"
            self.turns.append({"error": str(exc)[:200]})
        self.turns.append({"awaiting": awaiting, "answer": answer, "expected": expected})
        self.history.append((message, answer))
        return answer
