"""
LLM calls of the negotiation, with their algorithmic guards (Section 5.2).

phrase()      the LLM presents the options; every value of the message is checked against those computed
              by the solver, every option must be cited and no technical term may appear; otherwise the
              predefined message is used, so that no altered value reaches the user.
interpret()   a short answer ("A", "option b", "2") is read without an LLM; a free-form answer goes to the
              LLM (OPTION / PRIORITY / UNCLEAR), validated against the proposed options.
confirm()     the decision is applied only once the user confirms it (yes / no, read without an LLM).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..llm_call import ask_text
from .prompts import PHRASING_PROMPT, INTERPRETATION_PROMPT, PRIORITY_CONTEXT
from .proposal import Proposal
from .templates import predefined_message, options_context, describe_constraint

TECHNICAL_TERMS = ("correction set", "solver", "non-dominated", "pareto", "constraint", "mcs", "null", "none of the above")
_NUMBER = re.compile(r"(?<![\w.\-])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])")
_YES = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "confirm", "confirmed", "oui", "d'accord", "i confirm"}
_NO = {"no", "n", "nope", "non", "cancel", "do not confirm", "i do not confirm"}


@dataclass
class Interpretation:
    option: Optional[str] = None
    priority: Optional[str] = None
    unclear: bool = False
    source: str = "deterministic"           # deterministic | llm


def numbers_in(text: str) -> list[tuple[float, int]]:
    out = []
    for m in _NUMBER.finditer(text):
        ent = m.group(1).replace(",", "")
        dec = m.group(2) or ""
        out.append((float(ent + ("." + dec if dec else "")), len(dec)))
    return out


def unknown_numbers(text: str, allowed: list[float]) -> list[float]:
    bad = []
    for v, d in numbers_in(text):
        if not any(abs(round(a, d) - v) < 1e-9 for a in allowed):
            bad.append(v)
    return bad


def check_message(msg: str, p: Proposal) -> list[str]:
    """Reasons for rejecting a phrased message (empty list = accepted)."""
    reasons = []
    allowed = p.allowed_numbers() + [float(i + 1) for i in range(len(p.options))]
    bad = unknown_numbers(msg, allowed)
    if bad:
        reasons.append(f"values not computed by the solver: {bad}")
    for letter in p.letters:
        if not re.search(rf"\b{letter}\b", msg):
            reasons.append(f"option {letter} not cited")
    low = msg.lower()
    for t in TECHNICAL_TERMS:
        if t in low:
            reasons.append(f"technical term: {t}")
    return reasons


class Negotiator:
    def __init__(self, llm=None):
        self.llm = llm
        self.rejected: list[dict] = []        # phrased messages rejected by the check (evaluation measure)

    # ------------------------------------------------------------------ phrasing
    def phrase(self, p: Proposal) -> tuple[str, str]:
        """(message, source) where source is 'llm' or 'predefined'."""
        fallback = predefined_message(p)
        if self.llm is None:
            return fallback, "predefined"
        if p.kind == "no_feasible":
            reqs = "; ".join(describe_constraint(c, p.currency, p.decimals) for c in p.problem.constraints)
            situation = f"No composition satisfies all the requirements of the user ({reqs})."
            pq = ""
        else:
            situation = "Several compositions satisfy the requirements, and none is better on both cost and availability."
            pq = ", or which matters most, cost or availability"
        prompt = PHRASING_PROMPT.format(situation=situation, options=options_context(p), priority_question=pq)
        try:
            out = ask_text(self.llm, prompt)
        except Exception as exc:                      # noqa: BLE001 - any LLM error falls back
            self.rejected.append({"reason": f"LLM error: {exc}"})
            return fallback, "predefined"
        m = re.search(r"MESSAGE:\s*(.*)", out or "", re.S)
        msg = (m.group(1) if m else (out or "")).strip()
        reasons = check_message(msg, p) if msg else ["empty message"]
        if reasons:
            self.rejected.append({"message": msg, "reasons": reasons})
            return fallback, "predefined"
        return msg, "llm"

    # ------------------------------------------------------------------ interpretation
    def interpret(self, answer: str, p: Proposal) -> Interpretation:
        a = (answer or "").strip()
        m = re.fullmatch(r"(?i)\s*(?:option\s*)?([A-Z])\s*[.!]?\s*", a)
        if m and m.group(1).upper() in p.letters:
            return Interpretation(option=m.group(1).upper())
        m = re.fullmatch(r"\s*(\d+)\s*", a)
        if m and 1 <= int(m.group(1)) <= len(p.options):
            return Interpretation(option=p.letters[int(m.group(1)) - 1])
        if self.llm is None:
            return Interpretation(unclear=True)
        prompt = INTERPRETATION_PROMPT.format(options=options_context(p), priority_context=PRIORITY_CONTEXT,
                                              user_answer=a)
        try:
            out = ask_text(self.llm, prompt) or ""
        except Exception:                             # noqa: BLE001
            return Interpretation(unclear=True, source="llm")
        opt = _field(out, "OPTION")
        pri = _field(out, "PRIORITY")
        unclear = _field(out, "UNCLEAR").lower() == "true"
        opt = opt.upper() if opt and opt.lower() != "none" else None
        pri = pri.lower() if pri and pri.lower() in ("cost", "availability") else None
        if unclear or (opt is None) == (pri is None) or (opt is not None and opt not in p.letters):
            return Interpretation(unclear=True, source="llm")
        return Interpretation(option=opt, priority=pri, source="llm")

    # ------------------------------------------------------------------ confirmation
    @staticmethod
    def confirm(answer: str) -> Optional[bool]:
        """yes / no, from the whole answer or from its first words ("Yes, that is the option I chose."); None when
        the answer is neither, or both."""
        a = re.sub(r"[.!]+$", "", (answer or "").strip().lower())
        if a in _YES:
            return True
        if a in _NO:
            return False
        first = re.sub(r"^[^\w']+", "", a)
        yes = any(re.match(rf"{re.escape(w)}\b", first) for w in _YES)
        no = any(re.match(rf"{re.escape(w)}\b", first) for w in _NO)
        if yes != no:
            return yes
        return None


def _field(text: str, name: str) -> str:
    m = re.search(rf"{name}\s*:\s*([^\n]*)", text, re.I)
    return m.group(1).strip().strip("<>").strip() if m else ""
