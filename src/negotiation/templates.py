"""
Predefined messages: used without an LLM, and whenever a message phrased by the LLM fails the check
of its values (Section 5.2: "a predefined message is used otherwise").
"""
from __future__ import annotations

from ..selection.problem import QoSConstraint, RelationalConstraint, COST, AVAILABILITY, REGION
from .proposal import Proposal, ProposalOption


def money(v: float, currency: str) -> str:
    return f"{v:,.2f} {currency}"


def pct(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}%"


def bound_pct(v: float) -> str:
    """A required availability, with the decimals it needs (at least 2, at most 4)."""
    for d in (2, 3, 4):
        if abs(round(v, d) - v) < 1e-9:
            return f"{v:.{d}f}%"
    return f"{v:.4f}%"


def _scope(targets) -> str:
    t = list(targets)
    return t[0] if len(t) == 1 else ", ".join(t[:-1]) + " and " + t[-1]


def describe_constraint(c, currency: str, decimals: int = 2) -> str:
    if isinstance(c, RelationalConstraint):
        what = "region" if c.attribute == REGION else "provider"
        return f"the same {what} for {_scope(c.targets)}"
    if c.criterion == COST:
        return f"a monthly budget of {money(c.bound, currency)} for {_scope(c.targets)}"
    return f"an availability of at least {bound_pct(c.bound)} for {_scope(c.targets)}"


def describe_option(o: ProposalOption, p: Proposal) -> str:
    cur, d = p.currency, p.decimals
    av = (f"an availability of {pct(o.availability, d)}" if o.availability_known
          else "an availability that the provider does not publish")
    values = f"{money(o.cost_cents / 100, cur)} per month, with {av}"
    if not o.relaxation:
        return values
    parts = []
    for ch in o.relaxation.changes:
        c = p.problem.constraint(ch.constraint_id)
        if ch.kind == "removed" and ch.criterion == AVAILABILITY:
            parts.append(f"no longer require an availability for {_scope(c.targets)} (no offer publishes one)")
        elif ch.kind == "removed":
            what = "region" if ch.attribute == REGION else "provider"
            parts.append(f"no longer require the same {what} for {_scope(c.targets)}")
        elif ch.criterion == COST:
            parts.append(f"raise the budget for {_scope(c.targets)} from {money(ch.old, cur)} to {money(ch.new, cur)}")
        else:
            parts.append(f"lower the required availability for {_scope(c.targets)} from {bound_pct(ch.old)} to {bound_pct(ch.new)}")
    return "; ".join(parts) + f". The composition then costs {values}"


def option_lines(p: Proposal) -> list[str]:
    return [f"Option {o.letter}: {describe_option(o, p)}." for o in p.options]


def predefined_message(p: Proposal) -> str:
    if p.kind == "no_feasible":
        reqs = "; ".join(describe_constraint(c, p.currency, p.decimals) for c in p.problem.constraints)
        head = f"No composition satisfies all your requirements ({reqs})."
        tail = "Which option do you prefer?"
    else:
        head = "Several compositions satisfy your requirements, and none of them is better on both cost and availability."
        tail = "Which option do you prefer, or which matters most to you, cost or availability?"
    return "\n".join([head] + option_lines(p) + [tail])


def options_context(p: Proposal) -> str:
    """Options given to the LLM (context of the phrasing and interpretation prompts)."""
    return "\n".join(f"{o.letter}: {describe_option(o, p)}" for o in p.options)


def confirmation_message(p: Proposal, o: ProposalOption, priority: str = "") -> str:
    why = f"Since {priority} matters most to you, you choose" if priority else "You choose"
    return f"{why} option {o.letter}: {describe_option(o, p)}. Do you confirm? (yes / no)"
