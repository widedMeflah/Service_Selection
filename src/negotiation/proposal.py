"""
Proposal presented to the user (Section 5.2): the alternatives computed by the solver, with their values.
No value is computed by an LLM: the proposal is built from the signal of the selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..selection.mcs import RelaxationOption
from ..selection.problem import (Problem, Composition, QoSConstraint, RelationalConstraint, COST, AVAILABILITY,
                                 objectives, availability)
from ..selection.select import NoFeasible, SeveralOptimal

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class ProposalOption:
    letter: str
    composition: Composition
    cost_cents: int
    availability: float
    relaxation: Optional[RelaxationOption] = None      # set for a relaxation option
    availability_known: bool = True                    # False: an offer publishes no single-zone SLA


@dataclass
class Proposal:
    kind: str                                          # "no_feasible" | "several_optimal"
    options: list[ProposalOption]
    problem: Problem
    currency: str = "USD"
    decimals: int = 2                                  # decimals of the availabilities shown
    notes: list[str] = field(default_factory=list)

    def option(self, letter: str) -> Optional[ProposalOption]:
        for o in self.options:
            if o.letter == letter.upper():
                return o
        return None

    @property
    def letters(self) -> list[str]:
        return [o.letter for o in self.options]

    def allowed_numbers(self) -> list[float]:
        """Every number that a message may contain: values of the options and bounds (old and new)."""
        nums: list[float] = []
        for o in self.options:
            nums += [o.cost_cents / 100.0, round(o.availability, self.decimals), o.availability]
            if o.relaxation:
                for ch in o.relaxation.changes:
                    nums += [v for v in (ch.old, ch.new) if v is not None]
        for c in self.problem.constraints:
            if isinstance(c, QoSConstraint):
                nums.append(c.bound)
        return nums


def _decimals(values: list[float]) -> int:
    """Smallest number of decimals (2 to 4) that distinguishes the availabilities shown."""
    for d in (2, 3, 4):
        r = [round(v, d) for v in values]
        if len(set(r)) == len(set(round(v, 9) for v in values)):
            return d
    return 4


def build_proposal(signal, problem: Problem) -> Proposal:
    nodes = problem.node_names
    if isinstance(signal, NoFeasible):
        opts = [ProposalOption(LETTERS[i], o.composition, o.cost_cents, o.availability, o)
                for i, o in enumerate(signal.options)]
        kind = "no_feasible"
    elif isinstance(signal, SeveralOptimal):
        opts = []
        for i, x in enumerate(signal.compositions):
            c, a = objectives(x, nodes, problem.groups)
            opts.append(ProposalOption(LETTERS[i], x, c, a))
        kind = "several_optimal"
    else:
        raise ValueError("no proposal for a selected composition")
    for o in opts:
        o.availability_known = availability(o.composition, nodes, problem.groups) is not None
    vals = [o.availability for o in opts if o.availability_known] or [0.0]
    return Proposal(kind, opts, problem, problem.currency, _decimals(vals))
