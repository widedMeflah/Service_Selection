"""
No feasible composition (Section 5.1): minimal correction sets (MCS) and their relaxations.

MCS        C subset of G such that X(G \\ C) is not empty and X(G \\ C') is empty for every C' strictly in C.
           Enumerated by increasing size; the enumeration stops at the first size that yields some,
           so that each option relaxes as few constraints as possible. Since every node has at least one
           candidate offer, removing all global constraints restores feasibility: an MCS always exists.
Relaxation of C (Section 3.4)
           its relational constraints are removed, and the bound of each of its QoS constraints is replaced by
           its value in the composition that, without C, optimizes the criteria of C
           (e.g., the minimal budget for a single budget constraint).
           An availability constraint whose targets have no published availability in any composition (offers
           without a single-zone SLA) cannot be met by any bound: it is removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from .cpsat import solve, is_feasible
from .problem import (Problem, Composition, QoSConstraint, RelationalConstraint, COST, AVAILABILITY,
                      value, objectives)


@dataclass
class Change:
    constraint_id: str
    kind: str                        # "bound" | "removed"
    old: Optional[float] = None
    new: Optional[float] = None
    criterion: str = ""
    attribute: str = ""


@dataclass
class RelaxationOption:
    mcs: tuple[str, ...]
    changes: list[Change]
    composition: Composition
    constraints: list = field(default_factory=list)     # G after the relaxation
    cost_cents: int = 0
    availability: float = 0.0


def minimal_correction_sets(problem: Problem) -> list[tuple[str, ...]]:
    ids = [c.id for c in problem.constraints]
    for k in range(1, len(ids) + 1):
        found = []
        for subset in combinations(ids, k):
            rest = [c for c in problem.constraints if c.id not in subset]
            if is_feasible(problem, rest):
                found.append(subset)
        if found:
            return found
    return []


def _round_bound(criterion: str, v: float) -> float:
    if criterion == COST:
        return round(v, 2)
    # truncated (not rounded up) to 4 decimals of a percent, so that the composition still satisfies it
    return int(v * 10_000 + 1e-9) / 10_000


def relax(problem: Problem, mcs: tuple[str, ...]) -> Optional[RelaxationOption]:
    rest = [c for c in problem.constraints if c.id not in mcs]
    in_c = [problem.constraint(cid) for cid in mcs]
    qos = [c for c in in_c if isinstance(c, QoSConstraint)]
    allnodes = tuple(problem.node_names)
    spec = [(c.criterion, tuple(c.targets)) for c in qos] + [(COST, allnodes), (AVAILABILITY, allnodes)]
    x = solve(problem, rest, spec)
    if x is None:
        return None
    changes, new_cons = [], list(rest)
    for c in in_c:
        if isinstance(c, RelationalConstraint):
            changes.append(Change(c.id, "removed", attribute=c.attribute))
        elif value(x, c.criterion, c.targets, problem.groups) is None:
            # no composition publishes the availability of these targets (no single-zone SLA): no bound can be met
            changes.append(Change(c.id, "removed", criterion=c.criterion))
        else:
            v = value(x, c.criterion, c.targets, problem.groups)
            nb = _round_bound(c.criterion, v)
            changes.append(Change(c.id, "bound", c.bound, nb, criterion=c.criterion))
            new_cons.append(QoSConstraint(c.id, c.criterion, c.targets, nb, c.policy))
    order = {c.id: i for i, c in enumerate(problem.constraints)}
    new_cons.sort(key=lambda c: order.get(c.id, 0))
    cost, av = objectives(x, allnodes, problem.groups)
    return RelaxationOption(tuple(mcs), changes, x, new_cons, cost, av)


def relaxation_options(problem: Problem) -> list[RelaxationOption]:
    opts = []
    for mcs in minimal_correction_sets(problem):
        o = relax(problem, mcs)
        if o is not None:
            opts.append(o)
    return opts
