"""
Constraint-Based Selection (Section 5.1): returns one of three signals.

  Selected        a single non-dominated composition (or the user's priority is known)
  NoFeasible      X(G) is empty: priced relaxation options, one per minimal correction set
  SeveralOptimal  |P(G)| > 1 and the user's priority is unknown: the non-dominated compositions
                  (priced trade-off), each optimal for a different priority
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .cpsat import solve, pareto_front
from .mcs import RelaxationOption, relaxation_options
from .problem import Problem, Composition, COST, AVAILABILITY, CRITERIA, objectives


@dataclass
class Selected:
    composition: Composition
    cost_cents: int
    availability: float
    kind: str = "selected"


@dataclass
class NoFeasible:
    options: list[RelaxationOption]
    kind: str = "no_feasible"


@dataclass
class SeveralOptimal:
    compositions: list[Composition]
    points: list[tuple[int, float]] = field(default_factory=list)
    kind: str = "several_optimal"


Signal = Union[Selected, NoFeasible, SeveralOptimal]


def priority_spec(problem: Problem, priority: list[str]):
    allnodes = tuple(problem.node_names)
    order = [c for c in priority if c in CRITERIA] + [c for c in CRITERIA if c not in priority]
    return [(c, allnodes) for c in order]


def _selected(problem: Problem, x: Composition) -> Selected:
    c, a = objectives(x, problem.node_names, problem.groups)
    return Selected(x, c, a)


def select(problem: Problem, priority: Optional[list[str]] = None) -> Signal:
    if solve(problem, problem.constraints, [(COST, tuple(problem.node_names))]) is None:
        return NoFeasible(relaxation_options(problem))
    front = pareto_front(problem)
    if len(front) == 1:
        return _selected(problem, front[0])
    if priority:
        x = solve(problem, problem.constraints, priority_spec(problem, priority))
        return _selected(problem, x)
    front.sort(key=lambda x: objectives(x, problem.node_names, problem.groups)[0])
    return SeveralOptimal(front, [objectives(x, problem.node_names, problem.groups) for x in front])
