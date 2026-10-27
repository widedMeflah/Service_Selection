"""Exhaustive oracle of the selection (tests): enumerates every composition, exact evaluation."""
from __future__ import annotations

from itertools import product

from ..selection.problem import Problem, feasible, objectives, dominates, COST, AVAILABILITY


def all_compositions(problem: Problem, constraints=None):
    cons = problem.constraints if constraints is None else constraints
    names = problem.node_names
    for combo in product(*[problem.node(n).candidates for n in names]):
        x = dict(zip(names, combo))
        if feasible(x, cons, problem.groups):
            yield x


def front_points(problem: Problem, constraints=None) -> set:
    pts = {objectives(x, problem.node_names, problem.groups) for x in all_compositions(problem, constraints)}
    return {p for p in pts if not any(dominates(q, p) for q in pts if q != p)}


def is_feasible(problem: Problem, constraints) -> bool:
    return next(all_compositions(problem, constraints), None) is not None


def best(problem: Problem, constraints, spec):
    """Lexicographic optimum value vector for spec [(criterion, targets)]."""
    from ..selection.problem import value
    best_v = None
    for x in all_compositions(problem, constraints):
        v = []
        for crit, t in spec:
            val = value(x, crit, t, problem.groups)
            val = 0.0 if val is None else val
            v.append(round(val, 9) if crit == COST else -round(val, 9))
        v = tuple(v)
        if best_v is None or v < best_v:
            best_v = v
    return best_v
