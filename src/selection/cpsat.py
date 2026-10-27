"""
CP-SAT model of the joint selection (Section 5.1).

Variables     y[n, o] = 1 if node n takes candidate offer o (exactly one per node)
Constraints   cost:          sum of prices (integers, in cents) over S <= bound
              availability:  product over the blocks of S >= bound  <=>  sum of scaled logarithms >= log(bound);
                             a node alone contributes log(a_n) through y; a redundancy group (block in parallel)
                             contributes log(1 - prod (1 - a_n)) through one Boolean per combination of the offers
                             of its members (z = AND of their y)
              relational:    the offers of S have the same provider / the same (logical) region
Objectives    lexicographic list of (criterion, targets), then a canonical tie-break (offer rank)

The logarithms are rounded; every solution is therefore checked with the exact evaluation of
problem.py, and an exactly infeasible solution is excluded (no-good) before solving again, so that
every returned composition satisfies the constraints exactly.
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable, Optional

from ortools.sat.python import cp_model

from .problem import (Problem, Composition, QoSConstraint, RelationalConstraint, COST, AVAILABILITY,
                      feasible as exact_feasible, objectives, dominates, blocks, block_availability)

SCALE = 10 ** 9
MISSING_LOG = -60 * SCALE        # an offer without published availability counts as (almost) 0 %
TIME_LIMIT_S = 30.0
MAX_NOGOODS = 200
MAX_GROUP_COMBINATIONS = 250_000


def _log(a_percent: Optional[float]) -> int:
    if a_percent is None or a_percent <= 0:
        return MISSING_LOG
    return int(round(math.log(a_percent / 100.0) * SCALE))


def _log_fraction(a: Optional[float]) -> int:
    if a is None or a <= 0:
        return MISSING_LOG
    return int(round(math.log(a) * SCALE))


def log_value(x: Composition, nodes, groups=()) -> int:
    """Scaled logarithm of the availability of a composition, as in the model (block by block)."""
    return sum(_log_fraction(block_availability([x[n].availability for n in b])) for b in blocks(nodes, groups))


class _Model:
    def __init__(self, problem: Problem, constraints: list, nogoods: list[Composition]):
        self.p = problem
        m = self.m = cp_model.CpModel()
        self.y: dict[tuple[str, int], object] = {}
        self._rank: list = []
        self._z: dict[tuple[str, ...], list] = {}          # redundancy block -> [(z, scaled log)]
        avail_needed = {n for c in constraints if isinstance(c, QoSConstraint) and c.criterion == AVAILABILITY
                        for n in c.targets}
        for node in problem.nodes:
            ranked = sorted(range(len(node.candidates)),
                            key=lambda i: (node.candidates[i].price_cents, node.candidates[i].offer_id))
            vs = []
            for i, o in enumerate(node.candidates):
                v = m.NewBoolVar(f"y[{node.name},{i}]")
                self.y[(node.name, i)] = v
                if node.name in avail_needed and o.availability is None:
                    m.Add(v == 0)
                vs.append(v)
            m.AddExactlyOne(vs)
            for r, i in enumerate(ranked):
                self._rank.append(r * self.y[(node.name, i)])
        for c in constraints:
            if isinstance(c, QoSConstraint):
                if c.criterion == COST:
                    m.Add(self.cost_expr(c.targets) <= int(round(c.bound * 100)))
                else:
                    m.Add(self.log_expr(c.targets) >= _log(c.bound) - len(c.targets))
            elif isinstance(c, RelationalConstraint):
                self._same(c)
        for x in nogoods:
            m.AddBoolOr([self.y[(n, self._index(n, x[n]))].Not() for n in x])

    def _index(self, node: str, offer) -> int:
        cands = self.p.node(node).candidates
        for i, o in enumerate(cands):
            if o is offer or o.offer_id == offer.offer_id:
                return i
        raise KeyError(offer.offer_id)

    def cost_expr(self, targets: Iterable[str]):
        return sum(o.price_cents * self.y[(n, i)] for n in targets
                   for i, o in enumerate(self.p.node(n).candidates))

    def _block_terms(self, block: tuple[str, ...]) -> list:
        if block not in self._z:
            cands = [list(enumerate(self.p.node(n).candidates)) for n in block]
            if math.prod(len(c) for c in cands) > MAX_GROUP_COMBINATIONS:
                raise ValueError(f"redundancy group {block}: too many combinations of offers")
            terms = []
            for combo in itertools.product(*cands):
                z = self.m.NewBoolVar(f"z[{','.join(block)}:{','.join(str(i) for i, _ in combo)}]")
                ys = [self.y[(n, i)] for n, (i, _) in zip(block, combo)]
                for v in ys:
                    self.m.AddImplication(z, v)
                self.m.AddBoolOr([z] + [v.Not() for v in ys])
                terms.append((z, _log_fraction(block_availability([o.availability for _, o in combo]))))
            self._z[block] = terms
        return self._z[block]

    def log_expr(self, targets: Iterable[str]):
        out = []
        for b in blocks(targets, self.p.groups):
            if len(b) == 1:
                n = b[0]
                out += [_log(o.availability) * self.y[(n, i)] for i, o in enumerate(self.p.node(n).candidates)]
            else:
                out += [w * z for z, w in self._block_terms(b)]
        return sum(out)

    def _same(self, c: RelationalConstraint):
        nodes = list(c.targets)
        values = {o.attribute(c.attribute) for n in nodes for o in self.p.node(n).candidates}
        first = nodes[0]
        for v in values:
            ind0 = sum(self.y[(first, i)] for i, o in enumerate(self.p.node(first).candidates)
                       if o.attribute(c.attribute) == v)
            for n in nodes[1:]:
                ind = sum(self.y[(n, i)] for i, o in enumerate(self.p.node(n).candidates)
                          if o.attribute(c.attribute) == v)
                self.m.Add(ind == ind0)

    def _solver(self):
        s = cp_model.CpSolver()
        s.parameters.num_search_workers = 1
        s.parameters.random_seed = 0
        s.parameters.max_time_in_seconds = TIME_LIMIT_S
        return s

    def solve(self, objectives_spec: list[tuple[str, tuple[str, ...]]], extra: list = ()) -> Optional[Composition]:
        """objectives_spec: [(criterion, targets)], cost minimized, availability maximized; then rank."""
        m = self.m
        for e in extra:
            m.Add(e)
        solver = self._solver()
        exprs = []
        for crit, targets in objectives_spec:
            if crit == COST:
                exprs.append((self.cost_expr(targets), False))
            else:
                exprs.append((self.log_expr(targets), True))
        exprs.append((sum(self._rank), False))
        for expr, maximize in exprs:
            m.ClearObjective()
            (m.Maximize if maximize else m.Minimize)(expr)
            st = solver.Solve(m)
            if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return None
            m.Add(expr == int(round(solver.ObjectiveValue())))
        st = solver.Solve(m)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        return {n: self.p.node(n).candidates[i] for (n, i), v in self.y.items() if solver.Value(v)}

    def is_feasible(self) -> bool:
        return self._solver().Solve(self.m) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def default_order(problem: Problem) -> list[tuple[str, tuple[str, ...]]]:
    allnodes = tuple(problem.node_names)
    return [(COST, allnodes), (AVAILABILITY, allnodes)]


def solve(problem: Problem, constraints: Optional[list] = None,
          objectives_spec: Optional[list[tuple[str, tuple[str, ...]]]] = None,
          extra_builder=None) -> Optional[Composition]:
    """Lexicographic optimum under `constraints` (default: all those of the problem), exact."""
    cons = problem.constraints if constraints is None else constraints
    spec = objectives_spec or default_order(problem)
    nogoods: list[Composition] = []
    while len(nogoods) <= MAX_NOGOODS:
        model = _Model(problem, cons, nogoods)
        extra = extra_builder(model) if extra_builder else []
        x = model.solve(spec, extra)
        if x is None:
            return None
        if exact_feasible(x, cons, problem.groups):
            return x
        nogoods.append(x)
    return None


def is_feasible(problem: Problem, constraints: list) -> bool:
    return solve(problem, constraints, [(COST, tuple(problem.node_names))]) is not None


def pareto_front(problem: Problem, constraints: Optional[list] = None, max_points: int = 20) -> list[Composition]:
    """Non-dominated compositions P(G) for (cost min, availability max) over all nodes, by successive
    solver calls in which cost is minimized while availability is bounded from below (epsilon-constraint)."""
    cons = problem.constraints if constraints is None else constraints
    allnodes = tuple(problem.node_names)
    found: list[Composition] = []
    floor: Optional[int] = None
    while len(found) < max_points:
        fl = floor
        x = solve(problem, cons, [(COST, allnodes), (AVAILABILITY, allnodes)],
                  extra_builder=(None if fl is None else (lambda mdl, fl=fl: [mdl.log_expr(allnodes) >= fl + 1])))
        if x is None:
            break
        found.append(x)
        floor = log_value(x, allnodes, problem.groups)
    pts = [objectives(x, allnodes, problem.groups) for x in found]
    front = []
    for i, x in enumerate(found):
        if any(dominates(pts[j], pts[i]) for j in range(len(found)) if j != i):
            continue
        if any(pts[j] == pts[i] for j in range(i)):
            continue
        front.append(x)
    return front
