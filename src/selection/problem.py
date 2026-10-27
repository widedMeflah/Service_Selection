"""
Selection problem (Section 3.4 of the paper).

N        nodes of the template that represent cloud services, each with its candidate offers O_n (non-empty)
Q        QoS criteria: cost (aggregated by sum, minimized) and availability (maximized), aggregated by product
         over the blocks of the targets: a node alone is a block in series; a redundancy group R (replicas of one
         another, e.g. a primary and its standby) is a block in parallel, available when one replica is:
         A(R) = 1 - prod_{n in R} (1 - a_n)
A        categorical attributes of the offers: provider and region
R        redundancy groups: disjoint sets of at least two nodes (TOSCA policy Redundancy)
G        global constraints, each targeting at least two nodes:
           QoS constraint        (q, S, b):  F_q(x, S) <= b (cost) or >= b (availability)
           relational constraint (a, S):     the offers of S have the same value of a (provider or region)
Single-node constraints are enforced by discovery and are not part of G.

The module calls no solver: it gives the exact evaluation of a composition, shared by the solver
(exact post-check) and by the exhaustive oracle of the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

COST = "cost"
AVAILABILITY = "availability"
CRITERIA = (COST, AVAILABILITY)
REGION = "region"
PROVIDER = "provider"
ATTRIBUTES = (REGION, PROVIDER)
EPS = 1e-9


@dataclass(frozen=True)
class Offer:
    offer_id: str
    provider: str
    region: str
    price_cents: int                     # monthly price, in cents of the currency of the problem
    availability: Optional[float]        # in %, e.g. 99.95; None if not published
    name: str = ""
    location: str = ""                   # logical location of the region (e.g. "uk"), compares providers
    service: str = ""                    # service of the offer (e.g. "compute", "database_postgresql")

    def attribute(self, a: str) -> str:
        if a == PROVIDER:
            return self.provider
        return self.location or self.region


@dataclass
class Node:
    name: str
    candidates: list[Offer]


@dataclass
class QoSConstraint:
    id: str
    criterion: str                       # cost | availability
    targets: tuple[str, ...]
    bound: float                         # currency units (cost) or % (availability)
    policy: str = ""                     # name of the TOSCA policy it comes from

    @property
    def kind(self) -> str:
        return "qos"


@dataclass
class RelationalConstraint:
    id: str
    attribute: str                       # region | provider
    targets: tuple[str, ...]
    policy: str = ""

    @property
    def kind(self) -> str:
        return "relational"


Constraint = Union[QoSConstraint, RelationalConstraint]
Composition = dict                       # node name -> Offer


@dataclass
class Problem:
    nodes: list[Node]
    constraints: list[Constraint] = field(default_factory=list)
    currency: str = "USD"
    groups: list[tuple[str, ...]] = field(default_factory=list)      # redundancy groups (disjoint)

    def node(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    @property
    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]

    def constraint(self, cid: str) -> Constraint:
        for c in self.constraints:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def with_constraints(self, constraints: list[Constraint]) -> "Problem":
        return Problem(self.nodes, list(constraints), self.currency, self.groups)


# ----------------------------------------------------------------------------------------------
# exact evaluation
# ----------------------------------------------------------------------------------------------
def cost_cents(x: Composition, targets) -> int:
    return sum(x[n].price_cents for n in targets)


def blocks(targets, groups=()) -> list[tuple[str, ...]]:
    """The targets split into blocks: the members of a redundancy group among the targets form one block (when at
    least two of them are targeted), every other target is a block of its own."""
    targets = list(targets)
    out, done = [], set()
    for g in groups:
        m = tuple(n for n in g if n in targets)
        if len(m) >= 2:
            out.append(m)
            done.update(m)
    out += [(n,) for n in targets if n not in done]
    return out


def block_availability(avails) -> Optional[float]:
    """Availability (fraction) of a block whose members have the given availabilities in % (parallel when several)."""
    q = 1.0
    for a in avails:
        if a is None:
            return None
        q *= 1.0 - a / 100.0
    return 1.0 - q


def availability(x: Composition, targets, groups=()) -> Optional[float]:
    """End-to-end availability in %: product over the blocks (independent failures), a redundancy group in parallel."""
    p = 1.0
    for b in blocks(targets, groups):
        a = block_availability([x[n].availability for n in b])
        if a is None:
            return None
        p *= a
    return p * 100.0


def value(x: Composition, criterion: str, targets, groups=()) -> Optional[float]:
    if criterion == COST:
        return cost_cents(x, targets) / 100.0
    return availability(x, targets, groups)


def satisfies(x: Composition, c: Constraint, groups=()) -> bool:
    if isinstance(c, RelationalConstraint):
        return len({x[n].attribute(c.attribute) for n in c.targets}) <= 1
    if c.criterion == COST:
        return cost_cents(x, c.targets) <= round(c.bound * 100) + 0
    a = availability(x, c.targets, groups)
    return a is not None and a >= c.bound - EPS


def feasible(x: Composition, constraints, groups=()) -> bool:
    return all(satisfies(x, c, groups) for c in constraints)


def objectives(x: Composition, nodes, groups=()) -> tuple[int, float]:
    """(total cost in cents, end-to-end availability in %, 0 if unknown)."""
    a = availability(x, nodes, groups)
    return cost_cents(x, nodes), (a if a is not None else 0.0)


def dominates(p: tuple[int, float], q: tuple[int, float]) -> bool:
    """p dominates q: no worse on both criteria, strictly better on one (cost min, availability max)."""
    no_worse = p[0] <= q[0] and p[1] >= q[1] - EPS
    better = p[0] < q[0] or p[1] > q[1] + EPS
    return no_worse and better
