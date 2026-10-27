"""
Builds the selection problem from the input of the approach: the provider-independent TOSCA template
and the output of discovery.

Discovery output, two accepted forms
  - the output of discovery: {"nodes": [{"node_name", "applied_filters": {"sla_field_used"},
    "candidates": [{offer_id, offer_name, provider, region, price_month, sla, sla_single_zone, sla_multi_zone,
    offer_type, engine, ...}]}]}; the field named by sla_field_used gives the availability ("sla" in the
    evaluation dataset; "sla_single_zone" or "sla_multi_zone" in the output of our previous work)
  - a simple form: {node: [{provider, offer, region, monthly_price, availability}]}

Policies (recognized by the last segment of their type)
  Cost          max_cost | max_total_monthly_price | max_monthly_price | budget      -> QoS constraint on cost
  Availability  availability | min_end_to_end | min_availability                   -> QoS constraint on availability
  Colocation    scope: region | provider                                           -> relational constraint
  Redundancy    targets are replicas of one another                                -> redundancy group (parallel)
A policy is a global constraint when it targets at least two service nodes (no targets = all of them).
A policy targeting a single node is enforced by discovery and is not part of G.

Availability of an offer: the SLA of the configuration whose price is listed (field "sla" of the discovery
output): the single-zone SLA, except a Cloud SQL offer, listed by discovery in its high-availability configuration
(price x2) since only that configuration is covered by an SLA (src/eval/discovery.py, priced_configuration).
No fallback from one SLA field to another: an offer without a value in the field used has an unknown availability
(None), counted as 0 % in the objectives and excluded from the nodes targeted by an availability constraint.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ..common.tosca import iter_policies, policy_kind, policy_targets
from .problem import (Offer, Node, Problem, QoSConstraint, RelationalConstraint, COST, AVAILABILITY,
                      REGION, PROVIDER)

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

COST_KEYS = ("max_cost", "max_total_monthly_price", "max_monthly_price", "budget", "max_cost_month")
AVAIL_KEYS = ("availability", "min_end_to_end", "min_availability", "min_sla")
PRICE_KEYS = ("price_month", "monthly_price", "estimated_price_month")


class InputError(ValueError):
    """The input violates a guarantee of discovery (e.g. a node without any candidate offer)."""


def load_taxonomy(path: Optional[str] = None) -> dict[str, list[str]]:
    path = path or os.path.join(DATA, "region_taxonomy.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def logical_location(region: str, taxonomy: dict[str, list[str]]) -> str:
    """Smallest logical location of the taxonomy containing the region (e.g. eu-west-2 and uksouth -> uk)."""
    best = None
    for loc, regions in taxonomy.items():
        if region in regions and (best is None or len(regions) < len(taxonomy[best])):
            best = loc
    return best or region


def _service_of(c: dict) -> str:
    kind = str(c.get("offer_type") or c.get("service_type") or "").lower()
    engine = str(c.get("engine") or "").lower()
    return f"{kind}_{engine}" if engine and kind else (kind or "service")


def _availability(c: dict, sla_field: str) -> Optional[float]:
    if c.get("availability") is not None:
        return float(c["availability"])
    if sla_field == "sla":
        v = c.get("sla")
    else:
        v = c.get("sla_multi_zone") if sla_field.startswith("sla_multi") else c.get("sla_single_zone")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):            # e.g. "—": not published
        return None


def _price(c: dict) -> float:
    for k in PRICE_KEYS:
        if c.get(k) is not None:
            return float(c[k])
    raise InputError(f"offer without a monthly price: {c.get('offer_id') or c}")


def _offer(c: dict, sla_field: str, taxonomy: dict) -> Offer:
    region = str(c.get("region") or "")
    provider = str(c.get("provider") or "").lower()
    name = str(c.get("offer_name") or c.get("offer") or c.get("offer_id") or "")
    oid = str(c.get("offer_id") or f"{provider}-{'-'.join(name.split())}@{region}")
    return Offer(offer_id=oid, provider=provider, region=region,
                 price_cents=int(round(_price(c) * 100)), availability=_availability(c, sla_field),
                 name=name, location=logical_location(region, taxonomy), service=_service_of(c))


def nodes_from_discovery(discovery: dict, taxonomy: dict) -> list[Node]:
    nodes: list[Node] = []
    if isinstance(discovery.get("nodes"), list):
        for n in discovery["nodes"]:
            field = str((n.get("applied_filters") or {}).get("sla_field_used") or "sla_single")
            cands = [_offer(c, field, taxonomy) for c in n.get("candidates") or []]
            nodes.append(Node(str(n["node_name"]), cands))
    else:
        for name, cands in discovery.items():
            if isinstance(cands, list):
                nodes.append(Node(str(name), [_offer(c, "sla_single", taxonomy) for c in cands]))
    for n in nodes:
        if not n.candidates:
            raise InputError(f"node {n.name} has no candidate offer (the discovery step guarantees at least one)")
    return nodes


def _first(props: dict, keys) -> Optional[float]:
    for k in keys:
        if props.get(k) is not None:
            return float(props[k])
    return None


def constraints_from_template(tosca: dict, service_nodes: list[str]) -> list:
    sn = set(service_nodes)
    out = []
    for name, body in iter_policies(tosca):
        kind = policy_kind(body.get("type"))
        props = body.get("properties") or {}
        targets = tuple(policy_targets(tosca, body, sn))
        if len(targets) < 2:
            continue                                      # single-node: enforced by discovery
        if kind == "cost":
            b = _first(props, COST_KEYS)
            if b is not None:
                out.append(QoSConstraint(name, COST, targets, b, name))
        elif kind == "availability":
            b = _first(props, AVAIL_KEYS)
            if b is not None:
                out.append(QoSConstraint(name, AVAILABILITY, targets, b, name))
        elif kind == "colocation":
            scope = str(props.get("scope") or "").lower()
            if scope not in (REGION, PROVIDER):
                raise InputError(f"policy {name}: scope must be 'region' or 'provider'")
            out.append(RelationalConstraint(name, scope, targets, name))
    return out


def groups_from_template(tosca: dict, service_nodes: list[str]) -> list[tuple[str, ...]]:
    """Redundancy groups: the service nodes targeted by each Redundancy policy (at least two, disjoint groups)."""
    sn = set(service_nodes)
    out: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for name, body in iter_policies(tosca):
        if policy_kind(body.get("type")) != "redundancy":
            continue
        targets = tuple(policy_targets(tosca, body, sn))
        if len(targets) < 2:
            continue
        if seen & set(targets):
            raise InputError(f"policy {name}: a node belongs to two redundancy groups")
        seen.update(targets)
        out.append(targets)
    return out


def build_problem(tosca: dict, discovery: dict, taxonomy: Optional[dict] = None) -> Problem:
    taxonomy = taxonomy if taxonomy is not None else load_taxonomy()
    nodes = nodes_from_discovery(discovery, taxonomy)
    names = [n.name for n in nodes]
    cons = constraints_from_template(tosca, names)
    currency = str(discovery.get("currency") or "USD") if isinstance(discovery, dict) else "USD"
    return Problem(nodes, cons, currency, groups_from_template(tosca, names))
