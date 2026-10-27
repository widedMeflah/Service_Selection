"""
Writes the result of the negotiated selection into the template: each node is bound to its selected offer
(metadata), and the policies carry the values agreed during the negotiation.
"""
from __future__ import annotations

import copy

from .common.tosca import node_templates
from .selection.builder import COST_KEYS, AVAIL_KEYS
from .selection.problem import Problem, Composition, COST, objectives, availability


def _policies_list(t: dict) -> list:
    tt = t.setdefault("topology_template", {})
    pols = tt.get("policies") or []
    if isinstance(pols, dict):
        pols = [{k: v} for k, v in pols.items()]
    tt["policies"] = pols
    return pols


def write_selection(template: dict, problem: Problem, composition: Composition, decisions: list) -> dict:
    t = copy.deepcopy(template)
    nodes = node_templates(t)
    for name, o in composition.items():
        node = nodes.setdefault(name, {})
        md = node.setdefault("metadata", {})
        md.update({"provider": o.provider, "region": o.region, "offer": o.name or o.offer_id,
                   "offer_id": o.offer_id, "monthly_price": o.price_cents / 100.0,
                   "availability": o.availability, "currency": problem.currency})
    pols = _policies_list(t)
    removed = []
    for d in decisions:
        for ch in d.get("changes") or []:
            for item in list(pols):
                if ch["constraint_id"] not in item:
                    continue
                body = item[ch["constraint_id"]]
                if ch["kind"] == "removed":
                    pols.remove(item)
                    removed.append({"policy": ch["constraint_id"], "round": d.get("round")})
                    continue
                props = body.setdefault("properties", {})
                keys = COST_KEYS if ch["criterion"] == COST else AVAIL_KEYS
                k = next((k for k in keys if k in props), keys[0])
                props[k] = ch["new"]
                body.setdefault("metadata", {}).update({"requested_value": ch["old"], "negotiated_round": d.get("round")})
    cost, av = objectives(composition, problem.node_names, problem.groups)
    known = availability(composition, problem.node_names, problem.groups) is not None
    t.setdefault("metadata", {})["selection"] = {
        "total_monthly_price": cost / 100.0, "end_to_end_availability": round(av, 6) if known else None, "currency": problem.currency,
        "negotiation": decisions, "removed_policies": removed}
    return t
