"""
Evaluation dataset: cleaning of the templates of our previous work (C3), derived cases, and the exact ground truth of
the selection (exhaustive oracle).

Cleaning of a C3 template (every change is logged with the case)
  1. a policy target that names no node (e.g. "VM-A" for node VM_A) is mapped to the node it designates, when exactly
     one node matches its normalized name; otherwise the target is dropped;
  2. the nodes outside the scope of this work (network, block storage, object storage, load balancer) are removed,
     with the nodes hosted on them and the requirements towards them; a node whose custom type then lacks a mandatory
     requirement takes the first type of its derived_from chain whose mandatory requirements are present
     (e.g. computeWithnetwork -> tosca.nodes.Compute, since the network becomes a dependent resource);
  3. Latency policies are removed (the catalogs publish no latency), and so are Placement policies without a region;
  4. a DBMS hosted on a virtual machine is installed software (only the machine is a cloud service); the
     managed-database variant of a template, used for derived cases, removes this host requirement, and the machine
     when it hosts nothing else, its policies being moved to the DBMS;
  5. missing policy types are defined, unused node types are removed, and the template is validated.

Ground truth of the selection
  The offers dominated by another offer of the same node, provider and logical location (not cheaper and not more
  available) cannot change the optimum, the non-dominated compositions or the feasibility, since the constraints
  depend only on cost, availability, provider and location; they are removed, and the remaining compositions are
  enumerated exhaustively.
"""
from __future__ import annotations

import copy
import itertools
import re
from typing import Optional

from ..common.tosca import iter_policies, node_templates, policy_kind
from ..selection.problem import (Problem, Node, QoSConstraint, RelationalConstraint, COST, AVAILABILITY,
                                 feasible, objectives, value, dominates, availability)
from .discovery import catalog_of
from .tosca_check import validate, mandatory_requirements, _reqs

OUT_OF_SCOPE = {"network", "block_storage", "object_storage", "loadbalancer"}
POLICY_TYPES = {
    "acme.policies.Placement": {"derived_from": "tosca.policies.Placement",
                                "description": "Deployment locations of the targets (regions or logical locations).",
                                "properties": {"locations": {"type": "list", "required": True}}},
    "acme.policies.Cost": {"derived_from": "tosca.policies.Root",
                           "description": "Monthly budget of the targets (sum of their prices).",
                           "properties": {"max_cost": {"type": "float", "required": True},
                                          "currency": {"type": "string", "default": "USD"}}},
    "acme.policies.Availability": {"derived_from": "tosca.policies.Performance",
                                   "description": "End-to-end availability of the targets (product, percent).",
                                   "properties": {"availability": {"type": "float", "required": True}}},
    "acme.policies.Colocation": {"derived_from": "tosca.policies.Placement",
                                 "description": "The targets share a region (logical location) or a provider.",
                                 "properties": {"scope": {"type": "string", "required": True,
                                                          "constraints": [{"valid_values": ["region", "provider"]}]}}},
    "acme.policies.Redundancy": {"derived_from": "tosca.policies.Root",
                                 "description": "The targets are replicas of one another (e.g. a primary and its "
                                                "standby): the group is available when one of them is."},
    "acme.policies.Exposure": {"derived_from": "tosca.policies.Root",
                               "description": "Whether the targets may be reachable from the Internet.",
                               "properties": {"public": {"type": "boolean", "required": True}}},
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _tokens(s: str) -> list[str]:
    return [t for t in _norm(s).split("_") if t]


def _match_node(target: str, nodes: dict) -> Optional[str]:
    if target in nodes:
        return target
    exact = [n for n in nodes if _norm(n) == _norm(target)]
    if len(exact) == 1:
        return exact[0]
    tt = _tokens(target)
    scored = []
    for n in nodes:
        nt = _tokens(n)
        if len(nt) != len(tt):
            continue
        pairs = list(zip(sorted(nt), sorted(tt)))
        if all(a.startswith(b) or b.startswith(a) for a, b in pairs):
            scored.append((sum(a == b for a, b in pairs), n))
    if not scored:
        return None
    top = max(s for s, _ in scored)
    best = [n for s, n in scored if s == top]
    return best[0] if len(best) == 1 else None


def _set_reqs(node: dict, keep) -> None:
    out = []
    for r in node.get("requirements") or []:
        if isinstance(r, dict):
            r = {k: v for k, v in r.items() if keep(k, v.get("node") if isinstance(v, dict) else v)}
            if r:
                out.append(r)
    if out:
        node["requirements"] = out
    else:
        node.pop("requirements", None)


def _host(node: dict) -> Optional[str]:
    return next((t for k, t in _reqs(node) if k == "host"), None)


def _policies(tt: dict) -> list[tuple[str, dict]]:
    return [(n, b) for p in tt.get("policies") or [] for n, b in p.items()]


def _set_policies(tt: dict, items: list[tuple[str, dict]]) -> None:
    if items:
        tt["policies"] = [{n: b} for n, b in items]
    else:
        tt.pop("policies", None)


def clean_template(t0: dict, managed_db: bool = False) -> tuple[dict, list[str]]:
    t = copy.deepcopy(t0)
    log: list[str] = []
    tt = t.setdefault("topology_template", {})
    nodes = tt.setdefault("node_templates", {})
    node_types = t.get("node_types") or {}

    # 1. policy targets
    items = []
    for name, body in _policies(tt):
        tg = []
        for x in body.get("targets") or []:
            m = _match_node(x, nodes)
            if m is None:
                log.append(f"policy {name}: target {x} matches no node, dropped")
            else:
                if m != x:
                    log.append(f"policy {name}: target {x} -> {m}")
                if m not in tg:
                    tg.append(m)
        if body.get("targets") is not None:
            body["targets"] = tg
        items.append((name, body))
    _set_policies(tt, items)

    # undefined custom types of virtual machines (e.g. computeWithblocStorage used without definition)
    for n, b in nodes.items():
        ty = str(b.get("type") or "")
        if ty.lower().startswith("computewith") and ty not in node_types:
            log.append(f"node {n}: undefined type {ty} -> tosca.nodes.Compute")
            b["type"] = "tosca.nodes.Compute"

    # 2. nodes outside the scope
    removed = {n for n, b in nodes.items() if catalog_of(b.get("type")) in OUT_OF_SCOPE}
    changed = True
    while changed:                                   # nodes hosted on removed nodes
        changed = False
        for n, b in nodes.items():
            if n not in removed and _host(b) in removed:
                removed.add(n)
                changed = True
    for n in sorted(removed):
        log.append(f"node {n} ({nodes[n].get('type')}) removed: outside the scope (virtual machines, databases)")
        del nodes[n]
    for n, b in nodes.items():
        _set_reqs(b, lambda k, tgt: tgt not in removed)
        ep = ((b.get("capabilities") or {}).get("endpoint") or {}).get("properties") or {}
        if ep.get("network_name") in removed:
            log.append(f"node {n}: endpoint network_name {ep.pop('network_name')} removed with its node")
    for n, b in nodes.items():
        ty = b.get("type")
        present = {k for k, _ in _reqs(b)}
        new = ty
        while new in node_types and any(r not in present for r in mandatory_requirements(new, node_types)):
            new = node_types[new].get("derived_from")
        if new != ty:
            log.append(f"node {n}: type {ty} -> {new} (its mandatory requirement concerned a removed node)")
            b["type"] = new

    # 4. managed-database variant
    if managed_db:
        for n, b in list(nodes.items()):
            if catalog_of(b.get("type")) == "database" and _host(b):
                vm = _host(b)
                _set_reqs(b, lambda k, tgt: k != "host")
                log.append(f"node {n}: managed database (host {vm} removed)")
                cap = (((b.get("capabilities") or {}).get("host") or {}).get("properties")) or {}
                vm_cap = (((nodes.get(vm) or {}).get("capabilities") or {}).get("host") or {}).get("properties") or {}
                if not cap.get("num_cpus") and not cap.get("mem_size") and (vm_cap.get("num_cpus") or vm_cap.get("mem_size")):
                    cap = b.setdefault("capabilities", {}).setdefault("host", {}).setdefault("properties", {})
                    for k in ("num_cpus", "mem_size"):
                        if vm_cap.get(k):
                            cap[k] = vm_cap[k]
                    log.append(f"node {n}: size of {vm} kept for the managed database "
                               f"({cap.get('num_cpus')} CPU, {cap.get('mem_size')})")
                hosts_other = any(_host(o) == vm for m, o in nodes.items() if m != n)
                if vm in nodes and not hosts_other:
                    del nodes[vm]
                    log.append(f"node {vm} removed: it only hosted {n}")
                    for m, o in nodes.items():
                        _set_reqs(o, lambda k, tgt: tgt != vm)
                    items = []
                    for name, body in _policies(tt):
                        if body.get("targets"):
                            tg = []
                            for x in body["targets"]:
                                y = n if x == vm else x
                                if y not in tg:
                                    tg.append(y)
                            body["targets"] = tg
                        items.append((name, body))
                    _set_policies(tt, items)

    # 3. policies
    items = []
    for name, body in _policies(tt):
        kind = policy_kind(body.get("type"))
        props = body.get("properties") or {}
        if kind == "latency":
            log.append(f"policy {name} removed: no latency in the catalogs")
            continue
        if kind == "placement":
            locs = props.get("locations") or props.get("location")
            locs = [l for l in (locs if isinstance(locs, list) else [locs])
                    if (l.get("region") if isinstance(l, dict) else l)]
            if not locs:
                log.append(f"policy {name} removed: no region")
                continue
            props["locations"] = [{"region": (l.get("region") if isinstance(l, dict) else l)} for l in locs]
            props.pop("location", None)
        if body.get("targets") is not None:
            kept = [x for x in body["targets"] if x in nodes]
            for x in body["targets"]:
                if x not in nodes:
                    log.append(f"policy {name}: target {x} removed with its node")
            body["targets"] = kept
        if body.get("targets") is not None and not body["targets"]:
            log.append(f"policy {name} removed: no target left")
            continue
        items.append((name, body))
    _set_policies(tt, items)

    # 5. types and validation
    finalize_types(t)
    return t, log


def finalize_types(t: dict) -> None:
    tt = t["topology_template"]
    used = {b.get("type") for b in tt["node_templates"].values()}
    node_types = t.get("node_types") or {}
    keep = set()
    for ty in used:
        while ty in node_types and ty not in keep:
            keep.add(ty)
            ty = node_types[ty].get("derived_from")
    if node_types:
        t["node_types"] = {k: v for k, v in node_types.items() if k in keep}
        if not t["node_types"]:
            t.pop("node_types")
    pts = t.setdefault("policy_types", {})
    for _, body in _policies(tt):
        ty = body.get("type")
        if ty not in pts and ty in POLICY_TYPES:
            pts[ty] = copy.deepcopy(POLICY_TYPES[ty])
    used_p = {b.get("type") for _, b in _policies(tt)}
    t["policy_types"] = {k: v for k, v in pts.items() if k in used_p}
    if not t["policy_types"]:
        t.pop("policy_types")
    if any(b.get("type", "").startswith("acme.datatypes") for b in []):
        pass


def add_policy(t: dict, name: str, ptype: str, targets: Optional[list], props: dict) -> None:
    tt = t["topology_template"]
    items = [(n, b) for n, b in _policies(tt) if n != name]
    body = {"type": ptype, "properties": props}
    if targets is not None:
        body["targets"] = list(targets)
    items.append((name, body))
    _set_policies(tt, items)
    finalize_types(t)


def remove_policies(t: dict, kinds: set[str], global_only: bool = True, service_nodes=None) -> None:
    tt = t["topology_template"]
    items = []
    for n, b in _policies(tt):
        k = policy_kind(b.get("type"))
        tg = b.get("targets")
        is_global = (not tg) or len(tg) >= 2
        if k in kinds and (is_global or not global_only):
            continue
        items.append((n, b))
    _set_policies(tt, items)
    finalize_types(t)


# ---------------------------------------------------------------------------------------------- oracle
def reduced(problem: Problem) -> Problem:
    """Removes the offers dominated within the same node, provider and logical location (exact for the optimum,
    the non-dominated compositions and the feasibility)."""
    nodes = []
    for n in problem.nodes:
        groups: dict[tuple, list] = {}
        for o in n.candidates:
            groups.setdefault((o.provider, o.location or o.region), []).append(o)
        keep = []
        for _, offers in groups.items():
            best = []
            for o in sorted(offers, key=lambda o: (o.price_cents, -(o.availability or 0.0), o.offer_id)):
                a = o.availability or 0.0
                if any(b.price_cents <= o.price_cents and (b.availability or 0.0) >= a for b in best):
                    continue
                best.append(o)
            keep += best
        nodes.append(Node(n.name, keep))
    return Problem(nodes, problem.constraints, problem.currency, problem.groups)


def n_compositions(problem: Problem) -> int:
    n = 1
    for x in problem.nodes:
        n *= max(1, len(x.candidates))
    return n


def compositions(problem: Problem, constraints=None):
    cons = problem.constraints if constraints is None else constraints
    names = problem.node_names
    for combo in itertools.product(*[problem.node(n).candidates for n in names]):
        x = dict(zip(names, combo))
        if feasible(x, cons, problem.groups):
            yield x


def front(problem: Problem, constraints=None) -> list[tuple[int, float]]:
    pts = {objectives(x, problem.node_names, problem.groups) for x in compositions(problem, constraints)}
    return sorted(p for p in pts if not any(dominates(q, p) for q in pts if q != p))


def lex_best(problem: Problem, constraints, priority: str):
    order = [priority, AVAILABILITY if priority == COST else COST]
    allnodes = tuple(problem.node_names)
    best_k, best_x = None, None
    for x in compositions(problem, constraints):
        k = []
        for crit in order:
            v = value(x, crit, allnodes, problem.groups) or 0.0
            k.append(round(v, 9) if crit == COST else -round(v, 9))
        k.append(tuple(x[n].offer_id for n in problem.node_names))
        k = tuple(k)
        if best_k is None or k < best_k:
            best_k, best_x = k, x
    return best_x


def mcs_sets(problem: Problem) -> list[tuple[str, ...]]:
    ids = [c.id for c in problem.constraints]
    for k in range(1, len(ids) + 1):
        found = [s for s in itertools.combinations(ids, k)
                 if next(compositions(problem, [c for c in problem.constraints if c.id not in s]), None) is not None]
        if found:
            return found
    return []


def relaxation(problem: Problem, mcs: tuple[str, ...]) -> dict:
    """Relaxation of the article: relational constraints removed, each QoS bound replaced by its value in the
    composition that optimizes the criteria of C without C (then cost, then availability)."""
    rest = [c for c in problem.constraints if c.id not in mcs]
    in_c = [problem.constraint(i) for i in mcs]
    qos = [c for c in in_c if isinstance(c, QoSConstraint)]
    allnodes = tuple(problem.node_names)
    spec = [(c.criterion, tuple(c.targets)) for c in qos] + [(COST, allnodes), (AVAILABILITY, allnodes)]
    best_k, best_x = None, None
    for x in compositions(problem, rest):
        k = []
        for crit, tg in spec:
            v = value(x, crit, tg, problem.groups) or 0.0
            k.append(round(v, 9) if crit == COST else -round(v, 9))
        k = tuple(k)
        if best_k is None or k < best_k:
            best_k, best_x = k, x
    new = list(rest)
    changes = []
    for c in in_c:
        if isinstance(c, RelationalConstraint):
            changes.append({"constraint": c.id, "removed": True})
        elif value(best_x, c.criterion, c.targets, problem.groups) is None:
            # no composition publishes the availability of these targets: no bound can be met, the constraint goes
            changes.append({"constraint": c.id, "removed": True})
        else:
            v = value(best_x, c.criterion, c.targets, problem.groups)
            nb = round(v, 2) if c.criterion == COST else int(v * 10_000 + 1e-9) / 10_000
            changes.append({"constraint": c.id, "criterion": c.criterion, "from": c.bound, "to": nb})
            new.append(QoSConstraint(c.id, c.criterion, c.targets, nb, c.policy))
    cost, av = objectives(best_x, allnodes, problem.groups)
    return {"mcs": list(mcs), "changes": changes, "cost_cents": cost, "availability": av,
            "availability_published": availability(best_x, allnodes, problem.groups) is not None,
            "composition": {n: best_x[n].offer_id for n in problem.node_names}, "_constraints": new}


def describe(x: dict, problem: Problem) -> dict:
    allnodes = tuple(problem.node_names)
    cost, av = objectives(x, allnodes, problem.groups)
    return {"composition": {n: x[n].offer_id for n in problem.node_names}, "cost_cents": cost, "availability": av,
            "availability_published": availability(x, allnodes, problem.groups) is not None}
