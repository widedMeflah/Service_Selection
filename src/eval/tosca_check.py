"""
Structural validation of the TOSCA templates of the evaluation (TOSCA Simple Profile in YAML 1.3).

Checked: version; every node type is normative (full or short name) or defined in node_types and derived, through
derived_from, from a normative type; every requirement of a node targets an existing node; the mandatory requirements
of a custom type (occurrences with a minimum >= 1) are present; host chains have no cycle; every policy type is
normative or defined in policy_types; every policy target exists; the required properties of the policies are set.
"""
from __future__ import annotations

NORMATIVE_NODES = {
    "Root", "Compute", "SoftwareComponent", "WebServer", "WebApplication", "DBMS", "Database",
    "ObjectStorage", "BlockStorage", "Container.Runtime", "Container.Application", "LoadBalancer",
    "network.Network", "network.Port", "Storage.ObjectStorage", "Storage.BlockStorage",
}
NORMATIVE_POLICIES = {"Root", "Placement", "Scaling", "Update", "Performance"}


def _short(t: str) -> str:
    t = str(t or "")
    for p in ("tosca.nodes.", "tosca.policies."):
        if t.startswith(p):
            return t[len(p):]
    return t


def node_type_ok(t: str, node_types: dict, seen=None) -> bool:
    seen = seen or set()
    if _short(t) in NORMATIVE_NODES:
        return True
    if t in seen or t not in node_types:
        return False
    seen.add(t)
    return node_type_ok(node_types[t].get("derived_from"), node_types, seen)


def mandatory_requirements(t: str, node_types: dict) -> list[str]:
    out, seen = [], set()
    while t in node_types and t not in seen:
        seen.add(t)
        for r in node_types[t].get("requirements") or []:
            name, spec = next(iter(r.items()))
            occ = (spec or {}).get("occurrences")
            if isinstance(occ, list) and occ and isinstance(occ[0], int) and occ[0] >= 1:
                out.append(name)
        t = node_types[t].get("derived_from")
    return out


def _reqs(node: dict) -> list[tuple[str, str]]:
    out = []
    for r in node.get("requirements") or []:
        if isinstance(r, dict):
            for k, v in r.items():
                tgt = v.get("node") if isinstance(v, dict) else v
                out.append((k, str(tgt)))
    return out


def policy_required(ptype: str, policy_types: dict) -> list[str]:
    out = []
    for name, spec in ((policy_types.get(ptype) or {}).get("properties") or {}).items():
        if (spec or {}).get("required") and "default" not in (spec or {}):
            out.append(name)
    return out


def validate(t: dict) -> list[str]:
    issues = []
    if t.get("tosca_definitions_version") != "tosca_simple_yaml_1_3":
        issues.append(f"version {t.get('tosca_definitions_version')}")
    node_types = t.get("node_types") or {}
    policy_types = t.get("policy_types") or {}
    tt = t.get("topology_template") or {}
    nodes = tt.get("node_templates") or {}
    if not nodes:
        issues.append("no node template")
    for n, b in nodes.items():
        ty = (b or {}).get("type")
        if not node_type_ok(ty, node_types):
            issues.append(f"node {n}: unknown type {ty}")
        names = set()
        for k, tgt in _reqs(b or {}):
            names.add(k)
            if tgt not in nodes:
                issues.append(f"node {n}: requirement {k} targets missing node {tgt}")
        for k in mandatory_requirements(ty, node_types):
            if k not in names:
                issues.append(f"node {n}: mandatory requirement {k} of type {ty} is missing")
    for n in nodes:                                    # host cycles
        seen, cur = set(), n
        while cur is not None:
            if cur in seen:
                issues.append(f"node {n}: host cycle")
                break
            seen.add(cur)
            cur = next((tgt for k, tgt in _reqs(nodes.get(cur) or {}) if k == "host"), None)
    for p in tt.get("policies") or []:
        for name, body in (p.items() if isinstance(p, dict) else []):
            ptype = (body or {}).get("type")
            if _short(ptype) not in NORMATIVE_POLICIES and ptype not in policy_types:
                issues.append(f"policy {name}: unknown type {ptype}")
            for tg in (body or {}).get("targets") or []:
                if tg not in nodes:
                    issues.append(f"policy {name}: target {tg} missing")
            props = (body or {}).get("properties") or {}
            for req in policy_required(ptype, policy_types):
                if props.get(req) in (None, "", []):
                    issues.append(f"policy {name}: required property {req} missing")
    return issues
