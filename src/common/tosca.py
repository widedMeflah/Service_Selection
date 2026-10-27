"""
Reading the inputs of the approach (the output of service discovery) and the policies of a TOSCA template.

Input (Section 3.2 of the paper)
  - a provider-independent TOSCA template (YAML), whose policies express the user's non-functional
    requirements over target nodes;
  - the output of discovery (JSON): for each node that represents a cloud service, its candidate offers.

Policy types are recognized by the last segment of their name, case-insensitive, so that
"Cost", "cloud.policies.Cost" and "tosca.policies.Cost" are the same type.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Iterable, Optional

import yaml


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_yaml(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def policy_kind(policy_type: Any) -> str:
    """'cloud.policies.Cost' -> 'cost'."""
    return str(policy_type or "").split(".")[-1].strip().lower()


def node_templates(tosca: dict) -> dict:
    return ((tosca.get("topology_template") or {}).get("node_templates") or {})


def iter_policies(tosca: dict) -> Iterable[tuple[str, dict]]:
    """(name, body) of each policy, whatever the YAML form (list of single-key maps, or a map)."""
    pols = (tosca.get("topology_template") or {}).get("policies") or []
    if isinstance(pols, dict):
        pols = [{k: v} for k, v in pols.items()]
    for item in pols:
        if isinstance(item, dict):
            for name, body in item.items():
                yield str(name), (body or {})


def requirements(node: dict) -> list[tuple[str, Any]]:
    out = []
    for r in node.get("requirements") or []:
        if isinstance(r, dict):
            for k, v in r.items():
                out.append((k, v))
    return out


def requirement_target(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        n = value.get("node")
        return n if isinstance(n, str) else None
    return None


def host_of(tosca: dict, node: str) -> Optional[str]:
    for k, v in requirements(node_templates(tosca).get(node) or {}):
        if k == "host":
            return requirement_target(v)
    return None


def resolve_to(tosca: dict, node: str, service_nodes: set[str]) -> Optional[str]:
    """The service node that carries `node`: itself, or the first service node along its host chain."""
    seen = set()
    cur: Optional[str] = node
    while cur is not None and cur not in seen:
        if cur in service_nodes:
            return cur
        seen.add(cur)
        cur = host_of(tosca, cur)
    return None


def policy_targets(tosca: dict, body: dict, service_nodes: set[str]) -> list[str]:
    """Service nodes targeted by a policy (no targets = every service node)."""
    targets = body.get("targets")
    if not targets:
        return sorted(service_nodes)
    out = []
    for t in targets:
        r = resolve_to(tosca, str(t), service_nodes)
        if r is not None and r not in out:
            out.append(r)
    return out


def deep_copy(d: dict) -> dict:
    return copy.deepcopy(d)
