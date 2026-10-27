"""
Insertion step (Section 6.1): an algorithm, never the LLM, adds the dependent resources to the template.

  - each resource becomes a new node, attached to the node that requires it (a dependency requirement),
    and annotated with its source page, its quote and the settings of its provider;
  - a resource required by several nodes of the same provider and region (same name once normalized, e.g.
    "VPC" and "VPC (Virtual Private Cloud)", and properties that do not contradict each other), such as a VPC, is inserted once and attached to each of them, with the
    union of their properties; two resources whose properties contradict each other are inserted apart;
  - the node requires each of its resources; a resource that requires another resource (e.g., a subnet
    requires a VPC) is also attached to it;
  - a property is written in the properties of the node only when its TOSCA type defines it; the other
    values (those of the policy vocabulary, e.g. internet_exposure) are written in its metadata
    (policy_properties), so that the template stays valid for a TOSCA parser; true/false are booleans;
  - nothing is removed or rewritten: the selected services, their bindings and the policies are preserved
    by construction.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from ..common.tosca import node_templates
from .names import acronyms_of_names, identity, normalize, same_name, slug, strip_paren, words

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


class TypeMap:
    """TOSCA node type of a resource, from the head noun of its name (data/resource_types.json)."""

    def __init__(self, path: Optional[str] = None):
        with open(path or os.path.join(DATA, "resource_types.json"), encoding="utf-8") as f:
            d = json.load(f)
        self.default = d.get("default", "tosca.nodes.Root")
        self.rules = d.get("rules", [])
        self.type_properties: dict[str, list[str]] = d.get("type_properties", {})

    def defines(self, tosca_type: str, prop: str) -> bool:
        return prop in self.type_properties.get(tosca_type, [])

    def type_of(self, name: str) -> str:
        w = " ".join(words(name))
        for r in self.rules:
            for k in r.get("ends_with", []):
                kk = " ".join(words(k))
                if w == kk or w.endswith(" " + kk):
                    return r["type"]
        return self.default


@dataclass
class Inserted:
    """One inserted dependent resource, for one node that requires it (a shared resource appears once per node)."""
    id: str                       # name of the inserted node in the template
    node: str                     # service node that requires it
    service: str
    provider: str
    region: str
    name: str
    quote: str
    url: str
    properties: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    attached_to: Optional[str] = None      # id of the resource it is attached to (None: the service node)
    origin: str = "extracted"              # extracted | memory


def _subst(props: dict, region: str) -> dict:
    return {k: (str(v).replace("{region}", region) if isinstance(v, str) else v) for k, v in props.items()}


def _typed(v):
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return v


def _typed_settings(settings: dict) -> dict:
    return {k: [_typed(x) for x in vs] for k, vs in (settings or {}).items()}


def _compatible(a: dict, b: dict) -> bool:
    """Same resource when no property has two different values (e.g. a region written by one source only)."""
    return all(str(a[k]).lower() == str(b[k]).lower() for k in a.keys() & b.keys())


def _merge(node: dict, known: dict, props: dict, settings: Optional[dict], types: "TypeMap"):
    """A shared resource found again: the union of the properties and of the settings."""
    new = {k: v for k, v in props.items() if k not in known["properties"]}
    known["properties"].update(new)
    for k, v in new.items():
        v = _typed(v)
        if types.defines(node["type"], k):
            node.setdefault("properties", {})[k] = v
        else:
            node["metadata"].setdefault("policy_properties", {})[k] = v
    md = node["metadata"]
    for k, vs in _typed_settings(settings).items():
        cur = md["settings"].setdefault(k, [])
        for v in vs:
            if v not in cur:
                cur.append(v)


def insert(template: dict, plan: list[dict], types: Optional[TypeMap] = None) -> tuple[dict, list[Inserted]]:
    """plan: [{"node", "service", "provider", "region", "resources": [resource dicts], "origin"}]"""
    types = types or TypeMap()
    t = copy.deepcopy(template)
    nodes = node_templates(t)
    inserted: list[Inserted] = []
    by_id: dict[str, dict] = {}
    by_identity: dict[tuple, str] = {}
    acro = acronyms_of_names(r["name"] for item in plan for r in item["resources"])
    for item in plan:
        node, prov, region = item["node"], item["provider"], item["region"]
        ids: dict[str, str] = {}
        for r in item["resources"]:
            props = _subst(r.get("properties") or {}, region)
            key = (identity(r["name"], acro), slug(prov), slug(region))
            rid = by_identity.get(key) or f"{slug(strip_paren(r['name']))}_{slug(prov)}_{slug(region)}"
            if rid in by_id and (by_id[rid]["key"] != key or not _compatible(by_id[rid]["properties"], props)):
                rid = f"{rid}_{slug(node)}"
            by_identity.setdefault(key, rid)
            ids[r["name"]] = rid
            if rid in by_id:
                _merge(nodes[rid], by_id[rid], props, r.get("settings"), types)
            else:
                by_id[rid] = {"properties": props, "key": key}
                ttype = types.type_of(r["name"])
                typed = {k: _typed(v) for k, v in props.items()}
                own = {k: v for k, v in typed.items() if types.defines(ttype, k)}
                other = {k: v for k, v in typed.items() if not types.defines(ttype, k)}
                nodes[rid] = {
                    "type": ttype,
                    **({"properties": own} if own else {}),
                    "metadata": {"dependent_resource": True, "provider": prov, "region": region,
                                 "source": r.get("url", ""), "quote": r.get("quote", ""),
                                 **({"policy_properties": other} if other else {}),
                                 "settings": _typed_settings(r.get("settings")), "required_by": []},
                }
            md = nodes[rid]["metadata"]
            if node not in md["required_by"]:
                md["required_by"].append(node)
        for r in item["resources"]:
            rid = ids[r["name"]]
            target = ids.get(r.get("attached_to")) if r.get("attached_to") else None
            reqs = nodes[node].setdefault("requirements", [])          # the service requires every resource
            if {"dependency": rid} not in reqs:
                reqs.append({"dependency": rid})
            if target and target != rid:                              # e.g., a subnet requires its VPC
                rr = nodes[rid].setdefault("requirements", [])
                if {"dependency": target} not in rr:
                    rr.append({"dependency": target})
            inserted.append(Inserted(rid, node, item["service"], prov, region, r["name"], r.get("quote", ""),
                                     r.get("url", ""), dict(r.get("properties") or {}),
                                     {k: list(v) for k, v in (r.get("settings") or {}).items()},
                                     target, item.get("origin", "extracted")))
    return t, inserted
