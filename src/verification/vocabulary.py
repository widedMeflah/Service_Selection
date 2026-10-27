"""
Vocabulary shared by the extraction prompt and the verification: the properties that the policies of a
template constrain (data/policy_vocabulary.json), e.g.
    internet_exposure  <- Exposure policy (public: false  =>  internet_exposure must be false)
    region             <- Placement policy (location / locations => the region must belong to them)
It describes policies, not dependent resources: no rule about a particular resource is written here.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from ..common.tosca import iter_policies, policy_kind, policy_targets
from ..selection.builder import load_taxonomy

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


@dataclass
class Requirement:
    property: str
    compare: str                  # "equal" | "allows" | "location"
    expected: object              # "false" | list of allowed locations
    policy: str                   # name of the policy

    def text(self) -> str:
        if self.compare == "location":
            return f"{self.property} must be in {', '.join(map(str, self.expected))} (policy {self.policy})"
        if self.compare == "allows":
            return f"{self.property} must not be true (policy {self.policy})"
        return f"{self.property} must be {self.expected} (policy {self.policy})"


class Vocabulary:
    def __init__(self, path: Optional[str] = None, taxonomy: Optional[dict] = None):
        path = path or os.path.join(DATA, "policy_vocabulary.json")
        with open(path, encoding="utf-8") as f:
            self.props: dict[str, dict] = json.load(f)["properties"]
        self.taxonomy = taxonomy if taxonomy is not None else load_taxonomy()

    # ------------------------------------------------------------------ prompts
    def describe(self) -> str:
        lines = []
        for name, p in self.props.items():
            vals = f" ({' | '.join(p['values'])})" if p.get("values") else ""
            lines.append(f"  {name}{vals}: {p.get('description', '')}")
        return "\n".join(lines)

    def parse(self, text: str) -> dict[str, str]:
        """'internet_exposure=true; region={region}' -> {'internet_exposure': 'true', 'region': '{region}'}
        Only the properties of the vocabulary are kept: any other key written by the LLM is ignored."""
        out: dict[str, str] = {}
        if not text or text.strip().lower() in ("none", "-", "n/a"):
            return out
        for part in re.split(r"[;\n]", text):
            if "=" in part:
                k, v = part.split("=", 1)
                k, v = k.strip().strip("-").strip().lower(), v.strip().strip("\"'")
                if k in self.props:
                    out[k] = v.lower() if v.lower() in ("true", "false") else v
        return out

    # ------------------------------------------------------------------ policies -> requirements
    def requirements(self, tosca: dict, node: str, service_nodes: set[str]) -> list[Requirement]:
        out = []
        for pname, body in iter_policies(tosca):
            kind = policy_kind(body.get("type"))
            if node not in policy_targets(tosca, body, service_nodes):
                continue
            props = body.get("properties") or {}
            for prop, spec in self.props.items():
                if spec.get("policy") != kind:
                    continue
                fields = spec.get("policy_field")
                fields = fields if isinstance(fields, list) else [fields]
                val = next((props[f] for f in fields if f in props), None)
                if val is None:
                    continue
                if spec.get("compare") == "location":
                    locs = val if isinstance(val, list) else [val]
                    locs = [(l.get("region") if isinstance(l, dict) else l) for l in locs]
                    out.append(Requirement(prop, "location", [str(l) for l in locs if l], pname))
                else:
                    out.append(Requirement(prop, spec.get("compare", "equal"), str(val).lower(), pname))
        return out

    def _in_location(self, region: str, locations: list) -> bool:
        r = region.lower()
        for loc in locations:
            l = str(loc).lower()
            if r == l or r in [x.lower() for x in self.taxonomy.get(l, [])]:
                return True
        return False

    def violations(self, values: dict, reqs: list[Requirement], region: str) -> list[Requirement]:
        """Requirements that the given values (property -> value or list of values) do not satisfy."""
        bad = []
        for rq in reqs:
            if rq.property not in values:
                continue
            vals = values[rq.property]
            vals = vals if isinstance(vals, list) else [vals]
            for v in vals:
                v = str(v).replace("{region}", region)
                if rq.compare == "location":
                    ok = self._in_location(v, rq.expected)
                elif rq.compare == "allows":       # policy false forbids true; policy true forbids nothing
                    ok = not (str(rq.expected).lower() == "false" and v.lower() == "true")
                else:
                    ok = v.lower() == str(rq.expected).lower()
                if not ok:
                    bad.append(rq)
                    break
        return bad
