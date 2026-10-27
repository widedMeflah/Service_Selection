"""
Ground truth of the dependent resources (data/eval/annotation/dependent_resources.json), annotated on the frozen
official pages (data/eval/snapshots/), with the definition of the paper: a dependent resource is a resource that the deployment
code must create or reference for the service; the parts that the provider creates automatically with the service,
configured only through the attributes of the service, are excluded; a resource that the provider supplies by default
when none is specified is kept.

File format
  {"definition": ..., "annotated_on": date, "snapshot_date": date,
   "services": [{"service": "compute", "provider": "aws",
                 "resources": [{"name", "url", "quote", "properties": {}, "settings": {}, "attached_to": null,
                                "rationale"}],
                 "excluded":  [{"name", "url", "quote", "reason"}]}]}
  quote        copied word for word from the frozen page of url; it states that the service requires the resource
  properties   values of the policy vocabulary (data/policy_vocabulary.json) that the page states for the resource
  settings     values that the provider applies by default, according to the page (property -> list of values)
  attached_to  name of the resource it is attached to (e.g. a subnet to a VPC), null for the service node
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ..completion.text import find_quote

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DEFAULT_PATH = os.path.join(DATA, "eval", "annotation", "dependent_resources.json")


def load(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        return json.load(f)


def by_service(ann: dict) -> dict[tuple[str, str], dict]:
    return {(s["service"], s["provider"]): s for s in ann["services"]}


def memory_resources(entry: dict) -> list[dict]:
    """Resources in the format of the memory and of the insertion step."""
    return [{"name": r["name"], "url": r["url"], "quote": r["quote"], "properties": dict(r.get("properties") or {}),
             "settings": {k: list(v) for k, v in (r.get("settings") or {}).items()},
             "attached_to": r.get("attached_to")} for r in entry["resources"]]


def check(ann: dict, urls, fetcher) -> list[str]:
    """Issues of the annotation: unknown or non-official source, quote not found word for word in the frozen page,
    attachment to an unknown resource, duplicate names, missing rationale."""
    issues = []
    for s in ann["services"]:
        key = f"{s['provider']}/{s['service']}"
        known = urls.urls_for(s["service"], s["provider"])
        names = [r["name"] for r in s["resources"]]
        if len(set(names)) != len(names):
            issues.append(f"{key}: duplicate resource names")
        for kind in ("resources", "excluded"):
            for r in s.get(kind) or []:
                where = f"{key} {kind[:-1] if kind == 'resources' else 'excluded'} {r['name']}"
                if r["url"] not in known:
                    issues.append(f"{where}: url not in data/doc_urls.json for this service")
                if not urls.is_official(r["url"], s["provider"]):
                    issues.append(f"{where}: url not official")
                try:
                    page = fetcher.fetch(r["url"])
                except Exception as exc:              # noqa: BLE001
                    issues.append(f"{where}: page cannot be read ({exc})")
                    continue
                if not find_quote(r["quote"], page):
                    issues.append(f"{where}: quote not found word for word")
                if not (r.get("rationale") or r.get("reason")):
                    issues.append(f"{where}: no rationale")
                if kind == "resources" and r.get("attached_to") and r["attached_to"] not in names:
                    issues.append(f"{where}: attached to unknown resource {r['attached_to']}")
    return issues


def _sentences(text: str) -> list[str]:
    import re
    parts = []
    for line in (text or "").split("\n"):
        parts += [s.strip() for s in re.split(r"(?<=[.!?:;])\s+", line) if s.strip()]
    return parts


def align(ann: dict, fetcher, threshold: float = 0.9) -> tuple[dict, list[str]]:
    """Quotes aligned with the frozen pages: a quote found word for word is kept; otherwise it is replaced by the
    closest passage of the frozen page (one or two consecutive sentences, or the part of a sentence of the same
    length), when the similarity of the normalized texts is at least threshold. Every replacement is logged; a quote
    without such a passage is left as it is (and reported by check)."""
    import copy
    from difflib import SequenceMatcher
    from ..completion.text import normalize
    out, log, cache = copy.deepcopy(ann), [], {}
    for s in out["services"]:
        for kind in ("resources", "excluded"):
            for r in s.get(kind) or []:
                url = r["url"]
                if url not in cache:
                    try:
                        cache[url] = fetcher.fetch(url)
                    except Exception:                 # noqa: BLE001
                        cache[url] = None
                page = cache[url]
                if page is None or find_quote(r["quote"], page):
                    continue
                q = normalize(r["quote"]).strip(" .\"'")
                sents = _sentences(page)
                cands = sents + [a + " " + b for a, b in zip(sents, sents[1:])]
                best, score = None, 0.0
                for c in cands:
                    nc = normalize(c)
                    if len(nc) > len(q) * 1.3:      # a longer sentence: its window of the quote's length
                        m = SequenceMatcher(None, q, nc, autojunk=False).find_longest_match(0, len(q), 0, len(nc))
                        start = max(0, m.b - m.a)
                        window = c[start:start + len(r["quote"]) + 10]
                        nc_w = normalize(window)
                        sc = SequenceMatcher(None, q, nc_w, autojunk=False).ratio()
                        cand = window
                    else:
                        sc = SequenceMatcher(None, q, nc, autojunk=False).ratio()
                        cand = c
                    if sc > score:
                        best, score = cand, sc
                if best is not None and score >= threshold and find_quote(best, page):
                    log.append(f"{s['provider']}/{s['service']} {r['name']}: quote aligned ({score:.2f}) "
                               f"{r['quote']!r} -> {best!r}")
                    r["quote"] = best.strip()
                else:
                    log.append(f"{s['provider']}/{s['service']} {r['name']}: quote not found in the frozen page "
                               f"(best similarity {score:.2f})")
    return out, log
