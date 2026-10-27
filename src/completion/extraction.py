"""
Extraction step (Section 6.1): the dependent resources of a service, from its official documentation pages.

  1. each page is split into chunks that fit the context of the LLM;
  2. each LLM (three different LLMs by default) extracts, from every chunk, all the dependent resources of
     the service, each with a quote, its properties and the settings of its provider (Fig. 3);
     the resources found by one LLM in all the chunks and pages of the service are merged by name;
  3. majority vote: a resource is kept when a majority of the LLMs found it (same name, see names.py);
     it keeps the quote of the first LLM whose quote appears in its page, and every setting found by
     any of these LLMs, so that a setting that may contradict a policy is never discarded.
The algorithm only parses, merges and counts; the verification module checks the result afterwards.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..llm_call import ask_text, llm_name
from .names import acronyms, normalize, same_name, strip_paren
from .prompts import EXTRACTION_PROMPT
from .text import chunks, find_quote

CHUNK_CHARS = 12000


@dataclass
class Candidate:
    name: str
    quote: str
    url: str
    properties: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)          # property -> list of values
    attached_to: Optional[str] = None
    llm: str = ""


def _parse_assign(text: str, vocabulary) -> dict:
    return vocabulary.parse(text) if vocabulary is not None else {}


def parse_output(text: str, url: str, vocabulary, llm: str = "") -> list[Candidate]:
    out = []
    for block in re.split(r"(?im)^\s*RESOURCE\s*:", text or "")[1:]:
        lines = block.strip().split("\n")
        name = lines[0].replace("`", "").strip().strip("<>*").strip()
        if not name or name.lower() in ("none", "n/a", "-") or re.match(r"(?i)none\b", name):
            continue
        body = "\n".join(lines[1:])

        def field_(label):
            m = re.search(rf"(?im)^\s*{label}\s*:\s*(.*?)(?=^\s*(?:QUOTE|ATTACHED TO|PROPERTIES|SETTINGS)\s*:|\Z)",
                          body, re.S | re.M)
            return m.group(1).strip() if m else ""
        quote = field_("QUOTE").strip().strip("\"'“”*_ ").strip()
        attached = field_("ATTACHED TO").strip().strip("<>`* ")
        props = _parse_assign(field_("PROPERTIES"), vocabulary)
        sets = {k: [v] for k, v in _parse_assign(field_("SETTINGS"), vocabulary).items()}
        out.append(Candidate(name, quote, url, props, sets, attached or None, llm))
    return out


def _merge_into(dst: Candidate, src: Candidate, pages: dict[str, str]):
    if not find_quote(dst.quote, pages.get(dst.url, "")) and find_quote(src.quote, pages.get(src.url, "")):
        dst.quote, dst.url = src.quote, src.url
    for k, v in src.properties.items():
        dst.properties.setdefault(k, v)
    for k, vs in src.settings.items():
        cur = dst.settings.setdefault(k, [])
        for v in vs:
            if v not in cur:
                cur.append(v)
    if not dst.attached_to and src.attached_to:
        dst.attached_to = src.attached_to


def merge_by_name(cands: list[Candidate], pages: dict[str, str], acro: dict) -> list[Candidate]:
    merged: list[tuple[tuple, Candidate]] = []
    for c in cands:
        n = normalize(strip_paren(c.name), acro)
        for key, m in merged:
            if same_name(key, n):
                _merge_into(m, c, pages)
                break
        else:
            merged.append((n, Candidate(c.name, c.quote, c.url, dict(c.properties),
                                        {k: list(v) for k, v in c.settings.items()}, c.attached_to, c.llm)))
    return [m for _, m in merged]


def extract_with_llm(llm, service: str, provider: str, pages: dict[str, str], vocabulary,
                     max_chars: int = CHUNK_CHARS, errors: Optional[list] = None,
                     raw: Optional[list] = None) -> list[Candidate]:
    name = llm_name(llm)
    acro = acronyms("\n".join(pages.values()))
    found: list[Candidate] = []
    voc = vocabulary.describe() if vocabulary is not None else ""
    for url, text in pages.items():
        for ch in chunks(text, max_chars):
            prompt = EXTRACTION_PROMPT.format(service=service, provider=provider, page=ch, vocabulary=voc)
            try:
                out = ask_text(llm, prompt)
            except Exception as exc:                      # noqa: BLE001 - one failing call loses one chunk
                if errors is not None:
                    errors.append({"llm": name, "url": url, "error": str(exc)})
                continue
            parsed = parse_output(out, url, vocabulary, name)
            if raw is not None:                           # raw answers, logged for the evaluation
                raw.append({"llm": name, "url": url, "chunk_chars": len(ch), "output": out,
                            "parsed": [c.name for c in parsed]})
            found += parsed
    return merge_by_name(found, pages, acro)


def vote(per_llm: list[list[Candidate]], pages: dict[str, str], n_llms: int) -> list[dict]:
    acro = acronyms("\n".join(pages.values()))
    threshold = n_llms // 2 + 1
    clusters: list[dict] = []                     # {"key", "members": [Candidate], "llms": set}
    for i, cands in enumerate(per_llm):
        for c in cands:
            n = normalize(strip_paren(c.name), acro)
            for cl in clusters:
                if i not in cl["llms"] and same_name(cl["key"], n):
                    cl["members"].append(c)
                    cl["llms"].add(i)
                    break
            else:
                clusters.append({"key": n, "members": [c], "llms": {i}})
    kept = [cl for cl in clusters if len(cl["llms"]) >= threshold]
    out = []
    for cl in kept:
        anchored = [m for m in cl["members"] if find_quote(m.quote, pages.get(m.url, ""))]
        rep = anchored[0] if anchored else cl["members"][0]
        settings: dict[str, list] = {}
        for m in cl["members"]:
            for k, vs in m.settings.items():
                cur = settings.setdefault(k, [])
                for v in vs:
                    if v not in cur:
                        cur.append(v)
        out.append({"name": rep.name, "quote": rep.quote, "url": rep.url, "properties": dict(rep.properties),
                    "settings": settings, "attached_to": rep.attached_to, "votes": len(cl["llms"]),
                    "llms": sorted({m.llm for m in cl["members"]}), "_key": cl["key"]})
    # resolve "attached to" another kept resource (otherwise the resource is attached to the service)
    for r in out:
        target = None
        if r["attached_to"]:
            t = normalize(strip_paren(r["attached_to"]), acro)
            for other in out:
                if other is not r and same_name(other["_key"], t):
                    target = other["name"]
                    break
        r["attached_to"] = target
    for r in out:
        r.pop("_key", None)
    return out


def extract_service(llms: list, service: str, provider: str, pages: dict[str, str], vocabulary,
                    max_chars: int = CHUNK_CHARS) -> tuple[list[dict], dict]:
    errors: list = []
    raw: list = []
    per_llm = [extract_with_llm(llm, service, provider, pages, vocabulary, max_chars, errors, raw) for llm in llms]
    resources = vote(per_llm, pages, len(llms))
    report = {"llms": [llm_name(l) for l in llms],
              "found_per_llm": [[c.name for c in cs] for cs in per_llm],
              "per_llm": [[{"name": c.name, "quote": c.quote, "url": c.url, "properties": c.properties,
                            "settings": c.settings, "attached_to": c.attached_to} for c in cs] for cs in per_llm],
              "kept": [r["name"] for r in resources], "errors": errors, "raw": raw,
              "chunks": {u: len(chunks(t, max_chars)) for u, t in pages.items()}}
    return resources, report
