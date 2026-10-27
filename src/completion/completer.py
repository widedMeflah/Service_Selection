"""
Documentation-Grounded Completion module (Section 6.1): memory lookup, extraction, insertion,
and the reconfiguration of a resource sent back by the verification module.

Memory lookup  each service (name, provider) is looked up by exact match. When it is found, its source pages
               are read again if they have not been checked for `ttl_days`; a page whose fingerprint is
               unchanged lets its resources be reused without a new extraction; a changed page triggers the
               extraction step for the service, so that requirements added to the page are also detected.
               A page that cannot be read is reported, and the resources it justifies are not inserted.
Extraction     for a service not found, or whose source page has changed: an algorithm retrieves the pages
               from their URLs, and three LLMs extract the resources (extraction.py). When no URL is known,
               the LLM proposes candidate URLs; an algorithm keeps those on an official domain, reachable,
               whose page yields at least one resource whose quote appears in it; an expert validates them
               (once per service), and the validated URLs join the list.
Insertion      by an algorithm (insertion.py).
Nothing is written to the memory here: the verification module calls commit() once every verification passes.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from ..llm_call import ask_text
from .extraction import extract_service, CHUNK_CHARS
from .fetch import FetchError
from .insertion import insert, Inserted, TypeMap
from .memory import Memory, now_iso
from .prompts import URL_PROMPT, REPAIR_PROMPT
from .text import fingerprint, find_quote, chunks
from .urls import UrlList

TTL_DAYS = 7
MAX_URLS = 3


@dataclass
class ServiceState:
    service: str
    provider: str
    status: str                              # reused | extracted | changed | no_source | to_validate | unreachable
    resources: list[dict] = field(default_factory=list)
    sources: dict = field(default_factory=dict)        # url -> {fingerprint, checked_at} (written on commit)
    unreachable: list[str] = field(default_factory=list)
    proposals: list[dict] = field(default_factory=list)
    extraction: dict = field(default_factory=dict)
    label: str = ""                          # name given to the LLM, e.g. "EC2 m5.large (compute)"
    stored: bool = False                     # written to the memory after verification


class Completer:
    def __init__(self, llms: list, fetcher, urls: UrlList, memory: Memory, vocabulary,
                 expert: Optional[Callable] = None, ttl_days: int = TTL_DAYS, chunk_chars: int = CHUNK_CHARS,
                 types: Optional[TypeMap] = None, clock: Optional[Callable[[], datetime]] = None):
        self.llms = [l for l in llms if l is not None]
        self.fetcher = fetcher
        self.urls = urls
        self.memory = memory
        self.vocabulary = vocabulary
        self.expert = expert                 # expert(service, provider, url, resources) -> bool
        self.ttl = timedelta(days=ttl_days)
        self.chunk_chars = chunk_chars
        self.types = types or TypeMap()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.states: dict[tuple[str, str], ServiceState] = {}
        self.fetch_errors: dict[str, str] = {}       # url -> reason (reported with the unreachable pages)

    # ------------------------------------------------------------------ pages
    def _page(self, url: str) -> Optional[str]:
        try:
            return self.fetcher.fetch(url)
        except FetchError as exc:
            self.fetch_errors[url] = str(exc)[:300]
            return None

    def _fresh(self, checked_at: Optional[str]) -> bool:
        if not checked_at:
            return False
        try:
            return self.clock() - datetime.fromisoformat(checked_at) < self.ttl
        except ValueError:
            return False

    # ------------------------------------------------------------------ memory lookup / extraction
    def _lookup(self, service: str, provider: str, label: str = "") -> ServiceState:
        label = label or service
        entry = self.memory.get(service, provider)
        if entry is None:
            return self._extract(service, provider, self.urls.urls_for(service, provider), "extracted", label)
        sources, changed, unreachable = {}, False, []
        now = self.clock().isoformat()
        for url, src in (entry.get("sources") or {}).items():
            if self._fresh(src.get("checked_at")):
                sources[url] = dict(src)
                continue
            page = self._page(url)
            if page is None:
                unreachable.append(url)
            elif fingerprint(page) == src.get("fingerprint"):
                sources[url] = {"fingerprint": src.get("fingerprint"), "checked_at": now}
            else:
                changed = True
        if changed:
            urls = list((entry.get("sources") or {}).keys())
            urls += [u for u in self.urls.urls_for(service, provider) if u not in urls]
            st = self._extract(service, provider, urls, "changed", label)
            return st
        resources = [r for r in entry.get("resources", []) if r.get("url") not in unreachable]
        return ServiceState(service, provider, "reused" if not unreachable else "unreachable",
                            resources, sources, unreachable, label=label)

    def _extract(self, service: str, provider: str, urls: list[str], status: str, label: str = "") -> ServiceState:
        label = label or service
        proposals = []
        if not urls:
            urls, proposals = self._propose(service, provider, label)
            if not urls:
                return ServiceState(service, provider, "to_validate" if proposals else "no_source",
                                    proposals=proposals, label=label)
        pages, unreachable = {}, []
        for u in urls:
            p = self._page(u)
            if p is None:
                unreachable.append(u)
            else:
                pages[u] = p
        if not pages:
            return ServiceState(service, provider, "no_source", unreachable=unreachable, proposals=proposals,
                                label=label)
        resources, report = extract_service(self.llms, label, provider, pages, self.vocabulary, self.chunk_chars)
        now = self.clock().isoformat()
        sources = {u: {"fingerprint": fingerprint(t), "checked_at": now} for u, t in pages.items()}
        if report.get("errors") or unreachable:
            # a chunk could not be read by an LLM (e.g. a rate limit that lasted), or a page could not be
            # retrieved: the resources found are inserted and verified, but the service is reported and is
            # not written to the memory, so that it is extracted again in the next execution
            status = "incomplete"
        return ServiceState(service, provider, status, resources, sources, unreachable, proposals, report, label)

    # ------------------------------------------------------------------ URL proposals
    def _propose(self, service: str, provider: str, label: str = "") -> tuple[list[str], list[dict]]:
        if not self.llms:
            return [], []
        domains = self.urls.official_domains(provider)
        prompt = URL_PROMPT.format(service=label or service, provider=provider, domains=", ".join(domains) or "none",
                                   n=MAX_URLS)
        try:
            out = ask_text(self.llms[0], prompt)
        except Exception:                         # noqa: BLE001
            return [], []
        proposed = [u.strip().strip("<>") for u in re.findall(r"(?im)^\s*URL\s*:\s*(\S+)", out or "")]
        proposed = [u for u in proposed if u.lower() != "none"][:MAX_URLS]
        accepted, proposals = [], []
        for u in proposed:
            entry = {"url": u, "status": "refused"}
            if not self.urls.is_official(u, provider):
                entry["reason"] = "not an official domain of the provider"
            else:
                page = self._page(u)
                if page is None:
                    entry["reason"] = "unreachable"
                else:
                    res, _ = extract_service(self.llms, label or service, provider, {u: page}, self.vocabulary,
                                             self.chunk_chars)
                    anchored = [r for r in res if find_quote(r["quote"], page)]
                    if not anchored:
                        entry["reason"] = "no dependent resource whose quote appears in the page"
                    elif self.expert is None:
                        entry.update(status="to_validate", resources=[r["name"] for r in anchored])
                    elif self.expert(service, provider, u, anchored):
                        entry.update(status="validated")
                        self.urls.add_validated(service, provider, u)
                        accepted.append(u)
                    else:
                        entry.update(status="refused", reason="refused by the expert")
            proposals.append(entry)
        return accepted, proposals

    # ------------------------------------------------------------------ completion
    def complete(self, template: dict, bindings: dict) -> tuple[dict, list[Inserted]]:
        """bindings: node -> Offer (the selected composition). Returns the completed template and R."""
        plan = []
        for node, offer in bindings.items():
            k = (offer.service, offer.provider)
            if k not in self.states:
                label = f"{offer.name} ({offer.service})" if offer.name else offer.service
                self.states[k] = self._lookup(*k, label)
            st = self.states[k]
            plan.append({"node": node, "service": offer.service, "provider": offer.provider,
                         "region": offer.region, "resources": copy.deepcopy(st.resources),
                         "origin": "memory" if st.status in ("reused", "unreachable") else "extracted"})
        self.plan = plan
        return insert(template, plan, self.types)

    def complete_again(self, template: dict) -> tuple[dict, list[Inserted]]:
        """Insertion of the current plan (after reconfiguration or removal of failing resources)."""
        return insert(template, self.plan, self.types)

    def remove(self, node: str, name: str):
        """Removes a resource that is not anchored in its source page (an error of the extraction, independent
        of the template): from the plan, and from the resources to be written to the memory."""
        for item in self.plan:
            if item["node"] == node:
                item["resources"] = [r for r in item["resources"] if r["name"] != name]
                st = self.states.get((item["service"], item["provider"]))
                if st is not None:
                    st.resources = [r for r in st.resources if r["name"] != name]

    def reconfigure(self, r: Inserted, requirements: list) -> bool:
        """Sends a failing resource back to the LLM with the requirement it does not satisfy, so that it
        configures it again; the new properties override the settings of the provider. The result is
        verified again by the verification module (no repair is made by an algorithm)."""
        if not self.llms:
            return False
        page = self._page(r.url) or ""
        part = next((c for c in chunks(page, self.chunk_chars) if find_quote(r.quote, c)), page[: self.chunk_chars])
        st = self.states.get((r.service, r.provider))
        prompt = REPAIR_PROMPT.format(service=(st.label if st and st.label else r.service), provider=r.provider, resource=r.name, quote=r.quote,
                                      properties="; ".join(f"{k}={v}" for k, v in r.properties.items()) or "none",
                                      settings="; ".join(f"{k}={'/'.join(map(str, v))}" for k, v in r.settings.items()) or "none",
                                      requirement="; ".join(rq.text() for rq in requirements),
                                      page=part, vocabulary=self.vocabulary.describe())
        try:
            out = ask_text(self.llms[0], prompt) or ""
        except Exception:                         # noqa: BLE001
            return False
        m = re.search(r"(?im)^\s*PROPERTIES\s*:\s*(.*)$", out)
        new = self.vocabulary.parse(m.group(1)) if m else {}
        if not new:
            return False
        for item in self.plan:
            if item["node"] == r.node:
                for res in item["resources"]:
                    if res["name"] == r.name:
                        res["properties"] = {**(res.get("properties") or {}), **new}
        return True

    # ------------------------------------------------------------------ memory update (after verification)
    def commit(self):
        """Writes to the memory the resources extracted or found still valid, with their sources.
        A reconfiguration made for the policies of one template is not stored."""
        for (service, provider), st in self.states.items():
            if st.status in ("extracted", "changed", "reused") and st.sources:
                self.memory.put(service, provider, st.resources, st.sources)
                st.stored = True
        self.memory.save()

    def report(self) -> list[dict]:
        return [{"service": s.service, "provider": s.provider, "status": s.status, "stored": s.stored,
                 "resources": [r["name"] for r in s.resources], "unreachable": s.unreachable,
                 "unreachable_reasons": {u: self.fetch_errors.get(u, "") for u in s.unreachable},
                 "proposals": s.proposals, "extraction": s.extraction} for s in self.states.values()]
