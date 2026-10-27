"""
Scripted LLMs for the tests (no API key): they answer the prompts of the approach from a given knowledge,
and can be made to err (omission, hallucination, altered quote, region copied from an example, ...).
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from ..completion.text import find_quote


def _service(prompt: str) -> str:
    """Service key in a prompt: "Service: EC2 m5.large (compute)" -> "compute"."""
    s = _between(prompt, "Service: ", "\n").strip()
    if s.endswith(")") and "(" in s:
        return s[s.rindex("(") + 1:-1].strip()
    return s


def _between(prompt: str, start: str, end: str) -> str:
    i = prompt.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = prompt.find(end, i)
    return prompt[i:j if j >= 0 else None]


def _fmt(d: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in (d or {}).items()) or "none"


class FakeLLM:
    def __init__(self, name: str = "fake", knowledge: Optional[list[dict]] = None, omit=(), hallucinate=(),
                 alter_quotes: bool = False, property_overrides: Optional[dict] = None,
                 repair: Optional[dict] = None, urls: Optional[dict] = None,
                 phrase: Optional[Callable[[str], str]] = None, interpret: Optional[Callable[[str], str]] = None,
                 fail: bool = False):
        self.name = name
        self.knowledge = list(knowledge or [])       # [{name, quote, properties, settings, attached_to, service?, provider?}]
        self.omit = set(omit)
        self.hallucinate = list(hallucinate)         # resources output whatever the page
        self.alter_quotes = alter_quotes
        self.property_overrides = property_overrides or {}   # {resource name: {prop: value}}
        self.repair = repair or {}                   # {resource name: "prop=value; ..."}
        self.urls = urls or {}                       # {(service, provider): [urls]}
        self.phrase = phrase
        self.interpret = interpret
        self.fail = fail
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("LLM unavailable")
        if "Identify, in this page, all the dependent resources" in prompt:
            return self._extract(prompt)
        if "Propose the official documentation pages" in prompt:
            service = _service(prompt)
            provider = _between(prompt, "Provider: ", "\n").strip()
            us = self.urls.get((service, provider), [])
            return "\n".join(f"URL: {u}" for u in us) or "URL: none"
        if "Configure this resource again" in prompt:
            res = _between(prompt, "Resource: ", "\n").strip()
            return f"PROPERTIES: {self.repair.get(res, 'none')}"
        if "Present these options" in prompt:
            ctx = _between(prompt, "with their QoS values:\n", "\n\nTask (iii)")
            if self.phrase:
                return self.phrase(ctx)
            lines = [f"Option {l}" for l in ctx.strip().split("\n")]
            return "MESSAGE: Here are your options.\n" + "\n".join(lines) + "\nWhich option do you prefer?"
        if "Identify, in the user's answer" in prompt:
            ans = _between(prompt, "selected option or the stated priority: ", "\n\nInstructions").strip()
            if self.interpret:
                return self.interpret(ans)
            low = ans.lower()
            for crit in ("cost", "availability"):
                if crit in low or (crit == "cost" and ("cheap" in low or "price" in low)):
                    return f"OPTION: none\nPRIORITY: {crit}\nUNCLEAR: false"
            m = re.search(r"\boption ([a-z])\b", low)
            if m:
                return f"OPTION: {m.group(1).upper()}\nPRIORITY: none\nUNCLEAR: false"
            return "OPTION: none\nPRIORITY: none\nUNCLEAR: true"
        return ""

    def _extract(self, prompt: str) -> str:
        page = _between(prompt, "Page: ", "\n\nTask (iii)")
        service = _service(prompt)
        provider = _between(prompt, "Provider: ", "\n").strip()
        out = []
        for k in self.knowledge:
            if k["name"] in self.omit or not find_quote(k["quote"], page):
                continue
            if k.get("service") and (k["service"], k.get("provider")) != (service, provider):
                continue
            q = k["quote"] + " (sic)" if self.alter_quotes else k["quote"]
            props = {**(k.get("properties") or {}), **self.property_overrides.get(k["name"], {})}
            out.append(self._block(k["name"], q, k.get("attached_to") or "the service", props, k.get("settings")))
        for h in self.hallucinate:
            out.append(self._block(h["name"], h.get("quote", "This resource is invented by the model."),
                                   h.get("attached_to") or "the service", h.get("properties"), h.get("settings")))
        return "\n".join(out) if out else "RESOURCE: none"

    @staticmethod
    def _block(name, quote, attached, props, settings) -> str:
        return (f"RESOURCE: {name}\nQUOTE: {quote}\nATTACHED TO: {attached}\n"
                f"PROPERTIES: {_fmt(props)}\nSETTINGS: {_fmt(settings)}\n")
