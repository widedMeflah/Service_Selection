"""
Ablation "LLM verifier": the algorithmic verification (src/verification/verifier.py) replaced by one LLM that receives
the same inputs: the policies of each service node, the official domains of each provider, every inserted dependent
resource (source URL, quote, properties, settings of the provider) and the current text of each source page (or the
fact that the page cannot be retrieved). The LLM returns the failing resources with the verification they fail; the
removal and reconfiguration that follow are those of our approach.
"""
from __future__ import annotations

import re

from ..completion.fetch import FetchError
from ..llm_call import ask_text
from ..verification.verifier import Failure, ANCHORING, COMPLIANCE, NON_CONTRADICTION

LLM_VERIFIER_PROMPT = """Role (i)
You are an auditor of cloud deployment templates.

Context (ii)
Considering the following dependent resources, inserted in a deployment template for the cloud service nodes they
depend on, the policies of these nodes, the official documentation domains of each provider, and the current text
of each source page:

Policies of each service node:
{policies}

Official documentation domains of each provider:
{domains}

Inserted dependent resources:
{resources}

Source pages:
{pages}

Task (iii)
Find the inserted dependent resources that fail one of the three verifications below.

Instructions (iv)
Follow these instructions:
- anchoring: the source URL of the resource belongs to an official documentation domain of the provider of its node,
  the source page can be retrieved, and the quote of the resource appears word for word in that page.
- policy compliance: the properties of the resource, written in the template, satisfy every policy of its node.
- non-contradiction: the settings that the provider applies to the resource, according to its source page and not
  overridden by its properties, contradict no policy of its node.
- a resource shared by several nodes appears once for each node; verify it for each node, against the policies of
  that node.
- report a resource once for each verification it fails, and report only the resources that fail a verification.

Output format (v)
One line for each failure:
FAILURE: <reference of the resource> | <anchoring, policy compliance or non-contradiction> | <reason>
or, if no resource fails:
FAILURES: none"""

CHECKS = {"anchoring": ANCHORING, "policy compliance": COMPLIANCE, "non-contradiction": NON_CONTRADICTION,
          "non contradiction": NON_CONTRADICTION, "compliance": COMPLIANCE}


def _fmt(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items()) if d else "none"


def build_prompt(template, inserted, urls, fetcher, vocabulary, service_nodes) -> tuple[str, dict]:
    nodes = sorted({r.node for r in inserted})
    reqs = {n: vocabulary.requirements(template, n, service_nodes) for n in nodes}
    policies = "\n".join(f"- {n}: " + ("; ".join(q.text() for q in reqs[n]) or "no policy") for n in nodes)
    providers = sorted({r.provider for r in inserted})
    domains = "\n".join(f"- {p}: {', '.join(urls.official_domains(p)) or 'none'}" for p in providers)
    resources = "\n".join(
        f"- reference: R{i}\n  node: {r.node}\n  name: {r.name}\n  provider: {r.provider}\n  region: {r.region}\n"
        f"  source URL: {r.url or 'none'}\n  quote: {r.quote or 'none'}\n  properties: {_fmt(r.properties)}\n"
        f"  settings of the provider: {_fmt(r.settings)}" for i, r in enumerate(inserted, 1))
    pages = []
    for u in sorted({r.url for r in inserted if r.url}):
        try:
            text = fetcher.fetch(u)
        except FetchError:
            text = "(this page cannot be retrieved)"
        pages.append(f"=== {u}\n{text}")
    prompt = LLM_VERIFIER_PROMPT.format(policies=policies, domains=domains, resources=resources,
                                        pages="\n\n".join(pages) or "none")
    return prompt, reqs


def parse(out: str, inserted, reqs) -> tuple[list[Failure], list[str]]:
    by_ref = {f"R{i}": r for i, r in enumerate(inserted, 1)}
    failures, unknown, seen = [], [], set()
    for m in re.finditer(r"(?im)^\W*FAILURE\s*:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.*)$", out or ""):
        rid, check, why = m.group(1).strip(" `*[]<>"), m.group(2).strip(" `*").lower(), m.group(3).strip()
        r = by_ref.get(rid.upper())
        v = CHECKS.get(check)
        if r is None or v is None:
            unknown.append(m.group(0).strip())
            continue
        if (rid, v) in seen:
            continue
        seen.add((rid, v))
        # the requirements given to the reconfiguration: the policies of the node (the LLM gives no structured one)
        failures.append(Failure(r, v, why, [] if v == ANCHORING else list(reqs.get(r.node, []))))
    return failures, unknown


def llm_verify(llm, template, inserted, urls, fetcher, vocabulary, service_nodes) -> tuple[list[Failure], dict]:
    prompt, reqs = build_prompt(template, inserted, urls, fetcher, vocabulary, service_nodes)
    out = ask_text(llm, prompt)
    failures, unknown = parse(out, inserted, reqs)
    return failures, {"prompt_chars": len(prompt), "output": out, "unparsed_lines": unknown}
