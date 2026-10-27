"""
Algorithmic Verification module (Section 6.2, Algorithm 1). A deterministic algorithm, no LLM; it never
repairs the template.

for each inserted dependent resource r:
    P <- policies that target the node of r                                   (line 3)
    anchoring            r.url belongs to an official domain of the provider of its node, and r.quote
                         appears word for word in the current page                          (line 4)
    policy compliance    the properties of r, written in the template, satisfy P             (line 6)
    non-contradiction    the settings that the provider applies to r, according to its source page and
                         not overridden by its properties, contradict no policy of P          (line 8)
if no resource fails: the memory is updated and the template is returned       (lines 10-12)
otherwise: the failing resources are returned, each with the verification it fails (line 13)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..completion.fetch import FetchError
from ..completion.insertion import Inserted
from ..completion.text import find_quote
from .vocabulary import Vocabulary, Requirement

ANCHORING, COMPLIANCE, NON_CONTRADICTION = "anchoring", "policy compliance", "non-contradiction"


@dataclass
class Failure:
    resource: Inserted
    verification: str
    detail: str = ""
    requirements: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"resource": self.resource.name, "node": self.resource.node, "id": self.resource.id,
                "verification": self.verification, "detail": self.detail,
                "requirements": [r.text() for r in self.requirements]}


def anchored(r: Inserted, urls, fetcher) -> tuple[bool, str]:
    if not r.url or not urls.is_official(r.url, r.provider):
        return False, f"source {r.url or '(none)'} is not an official page of {r.provider}"
    try:
        page = fetcher.fetch(r.url)
    except FetchError as exc:
        return False, f"source page cannot be read: {exc}"
    if not find_quote(r.quote, page):
        return False, "quote not found word for word in the current page"
    return True, ""


def verify(template: dict, inserted: list[Inserted], urls, fetcher, vocabulary: Vocabulary,
           service_nodes: set[str]) -> list[Failure]:
    failures: list[Failure] = []
    for r in inserted:
        P: list[Requirement] = vocabulary.requirements(template, r.node, service_nodes)          # line 3
        ok, why = anchored(r, urls, fetcher)                                                       # line 4
        if not ok:
            failures.append(Failure(r, ANCHORING, why))
        bad = vocabulary.violations(r.properties, P, r.region)                                     # line 6
        if bad:
            failures.append(Failure(r, COMPLIANCE, "properties violate a policy", bad))
        effective = {k: v for k, v in r.settings.items() if k not in r.properties}
        bad = vocabulary.violations(effective, P, r.region)                                        # line 8
        if bad:
            failures.append(Failure(r, NON_CONTRADICTION, "a setting of the provider contradicts a policy", bad))
    return failures
