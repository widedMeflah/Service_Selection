"""
End-to-end approach (Fig. 1): negotiated selection, then verified completion of dependent resources.

    result = run(tosca, discovery, user=..., negotiation_llm=..., extraction_llms=[llm1, llm2, llm3], ...)

Negotiated selection    Constraint-Based Selection <-> Negotiation with the User (session.py)
Verified completion     Documentation-Grounded Completion <-> Algorithmic Verification, at most
                        `verification_rounds` rounds: a resource not anchored in its page is removed, a resource
                        that violates a policy is configured again by the LLM; after the last round the remaining
                        failures are reported with the template, and the memory is not updated.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from .completion.completer import Completer, TTL_DAYS
from .completion.extraction import CHUNK_CHARS
from .completion.fetch import HttpFetcher
from .completion.memory import Memory
from .completion.urls import UrlList
from .negotiation.session import NegotiationSession, Ask, Done, MAX_ROUNDS
from .output import write_selection
from .selection.builder import build_problem
from .verification.verifier import verify, ANCHORING
from .verification.vocabulary import Vocabulary

VERIFICATION_ROUNDS = 3


@dataclass
class Result:
    status: str                          # completed | completed_with_failures | no_agreement | awaiting_user
    template: Optional[dict] = None      # provider-specific template
    selection: Optional[Done] = None
    completion: list = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    pending: Optional[Ask] = None


def run(tosca: dict, discovery: dict, *, user: Optional[Callable] = None, negotiation_llm=None,
        extraction_llms: Optional[list] = None, expert: Optional[Callable] = None, fetcher=None,
        urls: Optional[UrlList] = None, memory: Optional[Memory] = None, vocabulary: Optional[Vocabulary] = None,
        max_rounds: int = MAX_ROUNDS, verification_rounds: int = VERIFICATION_ROUNDS, priority=None,
        ttl_days: int = TTL_DAYS, chunk_chars: int = CHUNK_CHARS, session: Optional[NegotiationSession] = None,
        selection_done: Optional[Done] = None) -> Result:
    urls = urls or UrlList.load()
    fetcher = fetcher or HttpFetcher(urls)
    memory = memory if memory is not None else Memory()
    vocabulary = vocabulary or Vocabulary()
    problem = build_problem(tosca, discovery)

    # ---------------------------------------------------------------- negotiated selection
    done = selection_done
    if done is None:
        session = session or NegotiationSession(problem, negotiation_llm, max_rounds, priority)
        ev = session.start()
        while isinstance(ev, Ask):
            if user is None:
                return Result("awaiting_user", pending=ev)
            ev = session.reply(user(ev.message, ev.awaiting))
        done = ev
    if done.status != "selected":
        return Result("no_agreement", selection=done)
    selected = write_selection(tosca, done.problem, done.composition, done.decisions)

    # ---------------------------------------------------------------- verified completion
    completer = Completer(extraction_llms or [], fetcher, urls, memory, vocabulary, expert, ttl_days, chunk_chars)
    template, inserted = completer.complete(selected, done.composition)
    initial = [dataclasses.asdict(r) for r in inserted]          # before any verification (ablation)
    service_nodes = set(done.problem.node_names)
    rounds, history, failures = 0, [], []
    while True:
        rounds += 1
        failures = verify(template, inserted, urls, fetcher, vocabulary, service_nodes)
        history.append([f.as_dict() for f in failures])
        if not failures:
            completer.commit()
            break
        if rounds >= verification_rounds:
            break
        for f in failures:                       # sent back to the completion module, with the verification
            if f.verification == ANCHORING:
                completer.remove(f.resource.node, f.resource.name)
        for f in failures:
            if f.verification != ANCHORING:
                completer.reconfigure(f.resource, f.requirements)
        template, inserted = completer.complete_again(selected)
    report = completer.report()
    incomplete = [c for c in report if c["status"] not in ("extracted", "changed", "reused", "to_validate")]
    status = "completed" if not failures and not incomplete else "completed_with_failures"
    verification = {"status": "verified" if not failures else "failures_reported", "rounds": rounds,
                    "failures": [f.as_dict() for f in failures], "history": history,
                    "memory_updated": any(c["stored"] for c in report),
                    "initial_inserted": initial, "final_inserted": [dataclasses.asdict(r) for r in inserted]}
    template.setdefault("metadata", {})["verification"] = verification
    template["metadata"]["completion"] = report
    return Result(status, template, done, report, verification)
