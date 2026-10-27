"""
Negotiated selection phase as a state machine (Sections 5.1 and 5.2), usable from a console, a web
interface, or a simulated user.

    s = NegotiationSession(problem, llm)
    ev = s.start()
    while isinstance(ev, Ask):
        ev = s.reply(user(ev.message))
    # ev is Done: ev.status in {"selected", "no_agreement"}

Rules
  - the solver computes every alternative; the LLM only phrases them and interprets the answers;
  - an unclear answer: the question is asked again;
  - the decision extracted from the answer is applied only once the user confirms it;
  - a selected relaxation option updates the corresponding constraints (e.g., the budget becomes 191 USD),
    a selected trade-off option fixes the composition, and a stated priority is recorded for the remainder
    of the phase, so that it is never asked again;
  - after a relaxation, the solver is called again on the updated constraints; several compositions may
    remain non-dominated, in which case a trade-off is negotiated in turn;
  - the number of rounds (questions on a proposal) is bounded; beyond it, the phase ends without agreement;
    a confirmation that stays unclear (MAX_UNCLEAR_CONFIRMATIONS answers) presents the proposal again, so that
    every exchange counts towards the bound and the phase always ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from ..selection.problem import Problem, Composition, COST, AVAILABILITY, objectives
from ..selection.select import select, Selected, priority_spec
from ..selection.cpsat import solve
from .negotiator import Negotiator
from .proposal import Proposal, ProposalOption, build_proposal
from .templates import confirmation_message

MAX_ROUNDS = 3
MAX_UNCLEAR_CONFIRMATIONS = 2      # then the proposal is presented again, which counts as a round


@dataclass
class Ask:
    message: str
    awaiting: str                      # "decision" | "confirmation"
    source: str = "predefined"         # how the message was phrased: "llm" | "predefined"


@dataclass
class Done:
    status: str                        # "selected" | "no_agreement"
    composition: Optional[Composition]
    problem: Problem                   # constraints after the negotiation
    decisions: list = field(default_factory=list)
    log: list = field(default_factory=list)
    cost_cents: int = 0
    availability: float = 0.0


Event = Union[Ask, Done]


class NegotiationSession:
    def __init__(self, problem: Problem, llm=None, max_rounds: int = MAX_ROUNDS, priority: Optional[list[str]] = None):
        self.problem = problem
        self.negotiator = Negotiator(llm)
        self.max_rounds = max_rounds
        self.priority: Optional[list[str]] = list(priority) if priority else None
        self.rounds = 0
        self.decisions: list[dict] = []
        self.log: list[dict] = []
        self.proposal: Optional[Proposal] = None
        self.pending: Optional[tuple[ProposalOption, str]] = None   # (option, priority) awaiting confirmation
        self.unclear_confirmations = 0

    # ------------------------------------------------------------------
    def start(self) -> Event:
        return self._run_selection()

    def reply(self, answer: str) -> Event:
        self.log.append({"round": self.rounds, "user": answer})
        if self.pending is not None:
            ok = self.negotiator.confirm(answer)
            if ok is None:
                self.unclear_confirmations += 1
                if self.unclear_confirmations < MAX_UNCLEAR_CONFIRMATIONS:
                    return self._ask(confirmation_message(self.proposal, *self.pending), "confirmation")
                ok = False                                    # still unclear: the proposal is presented again
            self.unclear_confirmations = 0
            option, pri = self.pending
            self.pending = None
            if not ok:
                return self._present()                       # the question is asked again
            return self._apply(option, pri)
        interp = self.negotiator.interpret(answer, self.proposal)
        self.log.append({"round": self.rounds, "interpretation": interp.__dict__})
        if interp.unclear:
            return self._present()
        if interp.option:
            option = self.proposal.option(interp.option)
            self.pending = (option, "")
        else:
            option = self._option_for_priority(interp.priority)
            self.pending = (option, interp.priority)
        return self._ask(confirmation_message(self.proposal, *self.pending), "confirmation")

    # ------------------------------------------------------------------
    def _run_selection(self) -> Event:
        signal = select(self.problem, self.priority)
        if isinstance(signal, Selected):
            return Done("selected", signal.composition, self.problem, self.decisions, self.log,
                        signal.cost_cents, signal.availability)
        self.proposal = build_proposal(signal, self.problem)
        return self._present()

    def _present(self) -> Event:
        if self.rounds >= self.max_rounds:
            return Done("no_agreement", None, self.problem, self.decisions, self.log)
        self.rounds += 1
        msg, source = self.negotiator.phrase(self.proposal)
        return self._ask(msg, "decision", source)

    def _ask(self, message: str, awaiting: str, source: str = "predefined") -> Ask:
        self.log.append({"round": self.rounds, "system": message, "awaiting": awaiting, "source": source})
        return Ask(message, awaiting, source)

    def _option_for_priority(self, priority: str) -> ProposalOption:
        """Option that best serves the stated priority (then the other criterion, then the letter)."""
        def key(o: ProposalOption):
            c, a = o.cost_cents, -o.availability
            return (c, a, o.letter) if priority == COST else (a, c, o.letter)
        return min(self.proposal.options, key=key)

    def _apply(self, option: ProposalOption, priority: str) -> Event:
        nodes = self.problem.node_names
        if priority:
            self.priority = [priority] + [c for c in (COST, AVAILABILITY) if c != priority]
        if option.relaxation is not None:
            self.decisions.append({"round": self.rounds, "type": "relaxation", "option": option.letter,
                                   "priority": priority or None,
                                   "changes": [ch.__dict__ for ch in option.relaxation.changes]})
            self.problem = self.problem.with_constraints(option.relaxation.constraints)
            return self._run_selection()
        self.decisions.append({"round": self.rounds, "type": "priority" if priority else "trade_off",
                               "option": option.letter, "priority": priority or None})
        c, a = objectives(option.composition, nodes, self.problem.groups)
        return Done("selected", option.composition, self.problem, self.decisions, self.log, c, a)


def run_with_user(session: NegotiationSession, user) -> Done:
    """Drives a session with a function user(message, awaiting) -> answer."""
    ev = session.start()
    while isinstance(ev, Ask):
        ev = session.reply(user(ev.message, ev.awaiting))
    return ev
