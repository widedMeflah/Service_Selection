"""
Runs the experiments of the evaluation and logs every result needed by the metrics (no metric is computed here).

    python run_eval.py --frozen data/eval/snapshots/ --run run1                  # every phase, in order
    python run_eval.py --frozen data/eval/snapshots/ --run run1 --phases selection completion
    python run_eval.py --frozen data/eval/snapshots/ --run dry --dry-run         # the whole chain with fake LLMs, no cost
    python run_eval.py ... --cases A-several-07m-anywhere B-aws-vm     # only these cases (subset)

The LLMs are those of eval_config.json (to be decided before the runs). Every unit of work (a case, a method and an
LLM) writes its result to runs/<run>/<phase>/...json as soon as it ends: an interrupted run is resumed by the same
command, and a unit that failed (e.g. a network error) is run again. Nothing is shared between runs except the
dataset: run the command three times with --run run1, run2, run3 for the three executions of our approach.

Phases (in this order)
  selection     our approach, negotiated selection, every case (twins included); the simulated user answers
  completion    our approach, verified completion of the selected composition, every case except the twins,
                in the order of the manifest, with a memory empty at the start of the phase; two variants,
                completion/vote (majority vote of extraction_llms) and completion/single (single_llm only)
  verification  family C: verification of the injected errors, then the referral loop (removal, reconfiguration)
  staleness     the staleness cases: memory filled with the annotated resources, edited page, lookup of the service
  minimal       ablation "minimal version": dependent resources from the knowledge of one LLM, no page
  baselines     single-shot, cot, rag, self-refine (each baseline LLM), react (each react LLM), and the reference
                methods (single-shot) of the strong LLMs alone (reference_llms); one thread per LLM
  cited         copy of every official page cited by a method outside the URL list (checked later, offline)
  scalability   solving time of the selection on generated problems of increasing size (no LLM)
  verification_llm  ablation: family C with the verification made by one LLM (verifier_llm) instead of the
                algorithm; same removal and reconfiguration; only with --phases verification_llm
Logged for every unit: inputs reference, raw outputs of every LLM call, parsed answer, statuses, wall-clock time,
time spent waiting for the simulated user, and per LLM call the model, input and output tokens, latency, success.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import yaml

from src.completion.completer import Completer, ServiceState
from src.completion.extraction import CHUNK_CHARS
from src.completion.fetch import FrozenFetcher, HttpFetcher, DictFetcher, FetchError
from src.completion.insertion import Inserted
from src.completion.memory import Memory
from src.completion.text import fingerprint, find_quote
from src.completion.urls import UrlList
from src.eval import baselines as B
from src.eval.simulated_user import LLMUser, StructuredUser
from src.llm import create_llm, EXTRACTION_LLMS, NEGOTIATION_LLM
from src.llm_call import set_tag, take_usage
from src.negotiation.session import NegotiationSession, Ask, MAX_ROUNDS
from src.pipeline import run as run_pipeline, VERIFICATION_ROUNDS
from src.selection.builder import build_problem
from src.selection.problem import Problem, Node, Offer, QoSConstraint, COST, AVAILABILITY
from src.selection.select import select
from src.verification.verifier import verify, ANCHORING
from src.eval.llm_verifier import llm_verify
from src.verification.vocabulary import Vocabulary

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT, "data", "eval", "dataset")
CONFIG = os.path.join(ROOT, "eval_config.json")
PHASES = ("selection", "completion", "verification", "staleness", "minimal", "baselines", "cited", "scalability",
          "verification_llm")
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------------------------- files
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def done_ok(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            return "error" not in json.load(f)
    except (OSError, json.JSONDecodeError):
        return False


def log(run_dir: str, msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    with _LOCK:
        print(line, flush=True)
        with open(os.path.join(run_dir, "run.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_case(cid: str):
    folder = os.path.join(DATASET, "cases", cid)
    t = yaml.safe_load(open(os.path.join(folder, "template.tosca.yaml"), encoding="utf-8"))
    d = json.load(open(os.path.join(folder, "discovery.json"), encoding="utf-8"))
    c = json.load(open(os.path.join(folder, "case.json"), encoding="utf-8"))
    return t, d, c


def manifest(only=None) -> list[dict]:
    m = json.load(open(os.path.join(DATASET, "manifest.json"), encoding="utf-8"))
    return [c for c in m if not only or c["id"] in only]


# ---------------------------------------------------------------------------------------------- LLMs
class FakeLLM:
    """Answers of the right format for every prompt of the project (dry run: the whole chain, no cost)."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, prompt: str) -> str:
        if "requirements analyst" in prompt:
            return "OPTION: A\nPRIORITY: none\nUNCLEAR: false"
        if "You play a user" in prompt:
            return "ANSWER: yes" if "confirm" in prompt.lower() else "ANSWER: I take option A."
        if "cloud consultant" in prompt:
            return ""                                   # the check rejects it: the predefined message is used
        if "using tools" in prompt:
            return 'Thought: done.\nAction: finish[{"feasible": true, "selection": {}, "dependent_resources": []}]'
        if "cloud architect" in prompt:
            return 'ANSWER: {"feasible": true, "selection": {}, "violated_policies": [], "dependent_resources": []}'
        if "configures again" in prompt or "PROPERTIES:" in prompt and "Requirement" in prompt:
            return "PROPERTIES: internet_exposure=false"
        if "auditor of cloud deployment templates" in prompt:
            return "FAILURES: none"
        if "from your knowledge" in prompt:
            return "RESOURCE: VPC\nURL: https://docs.aws.amazon.com/vpc/\nQUOTE: none\nATTACHED TO: the service"
        return "RESOURCE: none"


def make_llm(key: str, dry: bool):
    if dry:
        return FakeLLM(f"fake-{key}")
    llm = create_llm(key)
    if llm is None:
        raise ValueError(f"no LLM for {key}")
    return llm


def load_config(path: str) -> dict:
    cfg = {"extraction_llms": EXTRACTION_LLMS, "single_llm": EXTRACTION_LLMS[0], "negotiation_llm": NEGOTIATION_LLM,
           "user": "deterministic", "baseline_llms": [EXTRACTION_LLMS[0]], "baselines": list(B.METHODS[:-1]),
           "react_llms": [EXTRACTION_LLMS[0]], "reference_llms": [], "reference_methods": ["single-shot"],
           "minimal_llm": EXTRACTION_LLMS[0], "reconfigure_llm": EXTRACTION_LLMS[0], "verifier_llm": EXTRACTION_LLMS[0],
           "max_rounds": MAX_ROUNDS, "verification_rounds": VERIFICATION_ROUNDS}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    return cfg


def usage_summary(calls: list[dict]) -> dict:
    return {"calls": len(calls), "failed_calls": sum(not c["ok"] for c in calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "llm_seconds": round(sum(c["seconds"] for c in calls), 2)}


# ---------------------------------------------------------------------------------------------- serialization
def comp_json(x) -> dict:
    return {n: {"offer_id": o.offer_id, "provider": o.provider, "region": o.region,
                "price_cents": o.price_cents, "availability": o.availability} for n, o in (x or {}).items()}


def constraint_json(c) -> dict:
    if isinstance(c, QoSConstraint):
        return {"id": c.id, "criterion": c.criterion, "targets": list(c.targets), "bound": c.bound}
    return {"id": c.id, "attribute": c.attribute, "targets": list(c.targets)}


# ---------------------------------------------------------------------------------------------- selection
def phase_selection(ctx, case):
    cid = case["id"]
    out_json = os.path.join(ctx["dir"], "selection", f"{cid}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    if done_ok(out_json):
        return
    t, d, c = load_case(cid)
    priority = c["ground_truth"]["priority"]
    set_tag(f"selection/{cid}")
    take_usage()
    holder = {}
    proposal_fn = lambda: getattr(holder.get("session"), "proposal", None)      # noqa: E731
    user = LLMUser(ctx["user_llm"], priority, proposal_fn) if ctx["user_llm"] is not None \
        else StructuredUser(priority, proposal_fn)
    timed = {"user_seconds": 0.0}

    def answer(message, awaiting):
        t0 = time.time()
        try:
            return user(message, awaiting)
        finally:
            timed["user_seconds"] += time.time() - t0

    t0 = time.time()
    problem = build_problem(t, d)
    first = select(problem)
    session = NegotiationSession(problem, ctx["negotiation_llm"], ctx["cfg"]["max_rounds"])
    holder["session"] = session
    ev = session.start()
    turns, max_turns = 0, 4 * (ctx["cfg"]["max_rounds"] + 1)      # safety bound on the exchanges of one case
    while isinstance(ev, Ask):
        turns += 1
        if turns > max_turns:
            raise RuntimeError(f"negotiation not ended after {max_turns} answers of the user")
        ev = session.reply(answer(ev.message, ev.awaiting))
    wall = time.time() - t0
    calls = take_usage()
    with open(os.path.join(ctx["dir"], "selection", f"{cid}.pkl"), "wb") as f:
        pickle.dump(ev, f)
    proposal = session.proposal
    write_json(out_json, {
        "case": cid, "finished_at": now(), "status": ev.status, "first_signal": first.kind,
        "composition": comp_json(ev.composition), "cost_cents": ev.cost_cents, "availability": ev.availability,
        "constraints_after": [constraint_json(k) for k in ev.problem.constraints],
        "decisions": ev.decisions, "rounds": session.rounds, "dialogue": ev.log,
        "rejected_phrasings": session.negotiator.rejected,
        "last_proposal": None if proposal is None else {
            "kind": proposal.kind,
            "options": [{"letter": o.letter, "composition": comp_json(o.composition), "cost_cents": o.cost_cents,
                         "availability": o.availability, "availability_known": o.availability_known,
                         "relaxation": None if o.relaxation is None else [ch.__dict__ for ch in o.relaxation.changes]}
                        for o in proposal.options]},
        "user": {"kind": "llm" if ctx["user_llm"] is not None else "deterministic", "priority": priority,
                 "turns": getattr(user, "turns", [])},
        "time": {"wall_seconds": round(wall, 3), "user_seconds": round(timed["user_seconds"], 3),
                 "without_user_seconds": round(wall - timed["user_seconds"], 3)},
        "usage": usage_summary(calls), "llm_calls": calls})


# ---------------------------------------------------------------------------------------------- completion
def phase_completion(ctx, case, memory, variant="vote", llms=None):
    cid = case["id"]
    out_json = os.path.join(ctx["dir"], "completion", variant, f"{cid}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    if done_ok(out_json):
        return
    pkl = os.path.join(ctx["dir"], "selection", f"{cid}.pkl")
    if not os.path.exists(pkl):
        write_json(out_json, {"case": cid, "skipped": "no selection result (run the selection phase first)"})
        return
    with open(pkl, "rb") as f:
        done = pickle.load(f)
    if done.status != "selected":
        write_json(out_json, {"case": cid, "skipped": f"selection status {done.status}"})
        return
    t, d, c = load_case(cid)
    set_tag(f"completion/{variant}/{cid}")
    take_usage()
    t0 = time.time()
    r = run_pipeline(t, d, extraction_llms=llms or ctx["extraction_llms"], fetcher=ctx["fetcher"], urls=ctx["urls"],
                     memory=memory, selection_done=done, verification_rounds=ctx["cfg"]["verification_rounds"])
    wall = time.time() - t0
    calls = take_usage()
    if r.template is not None:
        with open(os.path.join(ctx["dir"], "completion", variant, f"{cid}.tosca.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump(r.template, sort_keys=False, allow_unicode=True, width=110))
    write_json(out_json, {"case": cid, "variant": variant, "finished_at": now(), "status": r.status,
                          "completion": r.completion,
                          "verification": r.verification, "time": {"wall_seconds": round(wall, 3)},
                          "usage": usage_summary(calls), "llm_calls": calls})


# ---------------------------------------------------------------------------------------------- family C
def phase_verification(ctx, folder):
    cid = os.path.basename(folder)
    out_json = os.path.join(ctx["dir"], "verification", f"{cid}.json")
    if done_ok(out_json):
        return
    meta = json.load(open(os.path.join(folder, "case.json"), encoding="utf-8"))
    template = yaml.safe_load(open(os.path.join(folder, "template.tosca.yaml"), encoding="utf-8"))
    inserted = [Inserted(**r) for r in json.load(open(os.path.join(folder, "inserted.json"), encoding="utf-8"))]
    base, _, bcase = load_case(meta["from"])
    services = {s["node"] for s in bcase["completion_services"]}
    set_tag(f"verification/{cid}")
    take_usage()
    t0 = time.time()
    # the plan of the completer, rebuilt from the inserted resources (ids -> names for the attachments)
    names = {(r.node, r.id): r.name for r in inserted}
    plan, states = [], {}
    for s in bcase["completion_services"]:
        rs = [r for r in inserted if r.node == s["node"]]
        plan.append({"node": s["node"], "service": s["service"], "provider": s["provider"], "region": s["region"],
                     "origin": "annotation",
                     "resources": [{"name": r.name, "quote": r.quote, "url": r.url, "properties": dict(r.properties),
                                    "settings": dict(r.settings),
                                    "attached_to": names.get((r.node, r.attached_to))} for r in rs]})
        states[(s["service"], s["provider"])] = ServiceState(s["service"], s["provider"], "extracted",
                                                             plan[-1]["resources"], label=s["service"])
    completer = Completer([ctx["reconfigure_llm"]], ctx["fetcher"], ctx["urls"], Memory(None), ctx["vocabulary"])
    completer.states, completer.plan = states, plan
    rounds, history, failures = 0, [], []
    current = (template, inserted)
    while True:
        rounds += 1
        failures = verify(current[0], current[1], ctx["urls"], ctx["fetcher"], ctx["vocabulary"], services)
        history.append([f.as_dict() for f in failures])
        if not failures or rounds >= ctx["cfg"]["verification_rounds"]:
            break
        actions = []
        for f in failures:
            if f.verification == ANCHORING:
                completer.remove(f.resource.node, f.resource.name)
                actions.append({"resource": f.resource.name, "node": f.resource.node, "action": "removed"})
        for f in failures:
            if f.verification != ANCHORING:
                ok = completer.reconfigure(f.resource, f.requirements)
                actions.append({"resource": f.resource.name, "node": f.resource.node,
                                "action": "reconfigured" if ok else "reconfiguration_failed"})
        history[-1] = {"failures": history[-1], "actions": actions}
        current = completer.complete_again(base)
    calls = take_usage()
    write_json(out_json, {"case": cid, "finished_at": now(), "expected": meta["expected_failures"],
                          "rounds": rounds, "history": history, "remaining_failures": [f.as_dict() for f in failures],
                          "time": {"wall_seconds": round(time.time() - t0, 3)}, "usage": usage_summary(calls),
                          "llm_calls": calls})


def phase_verification_llm(ctx, folder):
    """Ablation: phase_verification with the verification made by one LLM (src/eval/llm_verifier.py)."""
    cid = os.path.basename(folder)
    out_json = os.path.join(ctx["dir"], "verification_llm", f"{cid}.json")
    if done_ok(out_json):
        return
    meta = json.load(open(os.path.join(folder, "case.json"), encoding="utf-8"))
    template = yaml.safe_load(open(os.path.join(folder, "template.tosca.yaml"), encoding="utf-8"))
    inserted = [Inserted(**r) for r in json.load(open(os.path.join(folder, "inserted.json"), encoding="utf-8"))]
    base, _, bcase = load_case(meta["from"])
    services = {s["node"] for s in bcase["completion_services"]}
    set_tag(f"verification_llm/{cid}")
    take_usage()
    t0 = time.time()
    # the plan of the completer, rebuilt from the inserted resources (ids -> names for the attachments)
    names = {(r.node, r.id): r.name for r in inserted}
    plan, states = [], {}
    for s in bcase["completion_services"]:
        rs = [r for r in inserted if r.node == s["node"]]
        plan.append({"node": s["node"], "service": s["service"], "provider": s["provider"], "region": s["region"],
                     "origin": "annotation",
                     "resources": [{"name": r.name, "quote": r.quote, "url": r.url, "properties": dict(r.properties),
                                    "settings": dict(r.settings),
                                    "attached_to": names.get((r.node, r.attached_to))} for r in rs]})
        states[(s["service"], s["provider"])] = ServiceState(s["service"], s["provider"], "extracted",
                                                             plan[-1]["resources"], label=s["service"])
    completer = Completer([ctx["reconfigure_llm"]], ctx["fetcher"], ctx["urls"], Memory(None), ctx["vocabulary"])
    completer.states, completer.plan = states, plan
    rounds, history, failures, llm_outputs = 0, [], [], []
    current = (template, inserted)
    while True:
        rounds += 1
        failures, raw = llm_verify(ctx["verifier_llm"], current[0], current[1], ctx["urls"], ctx["fetcher"],
                                   ctx["vocabulary"], services)
        llm_outputs.append(raw)
        history.append([f.as_dict() for f in failures])
        if not failures or rounds >= ctx["cfg"]["verification_rounds"]:
            break
        actions = []
        for f in failures:
            if f.verification == ANCHORING:
                completer.remove(f.resource.node, f.resource.name)
                actions.append({"resource": f.resource.name, "node": f.resource.node, "action": "removed"})
        for f in failures:
            if f.verification != ANCHORING:
                ok = completer.reconfigure(f.resource, f.requirements)
                actions.append({"resource": f.resource.name, "node": f.resource.node,
                                "action": "reconfigured" if ok else "reconfiguration_failed"})
        history[-1] = {"failures": history[-1], "actions": actions}
        current = completer.complete_again(base)
    calls = take_usage()
    write_json(out_json, {"case": cid, "finished_at": now(), "expected": meta["expected_failures"],
                          "rounds": rounds, "history": history, "remaining_failures": [f.as_dict() for f in failures],
                          "verifier_outputs": llm_outputs,
                          # the final template checked by the algorithm (ground truth of "corrected"; no LLM)
                          "remaining_failures_algorithm": [f.as_dict() for f in verify(
                              current[0], current[1], ctx["urls"], ctx["fetcher"], ctx["vocabulary"], services)],
                          "time": {"wall_seconds": round(time.time() - t0, 3)}, "usage": usage_summary(calls),
                          "llm_calls": calls})


# ---------------------------------------------------------------------------------------------- staleness
def phase_staleness(ctx, folder):
    cid = os.path.basename(folder)
    out_json = os.path.join(ctx["dir"], "staleness", f"{cid}.json")
    if done_ok(out_json):
        return
    meta = json.load(open(os.path.join(folder, "case.json"), encoding="utf-8"))
    edited = open(os.path.join(folder, "page.txt"), encoding="utf-8").read()
    ann_path = os.path.join(DATASET, "annotation", "dependent_resources.aligned.json")
    ann = {(s["service"], s["provider"]): s for s in json.load(open(ann_path, encoding="utf-8"))["services"]}
    entry = ann[(meta["service"], meta["provider"])]
    urls_of = ctx["urls"].urls_for(meta["service"], meta["provider"])
    pages = {}
    for u in urls_of:                                  # a page without frozen copy stays unreachable, as in a run
        try:
            pages[u] = ctx["fetcher"].fetch(u)
        except FetchError:
            pass
    if meta["url"] not in pages:
        raise FetchError(f"no frozen copy of the edited page {meta['url']}")
    old = dict(pages)
    pages[meta["url"]] = edited
    mem = Memory(None)                                 # the memory before the edit: annotated resources, old pages
    mem.put(meta["service"], meta["provider"],
            [{k: r.get(k) for k in ("name", "quote", "url", "properties", "settings", "attached_to")}
             for r in entry["resources"]],
            {u: {"fingerprint": fingerprint(old[u]), "checked_at": "2000-01-01T00:00:00+00:00"} for u in old})
    set_tag(f"staleness/{cid}")
    take_usage()
    t0 = time.time()
    completer = Completer(ctx["extraction_llms"], DictFetcher(pages, ctx["urls"]), ctx["urls"], mem,
                          ctx["vocabulary"])
    st = completer._lookup(meta["service"], meta["provider"], meta["service"])
    calls = take_usage()
    write_json(out_json, {"case": cid, "finished_at": now(), "status": st.status,
                          "expected_status": meta["expected_status"],
                          "resources": [{"name": r["name"], "url": r.get("url"), "quote": r.get("quote"),
                                         "quote_in_edited_page": find_quote(r.get("quote") or "", pages.get(r.get("url"), ""))}
                                        for r in st.resources],
                          "extraction": st.extraction, "time": {"wall_seconds": round(time.time() - t0, 3)},
                          "usage": usage_summary(calls), "llm_calls": calls})


# ---------------------------------------------------------------------------------------------- ablation
def phase_minimal(ctx, case):
    cid = case["id"]
    out_json = os.path.join(ctx["dir"], "minimal", f"{cid}.json")
    if done_ok(out_json):
        return
    pkl = os.path.join(ctx["dir"], "selection", f"{cid}.pkl")
    if not os.path.exists(pkl):
        write_json(out_json, {"case": cid, "skipped": "no selection result"})
        return
    with open(pkl, "rb") as f:
        done = pickle.load(f)
    if done.status != "selected":
        write_json(out_json, {"case": cid, "skipped": f"selection status {done.status}"})
        return
    set_tag(f"minimal/{cid}")
    take_usage()
    t0 = time.time()
    per_node = {}
    for n, o in done.composition.items():
        label = f"{o.name} ({o.service})" if o.name else o.service
        per_node[n] = {"service": o.service, "provider": o.provider, **B.minimal_completion(ctx["minimal_llm"], label, o.provider)}
    calls = take_usage()
    write_json(out_json, {"case": cid, "finished_at": now(), "nodes": per_node,
                          "time": {"wall_seconds": round(time.time() - t0, 3)}, "usage": usage_summary(calls),
                          "llm_calls": calls})


# ---------------------------------------------------------------------------------------------- baselines
def phase_baselines_for(ctx, method, key, llm, cases):
    for case in cases:
        cid = case["id"]
        out_json = os.path.join(ctx["dir"], "baselines", f"{method}__{key.replace(':', '_')}", f"{cid}.json")
        if done_ok(out_json):
            continue
        t, d, c = load_case(cid)
        set_tag(f"baselines/{method}/{key}/{cid}")
        take_usage()
        t0 = time.time()
        try:
            r = B.run_method(method, llm, t, d, passages=ctx["passages"], reader=ctx["reader"])
            res = {"case": cid, "method": method, "llm": key, "finished_at": now(), **r}
        except Exception as exc:                          # noqa: BLE001
            res = {"case": cid, "method": method, "llm": key, "error": f"{type(exc).__name__}: {exc}",
                   "trace": traceback.format_exc()[-2000:]}
        calls = take_usage()
        res.update({"time": {"wall_seconds": round(time.time() - t0, 3)}, "usage": usage_summary(calls),
                    "llm_calls": calls})
        write_json(out_json, res)
        log(ctx["dir"], f"baselines {method} {key} {cid}: {'error' if 'error' in res else 'ok'}")


# ---------------------------------------------------------------------------------------------- cited pages
def phase_cited(ctx):
    """Copies, once, every official page cited outside the URL list (by the baselines, the minimal version or
    the completion), so that the anchoring of every method can be checked later on the pages of the run."""
    folder = os.path.join(ctx["dir"], "cited_pages")
    os.makedirs(folder, exist_ok=True)
    index_path = os.path.join(folder, "index.json")
    index = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else {}
    listed = {u for s in ctx["urls"].sources + ctx["urls"].validated for u in s["urls"]}
    cited = set()
    for root, _, files in os.walk(ctx["dir"]):
        for fn in files:
            if not fn.endswith(".json") or root.startswith(folder):
                continue
            try:
                obj = json.load(open(os.path.join(root, fn), encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            stack = [obj]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    u = x.get("url")
                    if isinstance(u, str) and u.startswith("http"):
                        cited.add(u.strip())
                    stack += list(x.values())
                elif isinstance(x, list):
                    stack += x
    http = HttpFetcher(ctx["urls"])
    failed = {}
    for u in sorted(cited - listed - set(index)):
        if not ctx["urls"].is_official(u):
            failed[u] = "not an official domain"
            continue
        try:
            text = http.fetch(u)
            name = f"{len(index):04d}.txt"
            with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
                f.write(text)
            index[u] = name
        except FetchError as exc:
            failed[u] = str(exc)[:300]
    write_json(index_path, index)
    write_json(os.path.join(folder, "not_copied.json"), {"date": now(), "failed": failed})
    log(ctx["dir"], f"cited pages: {len(index)} copied, {len(failed)} not copied")


# ---------------------------------------------------------------------------------------------- scalability
def phase_scalability(ctx):
    out_json = os.path.join(ctx["dir"], "scalability.json")
    if done_ok(out_json):
        return
    rng = random.Random(7)
    rows = []
    for n_nodes in (2, 4, 6, 8, 10, 15, 20):
        for n_offers in (10, 50, 100, 200):
            for rep in range(3):
                nodes = []
                for i in range(n_nodes):
                    offers = [Offer(f"o{i}_{j}", rng.choice(["aws", "azure", "gcp"]), f"r{rng.randint(0, 5)}",
                                    rng.randint(1000, 50000), round(rng.uniform(99.0, 99.99), 2))
                              for j in range(n_offers)]
                    nodes.append(Node(f"n{i}", offers))
                names = tuple(n.name for n in nodes)
                cheapest = sum(min(o.price_cents for o in n.candidates) for n in nodes) / 100
                cons = [QoSConstraint("budget", COST, names, round(cheapest * 1.3, 2)),
                        QoSConstraint("availability", AVAILABILITY, names, 97.0)]
                p = Problem(nodes, cons)
                t0 = time.time()
                s = select(p, [COST])
                rows.append({"nodes": n_nodes, "offers_per_node": n_offers, "rep": rep,
                             "seconds": round(time.time() - t0, 4), "signal": s.kind})
    write_json(out_json, {"finished_at": now(), "rows": rows})
    log(ctx["dir"], f"scalability: {len(rows)} problems")


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", required=True, help="folder of the frozen pages (index.json)")
    ap.add_argument("--run", required=True, help="name of the run (folder runs/<run>)")
    ap.add_argument("--phases", nargs="*", default=[p for p in PHASES if p != "verification_llm"], choices=PHASES)
    ap.add_argument("--cases", nargs="*", help="only these cases")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--dry-run", action="store_true", help="fake LLMs: checks the whole chain, no cost")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    B.PROMPT_VARIANT = cfg.get("baseline_prompt", "full")
    if a.cases and len(a.cases) == 1 and a.cases[0].startswith("@"):     # --cases @file: one case id per line
        with open(a.cases[0][1:], encoding="utf-8") as f:
            a.cases = [x.strip() for x in f if x.strip()]
    run_dir = os.path.join(ROOT, "runs", a.run)
    os.makedirs(run_dir, exist_ok=True)
    dry = a.dry_run
    urls = UrlList.load()
    fetcher = FrozenFetcher(a.frozen, urls)
    pages = {}
    for s in urls.sources:
        for u in s["urls"]:
            try:
                pages[u] = fetcher.fetch(u)
            except FetchError:
                pass
    cited_dir = os.path.join(run_dir, "cited_pages")
    reader_http = HttpFetcher(urls)

    def reader(u: str) -> str:                         # ReAct: frozen page, otherwise the official page online
        try:
            return fetcher.fetch(u)
        except FetchError:
            return reader_http.fetch(u)

    ctx = {"dir": run_dir, "cfg": cfg, "urls": urls, "fetcher": fetcher, "vocabulary": Vocabulary(),
           "passages": B.Passages(pages), "reader": reader,
           "negotiation_llm": make_llm(cfg["negotiation_llm"], dry),
           "extraction_llms": [make_llm(k, dry) for k in cfg["extraction_llms"]],
           "single_llm": make_llm(cfg["single_llm"], dry),
           "user_llm": None if cfg["user"] == "deterministic" else make_llm(cfg["user"], dry),
           "minimal_llm": make_llm(cfg["minimal_llm"], dry), "reconfigure_llm": make_llm(cfg["reconfigure_llm"], dry),
           "verifier_llm": make_llm(cfg["verifier_llm"], dry) if "verification_llm" in a.phases else None}
    write_json(os.path.join(run_dir, "config.json"),
               {"started_at": now(), "dry_run": dry, "config": cfg, "frozen": os.path.abspath(a.frozen),
                "dataset_summary": json.load(open(os.path.join(DATASET, "summary.json"), encoding="utf-8")),
                "python": sys.version})
    cases = manifest(a.cases)
    distinct = [c for c in cases if not c.get("twin_of")]
    log(run_dir, f"run {a.run}: {len(cases)} cases, phases {a.phases}{' (dry run)' if dry else ''}")

    def guarded(phase, unit, fn, *args):
        try:
            fn(*args)
        except Exception as exc:                          # noqa: BLE001 - logged, the unit is run again next time
            write_json(os.path.join(run_dir, phase, f"{unit}.json"),
                       {"case": unit, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:],
                        "usage": usage_summary(take_usage())})
            log(run_dir, f"{phase} {unit}: ERROR {type(exc).__name__}: {str(exc)[:200]}")

    for phase in [p for p in PHASES if p in a.phases]:
        t0 = time.time()
        log(run_dir, f"== phase {phase}")
        if phase == "selection":
            for c in cases:
                guarded(phase, c["id"], phase_selection, ctx, c)
                log(run_dir, f"selection {c['id']}")
        elif phase == "completion":
            # our approach with the majority vote (vote), and with a single LLM (single: comparison with the
            # baselines at the same LLM); each variant with its own memory, empty at the start (not on resume)
            for variant, llms in (("vote", ctx["extraction_llms"]), ("single", [ctx["single_llm"]])):
                memory = Memory(os.path.join(run_dir, f"memory_{variant}.json"))
                if not any(done_ok(os.path.join(run_dir, "completion", variant, f"{c['id']}.json")) for c in distinct):
                    memory.reset()
                for c in distinct:
                    guarded(os.path.join(phase, variant), c["id"], phase_completion, ctx, c, memory, variant, llms)
                    log(run_dir, f"completion {variant} {c['id']}")
        elif phase == "verification":
            root = os.path.join(DATASET, "verification")
            folders = sorted(os.path.join(root, x) for x in os.listdir(root)) if os.path.isdir(root) else []
            if not folders:
                log(run_dir, "verification: no case (run build_verification_cases.py first)")
            for f in folders:
                guarded(phase, os.path.basename(f), phase_verification, ctx, f)
        elif phase == "verification_llm":
            root = os.path.join(DATASET, "verification")
            for f in sorted(os.path.join(root, x) for x in os.listdir(root)):
                guarded(phase, os.path.basename(f), phase_verification_llm, ctx, f)
                log(run_dir, f"verification_llm {os.path.basename(f)}")
        elif phase == "staleness":
            root = os.path.join(DATASET, "staleness")
            folders = sorted(os.path.join(root, x) for x in os.listdir(root)) if os.path.isdir(root) else []
            if not folders:
                log(run_dir, "staleness: no case (run build_verification_cases.py first)")
            for f in folders:
                guarded(phase, os.path.basename(f), phase_staleness, ctx, f)
        elif phase == "minimal":
            for c in distinct:
                guarded(phase, c["id"], phase_minimal, ctx, c)
        elif phase == "baselines":
            per_llm: dict = {}                            # LLM -> its (method) units, run one after the other
            for key in cfg["baseline_llms"]:
                per_llm.setdefault(key, []).extend(cfg["baselines"])
            for key in cfg["react_llms"]:
                per_llm.setdefault(key, []).append("react")
            for key in cfg["reference_llms"]:              # strong LLMs alone (single-shot by default)
                have = per_llm.setdefault(key, [])
                have += [m for m in cfg["reference_methods"] if m not in have]
            jobs = []
            for key, methods in per_llm.items():
                llm = make_llm(key, dry)
                if cfg.get("parallel_methods"):               # one thread per (method, LLM): faster
                    jobs += [[(m, key, llm)] for m in methods]
                else:
                    jobs.append([(m, key, llm) for m in methods])

            def worker(job):                             # one thread per LLM: its methods one after the other
                for m, key, llm in job:
                    phase_baselines_for(ctx, m, key, llm, distinct)
            with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
                list(ex.map(worker, jobs))
        elif phase == "cited":
            phase_cited(ctx)
        elif phase == "scalability":
            phase_scalability(ctx)
        log(run_dir, f"== phase {phase} done in {time.time() - t0:.0f} s")
    log(run_dir, "run finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
