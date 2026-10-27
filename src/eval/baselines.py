"""
Baselines of the evaluation (implemented by us, same LLM as our approach, same inputs: the provider-independent
template and the candidate offers). Each one returns the selection and the dependent resources in one structured
answer (JSON), which the evaluation compares with the ground truth; no negotiation.

  single-shot   one prompt (BASELINE_PROMPT)
  cot           the same prompt, with step-by-step reasoning before the answer (COT_INSTRUCTION)
  rag           the same prompt, with the passages of the official pages closest to the template (BM25 over the
                frozen pages of the URL list), each with its URL (RAG_CONTEXT)
  self-refine   the single-shot answer, then one call that reviews and corrects it (REFINE_PROMPT)
  react         a tool-using agent (REACT_PROMPT): offers of a node, evaluation of a selection, solver, search in
                the official pages, reading of an official page; at most REACT_STEPS steps
Every prompt follows the structure of the prompts of the approach: Role (i), Context (ii), Task (iii),
Instructions (iv), Output format (v).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Callable, Optional

import yaml

from ..completion.text import chunks
from ..llm_call import ask_text
from ..selection.builder import build_problem
from ..selection.cpsat import solve
from ..selection.problem import COST, AVAILABILITY, objectives, satisfies, availability
from ..selection.select import priority_spec

METHODS = ("single-shot", "cot", "rag", "self-refine", "react")
RAG_PASSAGES = 6
RAG_PASSAGE_CHARS = 1500
REACT_STEPS = 8
READ_CHARS = 6000

DEFINITION = ("a dependent resource is a resource that the deployment code must create or reference for the "
              "service; exclude the parts that the provider creates automatically with the service and that are "
              "configured only through the attributes of the service; keep the resources that must be created or "
              "referenced for the service, including a resource that the provider supplies by default when none "
              "is specified")

OUTPUT_JSON = """{"feasible": <true | false>,
 "selection": {"<node>": "<offer_id>"},
 "violated_policies": ["<policy name>"],
 "dependent_resources": [{"node": "<node>", "name": "<resource>", "attached_to": "<resource name or null>",
                          "url": "<official documentation page>", "quote": "<quote copied word for word>"}]}"""

BASELINE_PROMPT = """Role (i)
You are a cloud architect who prepares the deployment of applications on cloud providers.

Context (ii)
Considering the provider-independent TOSCA template:
{template}
Considering the candidate offers of each node that represents a cloud service (monthly price in USD, availability
in % = SLA of the priced configuration):
{offers}{extra}

Task (iii)
Produce the provider-specific deployment: select one candidate offer for each cloud service node, and give the
dependent resources of each selected service.

Instructions (iv)
Follow these instructions:
- select only offers listed above, one for each cloud service node.
- satisfy every policy of the template: the cost of a Cost policy is the sum of the monthly prices of its targets;
  the availability of an Availability policy is the product of the availabilities of its targets, where the
  targets of a Redundancy policy are replicas, available when one of them is (1 - product of (1 - a));
  a Colocation policy requires the same provider (scope provider) or the same region (scope region) for its targets.
- choose a selection that no other selection satisfying the policies beats on both cost and availability.
- if no selection satisfies every policy, set feasible to false, give the selection you propose instead, and list
  the policies it violates.
- {definition}.
- give each dependent resource with the URL of an official documentation page of its provider and a quote copied
  word for word from that page, which states that the service requires the resource.{more}

Output format (v)
{output}"""

COT_INSTRUCTION = """
- reason step by step before answering: write your reasoning, then the line ANSWER: followed by the JSON object."""

RAG_CONTEXT = """
Considering these passages of official documentation pages, each with its URL:
{passages}"""

REFINE_PROMPT = """Role (i)
You are a cloud architect who reviews a provider-specific deployment.

Context (ii)
Considering the task and the inputs given to you before:
{task}
Considering your answer:
{answer}

Task (iii)
Review your answer and correct it.

Instructions (iv)
Follow these instructions:
- check that every selected offer is a listed candidate of its node.
- check every policy with the prices and availabilities of the selected offers.
- check that every dependent resource satisfies the definition, and that its URL and quote are those of an official
  documentation page.
- write the problems you find, then the line ANSWER: followed by the corrected JSON object.

Output format (v)
PROBLEMS: <problems found, or none>
ANSWER: <JSON object in the format of the task>"""

REACT_PROMPT = """Role (i)
You are a cloud architect who prepares the deployment of applications on cloud providers, using tools.

Context (ii)
Considering the provider-independent TOSCA template:
{template}
Considering the cloud service nodes: {nodes}
Considering the tools:
- offers[<node>]: the candidate offers of a node (monthly price in USD, availability in %)
- evaluate[{{"<node>": "<offer_id>", ...}}]: the cost, the availability and the violated policies of a selection
- solve[cost | availability]: the best selection that satisfies every policy for this criterion, if any
- search[<words>]: passages of official documentation pages, with their URL
- read[<url>]: the text of an official documentation page
- finish[<JSON object>]: your final answer

Task (iii)
Produce the provider-specific deployment: select one candidate offer for each cloud service node, and give the
dependent resources of each selected service.

Instructions (iv)
Follow these instructions:
- at each step write one line Thought: <reasoning> and one line Action: <tool>[<argument>]; you receive the line
  Observation: <result>.
- satisfy every policy of the template; if no selection satisfies them, set feasible to false and list the
  policies that your selection violates.
- {definition}.
- give each dependent resource with the URL of an official documentation page of its provider and a quote copied
  word for word from that page.
- finish in at most {steps} steps.

Output format (v)
Thought: <reasoning>
Action: <tool>[<argument>]
The argument of finish is the JSON object:
{output}

{history}"""


# ---------------------------------------------------------------------------------------------- selection variant
# "selection": the baselines rerun for the selection evaluation (QR1, QR2). The prompt states the task and its inputs
# only: no rule on infeasible cases, no Pareto rule, no formula of the policies (the template describes each policy
# type), and an output with the selection only (no feasible flag, no list of violated policies).
PROMPT_VARIANT = "full"

SELECTION_OUTPUT = """{"selection": {"<node>": "<offer_id>"}}"""

SELECTION_PROMPT = """Role (i)
You are a cloud architect who prepares the deployment of applications on cloud providers.

Context (ii)
Considering the provider-independent TOSCA template:
{template}
Considering the candidate offers of each node that represents a cloud service (monthly price in USD, availability
in % = SLA of the priced configuration):
{offers}{extra}

Task (iii)
Select one candidate offer for each cloud service node.

Instructions (iv)
Follow these instructions:
- select only offers listed above, one for each cloud service node.
- satisfy every policy of the template.
- choose the most appropriate selection.{more}

Output format (v)
{output}"""

REFINE_SELECTION_PROMPT = """Role (i)
You are a cloud architect who reviews a selection of cloud offers.

Context (ii)
Considering the task and the inputs given to you before:
{task}
Considering your answer:
{answer}

Task (iii)
Review your answer and correct it.

Instructions (iv)
Follow these instructions:
- check that every selected offer is a listed candidate of its node.
- check every policy with the prices and availabilities of the selected offers.
- write the problems you find, then the line ANSWER: followed by the corrected JSON object.

Output format (v)
PROBLEMS: <problems found, or none>
ANSWER: <JSON object in the format of the task>"""

REACT_SELECTION_PROMPT = """Role (i)
You are a cloud architect who prepares the deployment of applications on cloud providers, using tools.

Context (ii)
Considering the provider-independent TOSCA template:
{template}
Considering the cloud service nodes: {nodes}
Considering the tools:
- offers[<node>]: the candidate offers of a node (monthly price in USD, availability in %)
- evaluate[{{"<node>": "<offer_id>", ...}}]: the cost, the availability and the violated policies of a selection
- solve[cost | availability]: the best selection that satisfies every policy for this criterion, if any
- finish[<JSON object>]: your final answer

Task (iii)
Select one candidate offer for each cloud service node.

Instructions (iv)
Follow these instructions:
- at each step write one line Thought: <reasoning> and one line Action: <tool>[<argument>]; you receive the line
  Observation: <result>.
- satisfy every policy of the template.
- finish in at most {steps} steps.

Output format (v)
Thought: <reasoning>
Action: <tool>[<argument>]
The argument of finish is the JSON object:
{output}

{history}"""


# ---------------------------------------------------------------------------------------------- inputs
def template_text(t: dict) -> str:
    """The template as given to every method (the generated description of the case is left out)."""
    t = {k: v for k, v in t.items() if k != "description"}
    return yaml.safe_dump(t, sort_keys=False, allow_unicode=True, width=110).strip()


def offers_text(d: dict, node: Optional[str] = None) -> str:
    lines = []
    for n in d["nodes"]:
        if node is not None and n["node_name"] != node:
            continue
        lines.append(f"{n['node_name']}:")
        for o in n["candidates"]:
            dep = f", {o['deployment']}" if o.get("deployment") else ""
            lines.append(f"  {o['offer_id']} | {o['provider']} | {o['region']} | {o['price_month']} USD | "
                         f"{o.get('sla')} %{dep}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- retrieval
_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower())


class Passages:
    """BM25 over the passages of the frozen pages of the URL list (an algorithm, no LLM)."""

    def __init__(self, pages: dict[str, str], size: int = RAG_PASSAGE_CHARS):
        self.items = [(u, c) for u, t in pages.items() for c in chunks(t, size)]
        self.toks = [_tokens(c) for _, c in self.items]
        self.df = Counter(w for t in self.toks for w in set(t))
        self.avg = sum(len(t) for t in self.toks) / max(1, len(self.toks))

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        q, n = _tokens(query), len(self.toks)
        scores = []
        for i, t in enumerate(self.toks):
            tf, s = Counter(t), 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(t) / self.avg))
            scores.append((s, i))
        return [self.items[i] for s, i in sorted(scores, reverse=True)[:k] if s > 0]


def rag_query(t: dict, d: dict) -> str:
    words = ["requires", "virtual", "network", "subnet", "security", "private", "access"]
    for n in d["nodes"]:
        o = n["candidates"][0]
        words += ["database", str(o.get("engine") or "")] if o["offer_type"] == "database" else \
                 ["instance", "virtual", "machine", "compute"]
    return " ".join(words)


# ---------------------------------------------------------------------------------------------- answers
def parse_answer(text: str) -> Optional[dict]:
    """The JSON object of an answer (after ANSWER: when present), or None."""
    if not text:
        return None
    m = re.search(r"ANSWER:\s*", text)
    if m:
        text = text[m.end():]
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\" and not esc)
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        out = json.loads(text[start:i + 1])
                        return out if isinstance(out, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _prompt(t, d, extra="", more="") -> str:
    if PROMPT_VARIANT == "selection":
        return SELECTION_PROMPT.format(template=template_text(t), offers=offers_text(d), extra=extra, more=more,
                                       output=SELECTION_OUTPUT)
    return BASELINE_PROMPT.format(template=template_text(t), offers=offers_text(d), extra=extra,
                                  definition=DEFINITION, more=more, output=OUTPUT_JSON)


# ---------------------------------------------------------------------------------------------- methods
def run_method(method: str, llm, t: dict, d: dict, pages: Optional[dict] = None, passages: Optional[Passages] = None,
               reader: Optional[Callable[[str], str]] = None) -> dict:
    """Runs one baseline on one case: {"method", "steps": [{"prompt", "output"}], "answer": parsed JSON or None}."""
    steps = []

    def call(prompt):
        out = ask_text(llm, prompt)
        steps.append({"prompt": prompt, "output": out})
        return out

    if method == "single-shot":
        out = call(_prompt(t, d))
    elif method == "cot":
        out = call(_prompt(t, d, more=COT_INSTRUCTION))
    elif method == "rag":
        found = passages.search(rag_query(t, d), RAG_PASSAGES) if passages else []
        ctx = RAG_CONTEXT.format(passages="\n".join(f"[{u}]\n{c}" for u, c in found))
        out = call(_prompt(t, d, extra=ctx))
    elif method == "self-refine":
        task = _prompt(t, d)
        first = call(task)
        refine = REFINE_SELECTION_PROMPT if PROMPT_VARIANT == "selection" else REFINE_PROMPT
        out = call(refine.format(task=task, answer=first))
        if parse_answer(out) is None:            # the review gave no usable answer: the first answer stands
            out = first
    elif method == "react":
        out = react(llm, t, d, passages, reader, steps)
    else:
        raise ValueError(method)
    return {"method": method, "steps": steps, "answer": parse_answer(out)}


TOOLS = ("offers", "evaluate", "solve", "search", "read", "finish")


def parse_action(out: str):
    """(tool, argument, position) of the first action of a ReAct step, in the forms written by the LLMs:
    Action: tool[argument]; Action: tool(argument); Action: tool followed by the argument (e.g. a JSON block for
    finish); and a final JSON answer without an action line (finish). None when there is no action."""
    for m in re.finditer(r"Action\s*:\s*`?(\w+)`?\s*", out or ""):
        name = m.group(1).lower()
        if name not in TOOLS:
            continue
        rest = out[m.end():]
        if rest.startswith("["):
            depth = 0
            for i, ch in enumerate(rest):
                depth += ch == "["
                depth -= ch == "]"
                if depth == 0:
                    return name, rest[1:i], m.start()
            return name, rest[1:], m.start()
        if rest.startswith("("):
            j = rest.rfind(")")
            return name, rest[1:j if j > 0 else None], m.start()
        if name == "finish":
            return name, rest, m.start()
        line = re.sub(r"(?i)^action input\s*:\s*", "", rest.strip()).split("\n")[0]
        return name, line.strip(), m.start()
    a = parse_answer(out)
    if a is not None and "selection" in a:
        return "finish", json.dumps(a), len(out or "")
    return None


def react(llm, t, d, passages, reader, steps) -> str:
    p = build_problem(t, d)
    nodes = p.node_names
    history: list[str] = []

    def tool(name: str, arg: str) -> str:
        arg = arg.strip()
        if name == "offers":
            return offers_text(d, arg) or f"no cloud service node named {arg}"
        if name == "evaluate":
            sel = parse_answer(arg) or {}
            x = {}
            for n in nodes:
                o = next((o for o in p.node(n).candidates if o.offer_id == sel.get(n)), None)
                if o is None:
                    return f"no candidate offer {sel.get(n)} for node {n}"
                x[n] = o
            c, a = objectives(x, nodes, p.groups)
            known = availability(x, nodes, p.groups) is not None
            bad = [k.id for k in p.constraints if not satisfies(x, k, p.groups)]
            return (f"cost {c / 100:.2f} USD, availability {a:.4f} %" if known else f"cost {c / 100:.2f} USD") + \
                   (f", violated policies: {', '.join(bad)}" if bad else ", every policy is satisfied")
        if name == "solve":
            crit = AVAILABILITY if "avail" in arg.lower() else COST
            x = solve(p, p.constraints, priority_spec(p, [crit]))
            if x is None:
                return "no selection satisfies every policy"
            return json.dumps({n: x[n].offer_id for n in nodes})
        if name == "search":
            found = passages.search(arg, 3) if passages else []
            return "\n".join(f"[{u}] {c[:800]}" for u, c in found) or "no passage"
        if name == "read":
            try:
                return (reader(arg) if reader else "")[:READ_CHARS] or "empty page"
            except Exception as exc:                     # noqa: BLE001
                return f"cannot read {arg}: {exc}"
        return f"unknown tool {name}"

    last = ""
    for step in range(REACT_STEPS):
        if PROMPT_VARIANT == "selection":
            prompt = REACT_SELECTION_PROMPT.format(template=template_text(t), nodes=", ".join(nodes),
                                                   steps=REACT_STEPS, output=SELECTION_OUTPUT,
                                                   history="\n".join(history))
        else:
            prompt = REACT_PROMPT.format(template=template_text(t), nodes=", ".join(nodes), definition=DEFINITION,
                                         steps=REACT_STEPS, output=OUTPUT_JSON, history="\n".join(history))
        out = ask_text(llm, prompt)
        steps.append({"prompt": prompt, "output": out})
        last = out
        act = parse_action(out)
        if act is None:
            history.append(out.strip() + "\nObservation: write one Action line, e.g. Action: offers[<node>]")
            continue
        name, arg, start = act
        if name == "finish":
            return arg
        obs = tool(name, arg)
        thought = out[:start].strip()
        history.append(f"{thought}\nAction: {name}[{arg.strip()}]\nObservation: {obs}")
    return last


# ---------------------------------------------------------------------------------------------- ablation
MINIMAL_PROMPT = """Role (i)
You are an expert in cloud infrastructure and provider documentation.

Context (ii)
Considering the following service:
Service: {service}
Provider: {provider}

Task (iii)
Identify, from your knowledge, all the dependent resources of the service, i.e., the resources it requires from
its provider.

Instructions (iv)
Follow these instructions:
- {definition}.
- give each resource with the URL of an official documentation page of the provider and a quote of that page,
  which states that the service requires this resource.

Output format (v)
RESOURCE: <name>
URL: <official documentation page>
QUOTE: <quote>
ATTACHED TO: <the service, or the name of another resource that requires it>"""


def minimal_completion(llm, service_label: str, provider: str) -> dict:
    """Ablation "minimal version": the dependent resources of a service from the knowledge of one LLM (no page,
    no memory, no quote check, no verification)."""
    prompt = MINIMAL_PROMPT.format(service=service_label, provider=provider, definition=DEFINITION)
    out = ask_text(llm, prompt)
    res = []
    for block in re.split(r"(?im)^\s*RESOURCE\s*:", out or "")[1:]:
        lines = block.strip().split("\n")
        name = lines[0].strip().strip("<>")
        if not name or name.lower() in ("none", "n/a") or re.match(r"(?i)none\b", name):
            continue
        body = "\n".join(lines[1:])
        f = {k: (re.search(rf"(?im)^\s*{k}\s*:\s*(.*)$", body) or [None, ""])[1].strip()
             for k in ("URL", "QUOTE", "ATTACHED TO")}
        u = re.search(r"https?://[^\s\])>\"'`]+", f["URL"])            # also inside a markdown link [url](url)
        res.append({"name": name.strip("`* "), "url": u.group(0) if u else "", "quote": f["QUOTE"].strip("\"'“”*_ "),
                    "attached_to": f["ATTACHED TO"].strip("`* ")})
    return {"prompt": prompt, "output": out, "resources": res}
