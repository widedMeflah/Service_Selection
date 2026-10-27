"""
Command line: negotiated selection and verified completion of dependent resources.

  python run.py --tosca data/eval/dataset/cases/B-aws-vm/template.tosca.yaml --discovery data/eval/dataset/cases/B-aws-vm/discovery.json
  python run.py ... --negotiation-llm NVIDIA_nemotron --extraction-llms NVIDIA_nemotron Gemini_flash Groq_gpt120b
  python run.py ... --frozen data/eval/snapshots/    # frozen copies of the official pages (evaluation)
  python run.py ... --simulate availability        # simulated user instead of the console
  python run.py ... --out provider_specific.tosca.yaml

The user answers in the console; the expert validates, in the console, the URLs proposed for a new service.
"""
from __future__ import annotations

import argparse
import json
import sys

from src.common.tosca import load_yaml, load_json, dump_yaml
from src.completion.fetch import HttpFetcher, FrozenFetcher
from src.completion.memory import Memory
from src.completion.urls import UrlList
from src.eval.simulated_user import SimulatedUser
from src.llm import create_llm, LLM_CONFIGS, EXTRACTION_LLMS, NEGOTIATION_LLM
from src.pipeline import run


def console_user(message: str, awaiting: str) -> str:
    print("\n" + message)
    return input("> ")


def console_expert(service, provider, url, resources) -> bool:
    print(f"\n[expert] proposed page for {service} on {provider}: {url}")
    for r in resources:
        print(f"   - {r['name']}: \"{r['quote']}\"")
    return input("[expert] validate this URL? (yes / no) > ").strip().lower() in ("y", "yes", "oui")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tosca", required=True)
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--negotiation-llm", default=NEGOTIATION_LLM, help=f"{list(LLM_CONFIGS)} or provider:model")
    ap.add_argument("--extraction-llms", nargs="*", default=EXTRACTION_LLMS,
                    help=f"three LLMs (majority vote): {list(LLM_CONFIGS)} or provider:model")
    ap.add_argument("--frozen", help="folder of frozen pages (index.json); default: read the pages online")
    ap.add_argument("--memory", default="data/memory.json")
    ap.add_argument("--simulate", choices=["cost", "availability"], help="simulated user with this priority")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--verification-rounds", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    urls = UrlList.load()
    fetcher = FrozenFetcher(a.frozen, urls) if a.frozen else HttpFetcher(urls)
    neg = create_llm(a.negotiation_llm)
    ext = [create_llm(k) for k in a.extraction_llms]
    ext = [e for e in ext if e is not None]
    user = SimulatedUser(a.simulate) if a.simulate else console_user
    r = run(load_yaml(a.tosca), load_json(a.discovery), user=user, negotiation_llm=neg, extraction_llms=ext,
            expert=console_expert, fetcher=fetcher, urls=urls, memory=Memory(a.memory),
            max_rounds=a.max_rounds, verification_rounds=a.verification_rounds)
    print(f"\nstatus: {r.status}")
    if r.template is None:
        return 1
    for c in r.completion:
        errs = len((c.get("extraction") or {}).get("errors") or [])
        print(f"  {c['service']} on {c['provider']}: {c['status']}, {len(c['resources'])} resource(s)"
              f"{', ' + str(errs) + ' failed LLM call(s)' if errs else ''}"
              f"{', unreachable: ' + ', '.join(c['unreachable']) if c['unreachable'] else ''}"
              f", {'stored' if c['stored'] else 'not stored'} in the memory")
        for u, why in (c.get("unreachable_reasons") or {}).items():
            print(f"    unreachable {u}: {why}")
    print(json.dumps(r.verification["failures"], indent=2) if r.verification["failures"] else "verification: all passed")
    text = dump_yaml(r.template)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"provider-specific template written to {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
