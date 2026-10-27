"""
Memory of dependent resources (Section 6.1): data only, in JSON.

Key       a service, identified by its name and its provider (exact match, never by similarity)
Entry     its dependent resources, each with its source page and the passage of this page that requires it,
          quoted word for word, its properties and the settings its provider applies;
          and each source page with the fingerprint of its main text and the date of its last check.
The memory is written only by the verification module, once every verification passes.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from typing import Optional

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DEFAULT_PATH = os.path.join(DATA, "memory.json")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def key(service: str, provider: str) -> str:
    return f"{provider.lower()}/{service}"


class Memory:
    def __init__(self, path: Optional[str] = DEFAULT_PATH):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.entries = json.load(f).get("entries", {})

    def get(self, service: str, provider: str) -> Optional[dict]:
        e = self.entries.get(key(service, provider))
        return copy.deepcopy(e) if e is not None else None

    def put(self, service: str, provider: str, resources: list[dict], sources: dict[str, dict]):
        self.entries[key(service, provider)] = {
            "service": service, "provider": provider.lower(),
            "resources": copy.deepcopy(resources), "sources": copy.deepcopy(sources), "updated_at": now_iso()}

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"entries": self.entries}, f, indent=2)

    def reset(self):
        self.entries = {}
        self.save()
