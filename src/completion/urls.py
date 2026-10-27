"""
List of URLs of official documentation pages known for each service (Section 6.1).

data/doc_urls.json            initialized by hand for common services, with the official domains of each provider
data/doc_urls_validated.json  URLs validated by an expert at run time (added to the list, reused afterwards)

A service is identified by its name and its provider (exact match).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DEFAULT_PATH = os.path.join(DATA, "doc_urls.json")
VALIDATED_PATH = os.path.join(DATA, "doc_urls_validated.json")


class UrlList:
    def __init__(self, data: dict, validated_path: Optional[str] = None):
        self.domains: dict[str, list[str]] = {k.lower(): [d.lower() for d in v]
                                              for k, v in (data.get("official_domains") or {}).items()}
        self.sources: list[dict] = list(data.get("sources") or [])
        self.validated_path = validated_path
        self.validated: list[dict] = []
        if validated_path and os.path.exists(validated_path):
            with open(validated_path, encoding="utf-8") as f:
                self.validated = json.load(f).get("sources", [])

    @classmethod
    def load(cls, path: str = DEFAULT_PATH, validated_path: Optional[str] = VALIDATED_PATH) -> "UrlList":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), validated_path)

    # ------------------------------------------------------------------
    def official_domains(self, provider: str) -> list[str]:
        return self.domains.get(provider.lower(), [])

    def is_official(self, url: str, provider: Optional[str] = None) -> bool:
        host = (urlparse(url).hostname or "").lower()
        provs = [provider.lower()] if provider else list(self.domains)
        return any(host == d or host.endswith("." + d) for p in provs for d in self.domains.get(p, []))

    def urls_for(self, service: str, provider: str) -> list[str]:
        out = []
        for s in self.sources + self.validated:
            if s.get("service") == service and str(s.get("provider", "")).lower() == provider.lower():
                for u in s.get("urls", []):
                    if self.is_official(u, provider) and u not in out:
                        out.append(u)
        return out

    def add_validated(self, service: str, provider: str, url: str, expert: str = "expert"):
        entry = {"service": service, "provider": provider, "urls": [url], "validated_by": expert,
                 "validated_at": datetime.now(timezone.utc).isoformat()}
        self.validated.append(entry)
        if self.validated_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.validated_path)), exist_ok=True)
            with open(self.validated_path, "w", encoding="utf-8") as f:
                json.dump({"sources": self.validated}, f, indent=2)
