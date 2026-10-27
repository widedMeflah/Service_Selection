"""
Retrieval of the official documentation pages from their URLs (an algorithm, no LLM).

HttpFetcher    reads a page online (standard library only), refuses a URL outside the official domains
FrozenFetcher  reads frozen copies (tests, evaluation): a folder with index.json {url: file}
DictFetcher    pages given in memory (tests): {url: text}; a missing URL is unreachable
All return the plain text of the page (navigation, headers, footers and scripts removed).
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

from .text import html_to_text


class FetchError(Exception):
    pass


class _Base:
    def __init__(self, urls=None):
        self.urls = urls              # UrlList, used to refuse non-official domains
        self.cache: dict[str, str] = {}
        self.calls: list[str] = []

    def _check(self, url: str):
        if self.urls is not None and not self.urls.is_official(url):
            raise FetchError(f"URL outside the official domains: {url}")

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        self._check(url)
        if url not in self.cache:
            self.cache[url] = self._read(url)
        return self.cache[url]

    def refresh(self):
        """Forgets the cached pages (to read their current version again)."""
        self.cache.clear()

    def _read(self, url: str) -> str:
        raise NotImplementedError


class HttpFetcher(_Base):
    def __init__(self, urls=None, timeout: int = 20):
        super().__init__(urls)
        self.timeout = timeout

    def _read(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "selection-completion/2.0 (research prototype)"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                charset = r.headers.get_content_charset() or "utf-8"
                return html_to_text(r.read().decode(charset, errors="replace"))
        except Exception as exc:                     # noqa: BLE001
            raise FetchError(f"cannot read {url}: {exc}") from exc


class FrozenFetcher(_Base):
    def __init__(self, folder: str, urls=None):
        super().__init__(urls)
        self.folder = folder
        with open(os.path.join(folder, "index.json"), encoding="utf-8") as f:
            self.index = json.load(f)

    def _read(self, url: str) -> str:
        if url not in self.index:
            raise FetchError(f"no frozen copy for {url}")
        path = os.path.join(self.folder, self.index[url])
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        return html_to_text(raw) if path.endswith((".html", ".htm")) else raw


class DictFetcher(_Base):
    def __init__(self, pages: dict[str, str], urls=None):
        super().__init__(urls)
        self.pages = pages

    def _read(self, url: str) -> str:
        if url not in self.pages:
            raise FetchError(f"unreachable: {url}")
        v = self.pages[url]
        return html_to_text(v) if v.lstrip().startswith("<") else v

    def set_page(self, url: str, text: Optional[str]):
        """Simulates a provider update (text) or an unreachable page (None)."""
        if text is None:
            self.pages.pop(url, None)
        else:
            self.pages[url] = text
        self.cache.pop(url, None)
