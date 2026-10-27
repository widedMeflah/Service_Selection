"""
Text utilities of the memory (pure algorithms, no LLM):
  html_to_text     official page (HTML) -> plain text
  normalize        text used for fingerprints and quote matching
  fingerprint      SHA-256 of the normalized page: detects that a page changed
  find_quote       exact citation check (after normalization of spaces, quotes and dashes)
  chunks           splits a page into chunks that fit the context of the LLM
"""
from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser

MIN_QUOTE_CHARS = 20          # a shorter "quote" proves nothing

_SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "template"}
_BLOCK = {"p", "div", "li", "ul", "ol", "br", "tr", "td", "th", "section", "article", "pre", "table",
          "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "blockquote", "main"}


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextParser()
    p.feed(html)
    text = "".join(p.parts)
    lines = [re.sub(r"[ \t\r\f\v ]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "«": '"', "»": '"',
                        "–": "-", "—": "-", " ": " "})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").translate(_TRANS)).strip().lower()


def fingerprint(text: str) -> str:
    """Hash of the main text of a page (navigation, headers, footers and scripts already removed by
    html_to_text), after normalizing spaces and punctuation: a layout update does not change it."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def find_quote(quote: str, page_text: str) -> bool:
    """True if the quote appears word for word in the page (case, spaces, quotes and dashes normalized)."""
    q = normalize(quote).strip(" .\"'")
    return len(q) >= MIN_QUOTE_CHARS and q in normalize(page_text)


def chunks(text: str, max_chars: int = 12000) -> list[str]:
    """Splits a page into chunks that fit the context of the LLM, on paragraph boundaries
    (a paragraph longer than max_chars is cut). Every part of the page is kept."""
    paras = [p for p in (text or "").split("\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        while len(p) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:max_chars])
            p = p[max_chars:]
        if cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out
