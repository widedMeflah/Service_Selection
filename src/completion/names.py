"""
Names of dependent resources: normalization, matching, and node identifiers.

Two resources extracted for the same node are the same resource when their normalized names match:
they share their head noun (last word) and the words of one are all contained in the other. Normalization lowercases, removes punctuation,
stop words and plural endings, and expands the acronyms defined in the page itself
(e.g., "virtual private cloud (VPC)" makes "VPC" and "virtual private cloud" the same name).
No list of synonyms is written by hand.
"""
from __future__ import annotations

import re

STOP = {"a", "an", "the", "for", "of", "to", "in", "on", "and", "or", "with", "your", "its", "amazon", "aws",
        "azure", "google", "cloud", "gcp", "instance", "instances"}
_ACRO = re.compile(r"((?:[A-Za-z][\w-]*\s+){1,6}?[A-Za-z][\w-]*)\s*\(([A-Z][A-Za-z0-9]{1,9})\)")


def _singular(w: str) -> str:
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def words(name: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (name or "").lower())
    return [_singular(t) for t in toks if t not in STOP]


def acronyms(page_text: str) -> dict[str, list[str]]:
    """Acronyms defined in a page: {"vpc": ["virtual", "private", "cloud"]}.
    The long form is the run of words before the parenthesis whose initials spell the acronym."""
    out: dict[str, list[str]] = {}
    for m in _ACRO.finditer(page_text or ""):
        acro = m.group(2)
        toks = re.findall(r"[A-Za-z0-9]+", m.group(1))
        n = len(acro)
        cand = toks[-n:] if len(toks) >= n else toks
        if "".join(t[0] for t in cand).lower() == acro.lower():
            out[acro.lower()] = [_singular(t.lower()) for t in cand]
    return out


def normalize(name: str, acro: dict[str, list[str]] | None = None) -> tuple[str, ...]:
    out = []
    for w in words(name):
        if acro and w in acro:
            out += [x for x in acro[w] if x not in STOP]
        else:
            out.append(w)
    return tuple(out)


def same_name(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Same head noun (last word), and the words of one are all contained in the other
    ("subnet group" = "DB subnet group", but "subnet" != "DB subnet group")."""
    if not a or not b or a[-1] != b[-1]:
        return False
    sa, sb = set(a), set(b)
    return sa <= sb or sb <= sa


_PAREN = re.compile(r"^\s*(.+?)\s*\(([^()]+)\)\s*$")


def strip_paren(name: str) -> str:
    """'VPC (Virtual Private Cloud)' -> 'VPC'; 'IP address (private IPv4)' -> 'IP address'."""
    m = _PAREN.match(name or "")
    return m.group(1) if m else (name or "")


def acronyms_of_names(names) -> dict[str, list[str]]:
    """Acronyms defined by the names themselves, in both orders: 'Virtual Private Cloud (VPC)' and
    'VPC (Virtual Private Cloud)' both give {"vpc": ["virtual", "private", "cloud"]}."""
    out: dict[str, list[str]] = {}
    for n in names:
        m = _PAREN.match(n or "")
        if not m:
            continue
        for short, long in ((m.group(2), m.group(1)), (m.group(1), m.group(2))):
            toks = re.findall(r"[A-Za-z0-9]+", long)
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,9}", short.strip()) and len(toks) >= 2 and \
                    "".join(t[0] for t in toks).lower() == short.strip().lower():
                out[short.strip().lower()] = [_singular(t.lower()) for t in toks]
    return out


def identity(name: str, acro: dict[str, list[str]] | None = None) -> frozenset:
    """Identity of a resource across services: the set of its normalized words, acronyms expanded.
    Stricter than same_name: "security group" and "VPC security group" stay two resources."""
    return frozenset(normalize(strip_paren(name), acro)) or frozenset([slug(name)])


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return s or "resource"
