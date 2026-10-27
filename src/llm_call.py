"""
One way to call any LLM of the project: a LangChain chat model (invoke -> message with .content),
or a test double (a callable prompt -> text, or an object with invoke()).
"""
from __future__ import annotations

import os
import re
import sys
import time

# A rate limit (HTTP 429, "rate limit", provider overloaded) is temporary: the call waits for the delay
# given by the provider (e.g. "Please try again in 1.9s"), or an increasing delay, then tries again.
# A delay longer than MAX_WAIT (e.g. a daily quota) is not waited for: the call fails, and the failure
# is reported by the caller.
MAX_RETRIES = int(os.environ.get("SELCOMP_MAX_RETRIES", "6"))
MAX_WAIT = float(os.environ.get("SELCOMP_MAX_WAIT", "120"))
_RATE = re.compile(r"\b429\b|\b529\b|rate.?limit|too many requests|overloaded", re.I)
_WAIT = re.compile(r"(?:try again|retry) in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)\s*(ms|s))?", re.I)
_sleep = time.sleep

# Usage of every LLM call (evaluation): model, tag of the current unit of work (set by the caller), input and output
# tokens (from the provider when it reports them, otherwise estimated as characters / 4), latency, success.
# Kept per thread, so that units of work run in parallel threads do not mix their calls.
import threading as _threading

_LOCAL = _threading.local()


def _state():
    if not hasattr(_LOCAL, "usage"):
        _LOCAL.usage, _LOCAL.tag = [], ""
    return _LOCAL


def set_tag(tag: str) -> None:
    _state().tag = tag


def take_usage() -> list[dict]:
    """The calls recorded in this thread since the last call, which are then forgotten."""
    st = _state()
    out, st.usage = list(st.usage), []
    return out


def _record(llm, prompt: str, out, t0: float, ok: bool, error: str = "") -> None:
    um = getattr(out, "usage_metadata", None) or {}
    text = getattr(out, "content", out) if out is not None else ""
    text = text if isinstance(text, str) else str(text)
    _state().usage.append({"tag": _state().tag, "model": llm_name(llm), "ok": ok, "error": error[:200],
                  "input_tokens": int(um.get("input_tokens") or 0) or len(prompt) // 4,
                  "output_tokens": int(um.get("output_tokens") or 0) or len(text) // 4,
                  "reported": bool(um), "seconds": round(time.time() - t0, 2)})


def is_rate_limit(exc: Exception) -> bool:
    return bool(_RATE.search(f"{type(exc).__name__} {exc}"))


def wait_seconds(exc: Exception, attempt: int) -> float:
    m = _WAIT.search(str(exc))
    if m and any(m.groups()):
        h, mi, v, unit = m.groups()
        secs = int(h or 0) * 3600 + int(mi or 0) * 60
        if v:
            secs += float(v) / (1000 if unit.lower() == "ms" else 1)
        return secs + 0.5
    return min(60.0, 2.0 * 2 ** attempt)


def _invoke(llm, prompt: str):
    if callable(llm) and not hasattr(llm, "invoke"):
        return llm(prompt)
    return llm.invoke(prompt)


def ask_text(llm, prompt: str) -> str:
    if llm is None:
        raise RuntimeError("no LLM")
    attempt = 0
    t0 = time.time()
    while True:
        try:
            out = _invoke(llm, prompt)
            _record(llm, prompt, out, t0, True)
            break
        except Exception as exc:                          # noqa: BLE001
            if not is_rate_limit(exc) or attempt >= MAX_RETRIES:
                _record(llm, prompt, None, t0, False, f"{type(exc).__name__}: {exc}")
                raise
            delay = wait_seconds(exc, attempt)
            if delay > MAX_WAIT:
                raise
            print(f"  rate limit of {llm_name(llm)}: waiting {delay:.1f} s", file=sys.stderr)
            _sleep(delay)
            attempt += 1
    if isinstance(out, str):
        return plain_labels(strip_reasoning(out))
    content = getattr(out, "content", out)
    if isinstance(content, list):          # some providers return a list of parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return plain_labels(strip_reasoning(str(content)))


LABELS = ("RESOURCE", "QUOTE", "URL", "ATTACHED TO", "PROPERTIES", "SETTINGS", "MESSAGE", "OPTION", "PRIORITY",
          "UNCLEAR", "ANSWER", "PROBLEMS")
_BEFORE_LABEL = re.compile(r"(?mi)^[ \t]*(?:#{1,6}[ \t]*)?(?:[-*\u2022][ \t]+)?(?:\d+[.)][ \t]*)?(?=(?:"
                           + "|".join(LABELS) + r")[ \t]*:)")


def plain_labels(text: str) -> str:
    """The output format of the prompts in plain text, whatever markdown the LLM adds: bold markers removed
    ("**RESOURCE:** VPC" -> "RESOURCE: VPC"), headings, bullets and numbers before a label removed
    ("### 1. RESOURCE: VPC"), block quotes unwrapped ("> quote")."""
    text = re.sub(r"\*\*|__", "", text or "")
    text = re.sub(r"(?m)^[ \t]*>[ \t]?", "", text)
    return _BEFORE_LABEL.sub("", text)


_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Removes the reasoning that some models (e.g. DeepSeek, Nemotron) write before their answer, so that
    only the answer is parsed; an unclosed reasoning block is cut up to its end."""
    text = _THINK.sub("", text or "")
    m = re.search(r"</think(?:ing)?>", text, re.I)
    if m:
        text = text[m.end():]
    return text.strip()


def llm_name(llm) -> str:
    return str(getattr(llm, "name", None) or getattr(llm, "model_name", None) or getattr(llm, "model", None)
               or type(llm).__name__)
