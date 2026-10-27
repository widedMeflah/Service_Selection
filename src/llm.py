"""
LLM factory. Keys in a .env file:
  OPENAI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, ANTHROPIC_API_KEY, NVIDIA_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY
NVIDIA (build.nvidia.com, free endpoints) serves open models through an OpenAI-compatible API;
Google AI Studio serves the Gemini models (free tier).
LLMs of the evaluation (paid APIs, three families, eval_config.json): extraction by majority vote over Mistral
Large 3 (Mistral), Gemini 2.5 Flash (Google) and Claude Sonnet 4.5 (Anthropic); our approach with Mistral Large 3
alone for the comparison with the baselines; negotiation by Mistral Large 3.
"template" = no LLM (deterministic negotiator).
Any model of a supported provider can also be given as "provider:model", e.g. "groq:llama-3.3-70b-versatile"
or "mistral:mistral-small-latest" (providers: openai, groq, mistral, anthropic, nvidia, google).
"""
from __future__ import annotations

import os

# The project does not use LangSmith: tracing is switched off, whatever the Windows settings
# (set SELCOMP_LANGSMITH=1 to keep it).
if os.environ.get("SELCOMP_LANGSMITH") != "1":
    for _v in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_TRACING_V2"):
        os.environ[_v] = "false"
    for _v in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        os.environ.pop(_v, None)

LLM_CONFIGS = {
    "template":           None,
    "GPT_4o_mini":        {"provider": "openai",  "model": "gpt-4o-mini"},
    "Groq_gpt120b":       {"provider": "groq",    "model": "openai/gpt-oss-120b"},
    "Groq_gpt20b":        {"provider": "groq",    "model": "openai/gpt-oss-20b"},
    "Groq_llama70b":      {"provider": "groq",    "model": "llama-3.3-70b-versatile"},
    "mistral-large-2512": {"provider": "mistral", "model": "mistral-large-2512"},
    "mistral-medium":     {"provider": "mistral", "model": "mistral-medium-latest"},
    "mistral-small":      {"provider": "mistral", "model": "mistral-small-latest"},
    "Claude_sonnet":      {"provider": "anthropic", "model": "claude-sonnet-4-5"},
    "Claude_haiku":       {"provider": "anthropic", "model": "claude-haiku-4-5"},
    "NVIDIA_mistral_large": {"provider": "nvidia", "model": "mistralai/mistral-large-3-675b-instruct-2512"},
    "NVIDIA_deepseek":    {"provider": "nvidia",  "model": "deepseek-ai/deepseek-v4.1-flash"},
    "NVIDIA_nemotron":    {"provider": "nvidia",  "model": "nvidia/nemotron-3-ultra-550b-a55b"},
    "Gemini_flash":       {"provider": "google",  "model": "gemini-2.5-flash"},
    "Gemini_pro":         {"provider": "google",  "model": "gemini-3.1-pro-preview"},  # 2.5 Pro closed to new users
    "DeepSeek_flash":     {"provider": "deepseek", "model": "deepseek-flash"},     # DeepSeek-V4.1-Flash
}
PROVIDERS = ("openai", "groq", "mistral", "anthropic", "nvidia", "google", "deepseek")
EXTRACTION_LLMS = ["mistral-large-2512", "Gemini_flash", "Claude_sonnet"]
NEGOTIATION_LLM = "mistral-large-2512"
SLOW_PROVIDER_TIMEOUT = {"nvidia": 240.0, "google": 600.0, "deepseek": 180.0, "mistral": 300.0}  # reasoning models answer slowly
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TEMPERATURE = 0.1          # as in our previous work, for output stability


def resolve(key: str):
    """Configuration of an LLM, from a name of LLM_CONFIGS or a "provider:model" string."""
    if key in LLM_CONFIGS:
        return LLM_CONFIGS[key]
    provider, sep, model = key.partition(":")
    if sep and provider.lower() in PROVIDERS and model.strip():
        return {"provider": provider.lower(), "model": model.strip()}
    raise ValueError(f"Unknown LLM: {key}. Choices: {list(LLM_CONFIGS)} or provider:model "
                     f"with provider in {list(PROVIDERS)}")


def create_llm(key: str, temperature: float = TEMPERATURE, timeout: float = 60.0):
    cfg = resolve(key)
    if cfg is None:
        return None
    timeout = max(timeout, SLOW_PROVIDER_TIMEOUT.get(cfg["provider"], 0.0))
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)          # the .env file always wins, and is read again at each run
    except ImportError:
        pass
    if cfg["provider"] == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=cfg["model"], temperature=temperature, max_retries=1, timeout=timeout)
    if cfg["provider"] == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=cfg["model"], temperature=temperature, max_retries=1, timeout=timeout)
    if cfg["provider"] == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=cfg["model"], temperature=temperature, max_retries=1, timeout=timeout)
    if cfg["provider"] == "nvidia":
        from langchain_openai import ChatOpenAI
        key_ = os.environ.get("NVIDIA_API_KEY")
        if not key_:
            raise RuntimeError("NVIDIA_API_KEY is missing in the .env file")
        return ChatOpenAI(model=cfg["model"], temperature=temperature, max_retries=1, timeout=timeout,
                          base_url=NVIDIA_BASE_URL, api_key=key_, max_tokens=4096)
    if cfg["provider"] == "deepseek":
        from langchain_openai import ChatOpenAI
        key_ = os.environ.get("DEEPSEEK_API_KEY")
        if not key_:
            raise RuntimeError("DEEPSEEK_API_KEY is missing in the .env file")
        return ChatOpenAI(model=cfg["model"], temperature=temperature, max_retries=1, timeout=timeout,
                          base_url=DEEPSEEK_BASE_URL, api_key=key_, max_tokens=8192)
    if cfg["provider"] == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key_ = os.environ.get("GOOGLE_API_KEY")
        if not key_:
            raise RuntimeError("GOOGLE_API_KEY is missing in the .env file")
        # Google: "For all Gemini 3 models, we strongly recommend keeping the temperature parameter at its
        # default value of 1.0" (lower values may degrade them): the recommended value is used for these models
        t = 1.0 if cfg["model"].startswith("gemini-3") else temperature
        return ChatGoogleGenerativeAI(model=cfg["model"], temperature=t, max_retries=1, timeout=timeout,
                                      google_api_key=key_)
    if cfg["provider"] == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=cfg["model"], temperature=temperature, max_retries=1,
                             default_request_timeout=timeout, max_tokens=4096)
    raise ValueError(cfg["provider"])
