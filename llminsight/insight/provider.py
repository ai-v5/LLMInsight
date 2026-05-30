"""Pluggable LLM provider for the insight layer.

Design constraints (from the project plan):

- DISTRIBUTABLE: pure stdlib (urllib) — no third-party SDK — so LLMInsight runs
  anywhere it's shipped, not only inside Claude Code.
- PRIVACY-FIRST: only the structured metric *summary* (KB-level JSON, see
  summarizer.py) is ever sent. The raw 104MB trace never leaves the machine.
- DISABLED BY DEFAULT: the network call is a no-op unless explicitly enabled
  (LLMINSIGHT_LLM_ENABLED=1). With it off, the insight panel renders the
  rule-engine cards verbatim — full graceful degradation.
- PLUGGABLE: OpenAI-compatible by default (covers GLM-4-Flash / DeepSeek /
  Ollama / vLLM / SiliconFlow / OpenRouter), plus an Anthropic adapter. Switch
  with one env var / config field; nothing is locked to a single vendor.

Toggle / override via environment:
  LLMINSIGHT_LLM_ENABLED      "1" to turn the call on (default off)
  LLMINSIGHT_LLM_PROVIDER     glm | deepseek | openai | siliconflow | ollama | anthropic
  LLMINSIGHT_LLM_MODEL        override model id
  LLMINSIGHT_LLM_BASE_URL     override endpoint
  LLMINSIGHT_LLM_API_KEY      key (else falls back to secret/api_key.txt)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

# Provider presets: endpoint + default model + wire protocol ("openai"|"anthropic").
PRESETS = {
    "glm":         {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash",  "kind": "openai"},
    "deepseek":    {"base_url": "https://api.deepseek.com",             "model": "deepseek-chat", "kind": "openai"},
    "openai":      {"base_url": "https://api.openai.com/v1",            "model": "gpt-4o-mini",   "kind": "openai"},
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1",        "model": "Qwen/Qwen2.5-7B-Instruct", "kind": "openai"},
    "ollama":      {"base_url": "http://localhost:11434/v1",            "model": "qwen2.5",       "kind": "openai"},
    "anthropic":   {"base_url": "https://api.anthropic.com",            "model": "claude-3-5-haiku-latest", "kind": "anthropic"},
}

DEFAULT_PROVIDER = "glm"
_NO_KEY_PROVIDERS = {"ollama"}  # local servers need no API key


def _read_key_file() -> Optional[str]:
    """Read secret/api_key.txt (repo-local), if present. Never logged."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    path = os.path.join(repo, "secret", "api_key.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip() or None
        except OSError:
            return None
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = DEFAULT_PROVIDER
    kind: str = "openai"
    base_url: str = PRESETS[DEFAULT_PROVIDER]["base_url"]
    model: str = PRESETS[DEFAULT_PROVIDER]["model"]
    api_key: Optional[str] = None
    timeout: float = 90.0          # free GLM-4-Flash latency is variable (saw ~45s)
    max_tokens: int = 1500
    temperature: float = 0.3
    has_key: bool = False

    def public(self) -> dict:
        """Safe-to-serialize view — NEVER leaks the key itself."""
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "has_key": self.has_key,
            "timeout": self.timeout,
        }


def load_config() -> LLMConfig:
    provider = os.environ.get("LLMINSIGHT_LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    preset = PRESETS.get(provider, PRESETS[DEFAULT_PROVIDER])
    base_url = os.environ.get("LLMINSIGHT_LLM_BASE_URL", preset["base_url"]).rstrip("/")
    model = os.environ.get("LLMINSIGHT_LLM_MODEL", preset["model"])
    kind = os.environ.get("LLMINSIGHT_LLM_KIND", preset["kind"])
    key = os.environ.get("LLMINSIGHT_LLM_API_KEY") or _read_key_file()
    return LLMConfig(
        enabled=_env_bool("LLMINSIGHT_LLM_ENABLED", False),
        provider=provider, kind=kind, base_url=base_url, model=model,
        api_key=key, has_key=bool(key),
        timeout=float(os.environ.get("LLMINSIGHT_LLM_TIMEOUT", "90")),
        max_tokens=int(os.environ.get("LLMINSIGHT_LLM_MAX_TOKENS", "1500")),
        temperature=float(os.environ.get("LLMINSIGHT_LLM_TEMPERATURE", "0.3")),
    )


class Provider:
    """Thin OpenAI-compatible / Anthropic client. The call is gated behind
    `available` so a disabled or unconfigured provider is a hard no-op."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_config()

    @property
    def available(self) -> bool:
        c = self.config
        needs_key = c.provider not in _NO_KEY_PROVIDERS
        return bool(c.enabled and (c.has_key or not needs_key))

    def status(self) -> dict:
        c = self.config
        reason = None
        if not c.enabled:
            reason = "LLM 调用默认关闭（设环境变量 LLMINSIGHT_LLM_ENABLED=1 开启）"
        elif not self.available:
            reason = "已开启但缺少 API key（放到 secret/api_key.txt 或设 LLMINSIGHT_LLM_API_KEY）"
        d = c.public()
        d["available"] = self.available
        d["reason"] = reason
        return d

    def complete(self, system: str, user: str) -> str:
        if not self.available:
            raise RuntimeError("LLM provider disabled or unconfigured (no-op by default)")
        if self.config.kind == "anthropic":
            return self._anthropic(system, user)
        return self._openai(system, user)

    # -- wire protocols -----------------------------------------------------
    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _openai(self, system: str, user: str) -> str:
        c = self.config
        body = self._post(
            f"{c.base_url}/chat/completions",
            {"Content-Type": "application/json", "Authorization": f"Bearer {c.api_key}"},
            {
                "model": c.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": c.temperature,
                "max_tokens": c.max_tokens,
                "stream": False,
            },
        )
        return body["choices"][0]["message"]["content"]

    def _anthropic(self, system: str, user: str) -> str:
        c = self.config
        body = self._post(
            f"{c.base_url}/v1/messages",
            {"Content-Type": "application/json", "x-api-key": c.api_key or "",
             "anthropic-version": "2023-06-01"},
            {
                "model": c.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": c.max_tokens,
                "temperature": c.temperature,
            },
        )
        parts = body.get("content", [])
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def get_provider() -> Provider:
    return Provider()
