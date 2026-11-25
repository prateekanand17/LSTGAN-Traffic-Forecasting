"""Provider-agnostic chat client with full call logging (task P1.5).

- OpenAI-compatible endpoint (Groq, a local vLLM/Ollama server, ...); base URLs and model IDs from pipeline.config.
- No silent mock: a missing key or a failed call raises LLMError. Dry runs must be requested explicitly
  (dry_run=True) and every dry-run record is labelled "dry_run": true in the log.
- Every call (prompt, raw response, parsed JSON or parse error, tokens, latency, model, attempt) is appended to
  <log_dir>/calls.jsonl.
- Provider-side JSON validation failures (Groq HTTP 400 `json_validate_failed`, finding M22; seen when a reasoning
  model runs out of completion tokens) are returned as {"_parse_error": ...} like any malformed JSON, so the
  caller's retry gate handles them instead of aborting the scenario. Other API errors still raise LLMError.
- `usage` keeps running totals (calls, tokens, latency) so the orchestrator can report per-scenario cost.
"""
import json
import os
import threading
import time
from typing import Callable, Optional

from pipeline import config as C


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, log_dir: Optional[str] = None, dry_run: bool = False,
                 dry_run_fn: Optional[Callable[[str, str, dict], dict]] = None,
                 provider: str = C.LLM_PROVIDER):
        self.dry_run = dry_run
        self.dry_run_fn = dry_run_fn
        self.provider = provider
        self.log_path = os.path.join(log_dir, "calls.jsonl") if log_dir else None
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "parse_errors": 0}
        self.client = None
        if not dry_run:
            try:
                from dotenv import load_dotenv
                load_dotenv(os.path.join(C.ROOT, ".env"))
            except ImportError:
                pass
            key_env = C.LLM_API_KEY_ENV[provider]
            key = os.environ.get(key_env)
            if not key and provider != "local":
                raise LLMError(f"{key_env} is not set; refusing to run without an LLM (use dry_run=True explicitly)")
            from openai import OpenAI
            self.client = OpenAI(base_url=C.LLM_BASE_URLS[provider], api_key=key or "none",
                                 timeout=C.REQUEST_TIMEOUT_S, max_retries=2)

    def _log(self, rec: dict):
        if not self.log_path:
            return
        with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _count(self, rec: dict):
        with self._lock:
            u = self.usage
            u["calls"] += 1
            u["prompt_tokens"] += rec.get("prompt_tokens") or 0
            u["completion_tokens"] += rec.get("completion_tokens") or 0
            u["latency_s"] += rec.get("latency_s") or 0.0
            u["parse_errors"] += int("_parse_error" in (rec.get("parsed") or {}))

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.usage)

    def chat_json(self, role: str, model: str, system: str, user: str, temperature: float,
                  max_tokens: int, meta: dict, dry_value: Optional[dict] = None) -> dict:
        """Return the parsed JSON object. Raises LLMError on transport errors; JSON errors are returned as
        {"_parse_error": "..."} so the caller's structural gate and retry loop can handle them.
        dry_value: output to return in dry-run mode (callers pass the deterministic stand-in they want tested);
        ignored for live calls."""
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": model, "provider": self.provider,
               "temperature": temperature, "system": system, "user": user, "dry_run": self.dry_run, **meta}
        t0 = time.time()
        if self.dry_run:
            if dry_value is not None:
                out = dry_value
            elif self.dry_run_fn is not None:
                out = self.dry_run_fn(role, user, meta)
            else:
                raise LLMError("dry_run=True requires dry_value or dry_run_fn")
            rec.update(raw=json.dumps(out), parsed=out, latency_s=0.0, prompt_tokens=None, completion_tokens=None)
            self._log(rec)
            self._count(rec)
            return out
        try:
            resp = self.client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
            )
        except Exception as e:  # transport / API error: log and raise, never mock
            if is_json_validate_failure(e):   # M22: model output rejected by the provider's JSON check
                parsed = {"_parse_error": f"provider json_validate_failed: {failed_generation(e)}"[:500]}
                rec.update(error=f"{type(e).__name__}: {e}"[:2000], parsed=parsed, latency_s=round(time.time() - t0, 3))
                self._log(rec)
                self._count(rec)
                return parsed
            rec.update(error=f"{type(e).__name__}: {e}", latency_s=round(time.time() - t0, 3))
            self._log(rec)
            raise LLMError(rec["error"]) from e
        raw = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        rec.update(raw=raw, latency_s=round(time.time() - t0, 3),
                   prompt_tokens=getattr(usage, "prompt_tokens", None),
                   completion_tokens=getattr(usage, "completion_tokens", None),
                   finish_reason=resp.choices[0].finish_reason, response_model=getattr(resp, "model", None))
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("top-level JSON is not an object")
        except Exception as e:
            parsed = {"_parse_error": f"{type(e).__name__}: {e}"}
        rec["parsed"] = parsed
        self._log(rec)
        self._count(rec)
        if C.RATE_LIMIT_SLEEP_S:
            time.sleep(C.RATE_LIMIT_SLEEP_S)
        return parsed


def _error_body(e: Exception) -> dict:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        return body.get("error", body) if isinstance(body.get("error", body), dict) else body
    return {}


def is_json_validate_failure(e: Exception) -> bool:
    """True for the provider-side JSON validation error (Groq: HTTP 400, code json_validate_failed)."""
    if getattr(e, "status_code", None) not in (400, None):
        return False
    return _error_body(e).get("code") == "json_validate_failed" or "json_validate_failed" in str(e)


def failed_generation(e: Exception) -> str:
    return str(_error_body(e).get("failed_generation", ""))[:300]
