"""
Dependency-free env helper for VBL security hardening.

- `effective_host`: app/MCP listen host resolves to loopback unless explicitly
  overridden, so services are not exposed by default.
- `require_secret`: fails CLOSED on unset secrets — removes the old fail-open
  fallback tokens (`local-dev-token`, `local-dev-key`, etc.).
"""
from __future__ import annotations

import hmac
import os


def effective_host(env_key: str, fallback: str = "127.0.0.1") -> str:
    """Return the env var value (trimmed) if set and non-empty, else `fallback`.

    Defaults to loopback so any service that uses this helper is not bound to
    all interfaces unless the operator opts in explicitly.
    """
    value = (os.environ.get(env_key) or "").strip()
    return value if value else fallback


def require_secret(env_key: str, hint: str = "") -> str:
    """Return the env var's value or raise RuntimeError (fail closed).

    Never returns a hardcoded default — an absent secret is a configuration
    error, not something to paper over with a known-insecure token.
    """
    value = (os.environ.get(env_key) or "").strip()
    if not value:
        message = f"Required secret {env_key} is not set."
        if hint:
            message += f" {hint}"
        raise RuntimeError(message)
    return value


def secrets_equal(a: str | None, b: str | None) -> bool:
    """Constant-time string comparison for secrets (credentials-safe).

    Uses hmac.compare_digest so timing does not leak the secret length/content.
    Non-str inputs are treated as never-equal (fail closed) so None/bytes never
    raise or compare truthy.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return hmac.compare_digest(a, b)
