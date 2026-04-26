"""
Tenant context — per-request ContextVar that lets storage modules return
tenant-scoped file paths (data/tenants/{tenant_id}/<file>) instead of the
shared DATA_DIR root.

Set by main.py middleware on every authenticated request based on the
caller's session.tenant_id. When unset (owner / system / unauth), helpers
fall back to the legacy DATA_DIR path.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from pathlib import Path

DATA_DIR    = Path(__file__).parent / "data"
TENANTS_DIR = DATA_DIR / "tenants"
DATA_DIR.mkdir(exist_ok=True)
TENANTS_DIR.mkdir(exist_ok=True)

_current_tenant: ContextVar[str | None] = ContextVar("current_tenant", default=None)


def set_tenant(tenant_id: str | None) -> Token:
    return _current_tenant.set(tenant_id or None)


def reset_tenant(token: Token) -> None:
    _current_tenant.reset(token)


def current_tenant() -> str | None:
    return _current_tenant.get()


def _safe(tenant_id: str) -> str:
    return "".join(c for c in tenant_id if c.isalnum() or c in "-_") or "tenant_default"


def tenant_dir(tenant_id: str | None = None) -> Path:
    """Folder for the given tenant (or current request tenant). Created if missing."""
    tid = tenant_id or _current_tenant.get()
    if not tid:
        return DATA_DIR  # legacy root (owner / system / unauth)
    p = TENANTS_DIR / _safe(tid)
    p.mkdir(parents=True, exist_ok=True)
    return p


def tenant_data_path(filename: str, tenant_id: str | None = None) -> Path:
    """
    Tenant-scoped file path (e.g. data/tenants/acme/calls.json) when the
    request belongs to a tenant; otherwise legacy data/<filename>.
    """
    return tenant_dir(tenant_id) / filename
