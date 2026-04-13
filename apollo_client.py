"""
Apollo.io REST API — People API Search (master API key, x-api-key header).
Docs: https://docs.apollo.io/reference/people-api-search
"""
from __future__ import annotations

from __future__ import annotations

import logging
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)

APOLLO_BASE = "https://api.apollo.io/api/v1"
SEARCH_PATH = "/mixed_people/api_search"


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "x-api-key": config.APOLLO_API_KEY or "",
    }


def _build_search_params(
    *,
    page: int = 1,
    per_page: int = 25,
    q_keywords: str | None = None,
    person_titles: list[str] | None = None,
    person_locations: list[str] | None = None,
    organization_locations: list[str] | None = None,
    person_seniorities: list[str] | None = None,
    organization_num_employees_ranges: list[str] | None = None,
    q_organization_domains_list: list[str] | None = None,
    include_similar_titles: bool | None = None,
) -> list[tuple[str, Any]]:
    """Apollo expects POST with filter params as query string (arrays use key[])."""
    params: list[tuple[str, Any]] = [
        ("page", max(1, page)),
        ("per_page", min(100, max(1, per_page))),
    ]
    if q_keywords:
        params.append(("q_keywords", q_keywords.strip()))
    if include_similar_titles is not None:
        params.append(("include_similar_titles", str(include_similar_titles).lower()))
    for t in person_titles or []:
        t = (t or "").strip()
        if t:
            params.append(("person_titles[]", t))
    for loc in person_locations or []:
        loc = (loc or "").strip()
        if loc:
            params.append(("person_locations[]", loc))
    for loc in organization_locations or []:
        loc = (loc or "").strip()
        if loc:
            params.append(("organization_locations[]", loc))
    for s in person_seniorities or []:
        s = (s or "").strip()
        if s:
            params.append(("person_seniorities[]", s))
    for r in organization_num_employees_ranges or []:
        r = (r or "").strip()
        if r:
            params.append(("organization_num_employees_ranges[]", r))
    for d in q_organization_domains_list or []:
        d = (d or "").strip()
        if d:
            params.append(("q_organization_domains_list[]", d))
    return params


async def search_people(
    *,
    page: int = 1,
    per_page: int = 25,
    q_keywords: str | None = None,
    person_titles: list[str] | None = None,
    person_locations: list[str] | None = None,
    organization_locations: list[str] | None = None,
    person_seniorities: list[str] | None = None,
    organization_num_employees_ranges: list[str] | None = None,
    q_organization_domains_list: list[str] | None = None,
    include_similar_titles: bool | None = True,
) -> dict[str, Any]:
    config.reload_secrets()
    if not config.APOLLO_API_KEY:
        raise RuntimeError("APOLLO_API_KEY is not set")

    params = _build_search_params(
        page=page,
        per_page=per_page,
        q_keywords=q_keywords,
        person_titles=person_titles,
        person_locations=person_locations,
        organization_locations=organization_locations,
        person_seniorities=person_seniorities,
        organization_num_employees_ranges=organization_num_employees_ranges,
        q_organization_domains_list=q_organization_domains_list,
        include_similar_titles=include_similar_titles,
    )
    url = f"{APOLLO_BASE}{SEARCH_PATH}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=_headers(), params=params)
    if r.status_code >= 400:
        logger.warning("Apollo search failed: %s %s", r.status_code, r.text[:500])
        r.raise_for_status()
    return r.json()


async def test_connection() -> dict[str, Any]:
    """Minimal search to verify API key."""
    data = await search_people(page=1, per_page=1, q_keywords="sales")
    total = data.get("pagination", {}).get("total_entries") or data.get("total_entries")
    return {"ok": True, "sample_total_hint": total}
