"""
Sequential outbound campaign: dial one prospect at a time, wait for call to end,
then optional spacing delay before the next dial.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# call_control_id -> Event set when Telnyx sends call.hangup
call_end_events: dict[str, asyncio.Event] = {}


def signal_call_ended(call_control_id: str) -> None:
    ev = call_end_events.pop(call_control_id, None)
    if ev and not ev.is_set():
        ev.set()


def normalize_phone(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    if s.startswith("+"):
        return "+" + digits
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits[0] == "1":
        return "+" + digits
    if len(digits) >= 10:
        return "+" + digits
    return None


def prospect_display_name(p: dict[str, Any]) -> str:
    fn = (p.get("first_name") or "").strip()
    ln = (p.get("last_name") or "").strip()
    name = f"{fn} {ln}".strip()
    return name if name else "there"


@dataclass
class CampaignState:
    status: str = "idle"  # idle | running | paused | stopped | completed
    index: int = 0
    total: int = 0
    spacing_seconds: float = 60.0
    last_error: str = ""
    last_to: str = ""
    last_call_control_id: str = ""
    skipped: list[str] = field(default_factory=list)


state = CampaignState()
_runner_task: asyncio.Task[Any] | None = None


def is_busy() -> bool:
    return state.status == "running"


async def run_campaign(
    queue: list[dict[str, Any]],
    spacing_seconds: float,
    dial: Callable[[dict[str, Any]], Awaitable[str | None]],
) -> None:
    """
    dial(prospect) -> call_control_id or None if failed.
    Waits until hangup before pacing to next.
    """
    global state
    state = CampaignState(
        status="running",
        index=0,
        total=len(queue),
        spacing_seconds=spacing_seconds,
    )

    for i, prospect in enumerate(queue):
        state.index = i
        if state.status == "stopped":
            logger.info("Campaign stopped by user")
            break
        while state.status == "paused":
            await asyncio.sleep(0.5)
            if state.status == "stopped":
                break
        if state.status == "stopped":
            break

        phone = normalize_phone(prospect.get("phone"))
        if not phone:
            state.skipped.append(prospect_display_name(prospect) + " (no phone)")
            continue

        cc_id: str | None = None
        try:
            cc_id = await dial(prospect)
        except Exception as e:
            state.last_error = str(e)
            logger.exception("Campaign dial failed: %s", e)
            continue

        if not cc_id:
            state.skipped.append(prospect_display_name(prospect) + " (dial failed)")
            continue

        state.last_to = phone
        state.last_call_control_id = cc_id

        ev = asyncio.Event()
        call_end_events[cc_id] = ev
        try:
            await asyncio.wait_for(ev.wait(), timeout=7200.0)
        except asyncio.TimeoutError:
            logger.warning("Campaign: timeout waiting for hangup %s", cc_id)
        finally:
            call_end_events.pop(cc_id, None)

        if state.status == "stopped":
            break

        if i < len(queue) - 1 and state.status == "running":
            await asyncio.sleep(max(0.0, float(spacing_seconds)))

    if state.status == "running":
        state.status = "completed"
    logger.info("Campaign finished: %s", state.status)


def start_campaign(
    queue: list[dict[str, Any]],
    spacing_seconds: float,
    dial: Callable[[dict[str, Any]], Awaitable[str | None]],
) -> bool:
    global _runner_task
    if _runner_task is not None and not _runner_task.done():
        return False
    if state.status == "running":
        return False

    async def _go() -> None:
        try:
            await run_campaign(queue, spacing_seconds, dial)
        finally:
            global _runner_task
            _runner_task = None

    _runner_task = asyncio.create_task(_go())
    return True


def pause_campaign() -> None:
    if state.status == "running":
        state.status = "paused"


def resume_campaign() -> None:
    if state.status == "paused":
        state.status = "running"


def stop_campaign() -> None:
    state.status = "stopped"
