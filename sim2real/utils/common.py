"""Shared scaffolding for the robot runtime and the policy operator.

env loading, LiveKit token minting, and an async fps pacer -- plumbing both entry
scripts (robot/run.py, policy/run.py) need. Pure Python over python-dotenv +
livekit-api. Entry points add the sim2real root to sys.path and `from utils.common import ...`.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import pathlib
import time
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.room import RoomConfiguration


def load_env(start: Optional[pathlib.Path] = None) -> None:
    """Load `.env` walking up from `start` to root (plus cwd). Nearest `.env`
    wins (`override=False`); `.env.local` always overrides. Pass the calling
    script's directory."""
    start = (start or pathlib.Path.cwd()).resolve()
    seen: set[pathlib.Path] = set()
    for d in (start, *start.parents, pathlib.Path.cwd().resolve()):
        if d in seen:
            continue
        seen.add(d)
        if (f := d / ".env").exists():
            load_dotenv(f, override=False)
        if (f := d / ".env.local").exists():
            load_dotenv(f, override=True)


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.environ.get(name) or default
    if required and not val:
        raise RuntimeError(f"{name} must be set (see .env.example)")
    return val  # type: ignore[return-value]


def mint_token(
    identity: str,
    room: str,
    *,
    name: str | None = None,
    attributes: dict[str, str] | None = None,
) -> str:
    """Mint a 6h join token for `identity` in `room`. `name` sets the display
    name; `attributes` are published as readable participant attributes (a web-ui
    uses them to discover the policy's control descriptor)."""
    grants = api.VideoGrants(
        room_join=True, room=room, can_publish=True, can_publish_data=True,
        can_subscribe=True, can_update_own_metadata=True,
    )
    room_cfg = RoomConfiguration(name=room, min_playout_delay=0, max_playout_delay=1)
    builder = api.AccessToken(
        env("LIVEKIT_API_KEY", required=True), env("LIVEKIT_API_SECRET", required=True)
    ).with_identity(identity).with_grants(grants).with_room_config(room_cfg)
    if name:
        builder = builder.with_name(name)
    if attributes:
        builder = builder.with_attributes(attributes)
    return builder.with_ttl(datetime.timedelta(hours=6)).to_jwt()


async def pace(fps: int) -> AsyncIterator[int]:
    """Async tick generator at `fps`; yields a monotonically increasing index."""
    interval, next_tick, i = 1.0 / fps, time.perf_counter(), 0
    while True:
        yield i
        i += 1
        next_tick += interval
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()
