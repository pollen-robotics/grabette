"""A device handed back to an HF account must re-report its tasks to that account.

The fleet keeps an in-memory entry per (account, device). When a device is
logged into account B and then back into A, A's fleet may still hold the entry
from A's earlier session — so the poll succeeds and, without this, the device
never re-registers: A keeps showing the task report from before B, missing every
episode recorded since.
"""
import asyncio

import pytest

from grabette.relay_client import RelayClient


def _run(tokens: list[str]) -> list[str]:
    """Drive the relay loop once per token; return the token of each register."""
    feed = iter(tokens)
    current = {"t": None}
    registers: list[str] = []

    def token_provider():
        return current["t"]

    client = RelayClient("http://fleet", token_provider, "dev-1", poll_interval=0)

    async def _register(session, token):
        registers.append(token)
        client._registered_token = token

    async def _poll(session, token):
        try:
            current["t"] = next(feed)
        except StopIteration:
            raise asyncio.CancelledError
        return []

    client._register = _register
    client._poll = _poll
    current["t"] = next(feed)

    async def _handler(cmd):
        return {}

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(client.run(_handler))
    return registers


def test_same_token_registers_once():
    assert _run(["A", "A", "A"]) == ["A"]


def test_account_switch_re_registers():
    # A, then someone logs B in, then A again: each switch re-reports.
    assert _run(["A", "A", "B", "B", "A"]) == ["A", "B", "A"]
