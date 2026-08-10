"""Cancellation of long-running fleet commands.

A fleet command like upload_episodes or process_dataset runs for minutes. When the
operator cancels the dataset build, the fleet sends a `cancel_dataset` naming the
command ids to abort — and the device needs somewhere to record "this command id is
cancelled" that the running handler can see.

That is all this registry is: a bounded set of cancelled command ids.

Two properties matter, and both come from how the relay executes commands:

* A cancel can land BEFORE its target command starts (the relay's worker runs
  commands one at a time, so an upload can sit queued behind another). A handler
  must therefore check `is_cancelled` on entry, not only mid-run — otherwise a
  cancelled upload would still start.
* A cancel must never be executed by that same serialized worker, or it would
  queue behind the very upload it is meant to stop and only run once that upload
  finished. The relay dispatches `cancel_dataset` on a fast path (see
  relay_client._FAST_PATH_TYPES); this registry is the hand-off between that fast
  path and the running handler.

Cancelled ids are kept in a bounded FIFO: a cancel whose target never arrives (the
fleet dropped the command before delivering it) must not leak, and the ids stay
long enough for any queued command to see its own.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterable

logger = logging.getLogger("grabette.cancel")

# Plenty for the handful of commands in flight at once, small enough that the
# linear membership test is free.
_MAX_REMEMBERED = 256


class CancelRegistry:
    """Command ids the fleet asked us to abort. Not thread-safe by design: the
    relay and every handler live on the same asyncio loop, and these are plain
    non-awaiting operations, so no lock can interleave them."""

    def __init__(self, max_remembered: int = _MAX_REMEMBERED) -> None:
        self._ids: deque[str] = deque(maxlen=max_remembered)

    def cancel(self, command_ids: Iterable[str]) -> list[str]:
        """Mark command ids as cancelled. Returns the ones newly marked."""
        added = []
        for cid in command_ids:
            if not cid or cid in self._ids:
                continue
            self._ids.append(cid)
            added.append(cid)
        if added:
            logger.info("cancel requested for %s", ", ".join(added))
        return added

    def is_cancelled(self, command_id: str | None) -> bool:
        return bool(command_id) and command_id in self._ids

    def clear(self, command_id: str | None) -> None:
        """Forget a command id, once nothing can act on it any more (the relay
        calls this when a command finishes)."""
        if command_id and command_id in self._ids:
            self._ids.remove(command_id)


_registry: CancelRegistry | None = None


def get_cancel_registry() -> CancelRegistry:
    global _registry
    if _registry is None:
        _registry = CancelRegistry()
    return _registry
