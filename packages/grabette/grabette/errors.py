"""Small shared error helpers."""

from __future__ import annotations


def exc_text(e: BaseException) -> str:
    """Never-empty description of an exception.

    `str(asyncio.TimeoutError())` is the EMPTY STRING, so interpolating an
    exception straight into a message can report only "processing failed:" —
    leaving the operator nothing to act on, and the failure that says the least
    is the most common one. Fall back to the class name, which at least names
    the failure mode.
    """
    return str(e).strip() or type(e).__name__
