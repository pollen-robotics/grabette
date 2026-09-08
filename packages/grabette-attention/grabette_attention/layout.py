"""Geometry and token bookkeeping. Pure numpy; no torch, no lerobot.

This module is where multi-camera genericity is won or lost. Every number it
returns is derived from shapes and config passed in by the caller. Nothing here
may assume a patch count, a grid size, or how many cameras exist.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LetterboxGeometry:
    """Maps a letterboxed model input back to the source frame's pixels.

    Mirrors lerobot's `resize_with_pad_torch`: the aspect ratio is preserved,
    the shorter axis is padded symmetrically, and an odd leftover pixel goes to
    the bottom (or right). Padding is black, which after the policy's `*2-1`
    becomes -1, so those regions carry no image content and their attention must
    not be plotted.
    """

    src_h: int
    src_w: int
    dst_h: int
    dst_w: int
    ratio: float
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    @classmethod
    def from_shapes(
        cls, src_hw: tuple[int, int], dst_hw: tuple[int, int]
    ) -> "LetterboxGeometry":
        src_h, src_w = src_hw
        dst_h, dst_w = dst_hw
        # max(), so the whole source fits inside the target and we pad rather
        # than crop. This is lerobot's choice, not ours to change.
        ratio = max(src_w / dst_w, src_h / dst_h)
        resized_h = int(src_h / ratio)
        resized_w = int(src_w / ratio)
        pad_top, rem_h = divmod(dst_h - resized_h, 2)
        pad_left, rem_w = divmod(dst_w - resized_w, 2)
        return cls(
            src_h=src_h,
            src_w=src_w,
            dst_h=dst_h,
            dst_w=dst_w,
            ratio=ratio,
            pad_top=pad_top,
            pad_bottom=pad_top + rem_h,
            pad_left=pad_left,
            pad_right=pad_left + rem_w,
        )

    def to_source_pixel(self, u: float, v: float) -> tuple[float, float]:
        """Letterboxed pixel (u across, v down) -> source pixel (x, y)."""
        return ((u - self.pad_left) * self.ratio, (v - self.pad_top) * self.ratio)

    def content_rows(self, patch: int) -> tuple[int, int]:
        """Half-open range of grid rows that carry image content.

        A row that is only partly padded is KEPT, which is the conservative
        choice: we would rather show a slightly-too-tall map than silently drop
        real content.
        """
        first = self.pad_top // patch
        content_end = self.dst_h - self.pad_bottom
        last = -(-content_end // patch)  # ceiling division
        return first, last

    def content_cols(self, patch: int) -> tuple[int, int]:
        """Half-open range of grid columns that carry image content."""
        first = self.pad_left // patch
        content_end = self.dst_w - self.pad_right
        last = -(-content_end // patch)
        return first, last
