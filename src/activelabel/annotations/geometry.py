"""Geometry shared by the annotation readers (XML files) and the CVAT adapter."""

from __future__ import annotations

import math


def enclosing_box(points: list[float], rotation: float) -> list[float]:
    """The axis-aligned box ``[x0, y0, x1, y1]`` around a rectangle turned by ``rotation`` about its centre.

    CVAT keeps the corners of the rectangle before the turn (``xtl, ytl, xbr, ybr``) and the angle apart,
    in degrees. Detection works on axis-aligned boxes, so a turned rectangle becomes the smallest box that
    holds its four corners (the same for either direction of the turn).

    Args:
        points: ``[xtl, ytl, xbr, ybr]`` of the rectangle before the turn, in pixels.
        rotation: The angle in degrees.

    Returns:
        The enclosing box in pixels; ``points`` itself when there is no turn.
    """
    x0, y0, x1, y1 = (float(v) for v in points[:4])
    turn = math.radians(float(rotation or 0) % 360)
    if not turn:
        return [x0, y0, x1, y1]
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
    ex = hw * abs(math.cos(turn)) + hh * abs(math.sin(turn))
    ey = hw * abs(math.sin(turn)) + hh * abs(math.cos(turn))
    return [cx - ex, cy - ey, cx + ex, cy + ey]
