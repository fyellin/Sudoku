from collections import UserDict
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

import numpy as np

if TYPE_CHECKING:
    from feature import Square
    from cell import House

class DrawContext(UserDict):
    _axis: Axes
    done: bool
    result: bool

    def __init__(self, axis, *, done: bool, result: bool) -> None:
        super().__init__()
        self._axis = axis
        self.done = done
        self.result = result

    def draw_circle(self, center: tuple[float, float], radius: float, **args: Any) -> None:
        self._axis.add_patch(Circle(center, radius=radius, **args))

    def draw_text(self, x: float, y: float, text: str, **args: Any) -> None:
        self._axis.text(x, y, text, **args)

    def draw_rectangle(self, corner: tuple[float, float], *, width: float, height: float, **args: Any):
        self._axis.add_patch(Rectangle(corner, width=width, height=height, **args))

    def draw_rectangles(self, points: Sequence[tuple[int, int]], **args: Any):
        args = {'color': 'lightgrey', 'fill': True, **args}
        for row, column in points:
            self._axis.add_patch(Rectangle((column, row), width=1, height=1, **args))

    def draw_line(self, points: Sequence[tuple[int, int]], *, closed: bool = False, **kwargs: Any) -> None:
        ys = [row + .5 for row, _ in points]
        xs = [column + .5 for _, column in points]
        if closed:
            ys.append(ys[0])
            xs.append(xs[0])
        self._axis.plot(xs, ys, **{'color': 'black', **kwargs})

    def plot(self, xs, ys, **args: Any):
        self._axis.plot(xs, ys, **args)

    def arrow(self, x: float, y: float, dx: float, dy: float, **args: Any):
        self._axis.arrow(x, y, dx, dy, **args)

    def add_fancy_bbox(self, center, width, height, **args: Any):
        self._axis.add_patch(FancyBboxPatch(center, width=width, height=height, **args))

    def draw_outside(self, value: Any, htype: 'House.Type', row_or_column: int, *,
                     is_right: bool = False, padding: float = 0, **args: Any):
        args = {'fontsize': 20, 'weight': 'bold', **args}

        if htype == House.Type.ROW:
            if not is_right:
                self.draw_text(.9 - padding, row_or_column + .5, str(value),
                               verticalalignment='center', horizontalalignment='right', **args)
            else:
                self.draw_text(10.1 + padding, row_or_column + .5, str(value),
                               verticalalignment='center', horizontalalignment='left', **args)
        else:
            if not is_right:
                self.draw_text(row_or_column + .5, .9 - padding, str(value),
                               verticalalignment='bottom', horizontalalignment='center', **args)
            else:
                self.draw_text(row_or_column + .5, 10.1 + padding, str(value),
                               verticalalignment='top', horizontalalignment='center', **args)

    def draw_outline(self, squares: Sequence['Square'], *,
                     inset: float = .1, **args: Any) -> None:
        args = {'color': 'black', 'linewidth': 2, 'linestyle': "dotted", **args}
        squares_set = set(squares)

        # A wall is identified by the square it is in, and the direction you'd be facing from the center of that
        # square to see the wall.  A wall separates a square inside of "squares" from a square out of it.
        walls = {(row, column, dr, dc)
                 for row, column in squares for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
                 if (row + dr, column + dc) not in squares_set}

        while walls:
            start_wall = current_wall = next(iter(walls))  # pick some wall
            points: list[np.ndarray] = []

            while True:
                # Find the connecting point between the current wall and the wall to the right and add it to our
                # set of points

                row, column, ahead_dr, ahead_dc = current_wall  # square, and direction of wall from center
                right_dr, right_dc = ahead_dc, -ahead_dr  # The direction if we turned right

                # Three possible next walls, in order of preference.
                #  1) The wall makes a right turn, staying with the current square
                #  2) The wall continues in its direction, going into the square to our right
                #  3) The wall makes a left turn, continuing in the square diagonally ahead to the right.
                next1 = (row, column, right_dr, right_dc)   # right
                next2 = (row + right_dr, column + right_dc, ahead_dr, ahead_dc)  # straight
                next3 = (row + right_dr + ahead_dr, column + right_dc + ahead_dc, -right_dr, -right_dc)  # left

                # It is possible for next1 and next3 to both be in walls if we have two squares touching diagonally.
                # In that case, we prefer to stay within the same cell, so we prefer next1 to next3.
                next_wall = next(x for x in (next1, next2, next3) if x in walls)
                walls.remove(next_wall)

                if next_wall == next2:
                    # We don't need to add a point if the wall is continuing in the direction it was going.
                    pass
                else:
                    np_center = np.array((row, column)) + .5
                    np_ahead = np.array((ahead_dr, ahead_dc))
                    np_right = np.array((right_dr, right_dc))
                    right_inset = inset if next_wall == next1 else -inset
                    points.append(np_center + (.5 - inset) * np_ahead + (.5 - right_inset) * np_right)

                if next_wall == start_wall:
                    break
                current_wall = next_wall

            points.append(points[0])
            pts = np.vstack(points)
            self.plot(pts[:, 1], pts[:, 0], **args)
