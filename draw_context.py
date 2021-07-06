from collections import UserDict
from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle


class DrawContext(UserDict):
    _axis: Axes

    def __init__(self, axis) -> None:
        super().__init__()
        self._axis = axis

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
