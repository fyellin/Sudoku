import abc
import atexit
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence, Callable
from typing import Any, ClassVar, Union, cast, Optional

import numpy as np

from cell import Cell, House
from draw_context import DrawContext
from grid import Grid

Square = tuple[int, int]
CheckFunction = Callable[['Feature'], bool]


class Feature(abc.ABC):
    name: str
    grid: Grid

    def __init__(self, *, name: Optional[str] = None) -> None:
        self.name = name or self.get_default_feature_name()

    def initialize(self, grid) -> None:
        self.grid = grid

    def reset(self) -> None:
        pass

    def get_neighbors(self, cell: Cell) -> Iterable[Cell]:
        return ()

    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        return ()

    def check(self) -> bool:
        return False

    def check_special(self) -> bool:
        return False

    def draw(self, context: DrawContext) -> None:
        pass

    def __str__(self) -> str:
        return self.name

    def __matmul__(self, square: Square) -> Cell:
        return self.grid.matrix[square]

    @staticmethod
    def neighbors_from_offsets(grid: Grid, cell: Cell, offsets: Iterable[Square]) -> Iterable[Cell]:
        row, column = cell.index
        for dr, dc in offsets:
            if 1 <= row + dr <= 9 and 1 <= column + dc <= 9:
                yield grid.matrix[row + dr, column + dc]

    __DESCRIPTORS = dict(N=(-1, 0), S=(1, 0), E=(0, 1), W=(0, -1), NE=(-1, 1), NW=(-1, -1), SE=(1, 1), SW=(1, -1))

    @staticmethod
    def parse_squares(descriptor: Union[str, int, Sequence[Square]]) -> Sequence[Square]:
        if isinstance(descriptor, int):
            descriptor = str(descriptor)
        if not isinstance(descriptor, str):
            return descriptor
        descriptors = Feature.__DESCRIPTORS
        pieces = deque(piece.strip() for piece in descriptor.split(','))

        squares = []
        while pieces:
            if pieces[0][0] in "123456789":
                value = int(pieces.popleft())
                if value <= 9:
                    row, column = value, int(pieces.popleft())
                else:
                    row, column = divmod(value, 10)
                squares.append((row, column))
            else:
                dr, dc = descriptors[pieces.popleft().upper()]
                row, column = squares[-1]
                squares.append((row + dr, column + dc))
        return squares

    @staticmethod
    def parse_square(descriptor: Union[str, int, Square]) -> Square:
        temp = Feature.parse_squares(descriptor)
        assert len(temp) == 1
        return temp[0]

    @staticmethod
    def box_for_square(square) -> tuple:
        row, column = square
        return ((row - 1) // 3, (column - 1) // 3)

    class_count: ClassVar[dict[Any, int]] = defaultdict(int)

    def get_default_feature_name(self):
        klass = self.__class__
        Feature.class_count[klass] += 1
        return f'{klass.__name__} #{Feature.class_count[klass]}'

    @staticmethod
    def draw_outside(context: DrawContext, value: Any, htype: House.Type, row_or_column: int, *,
                     is_right: bool = False, padding: float = 0, **args: Any):
        args = {'fontsize': 20, 'weight': 'bold', **args}

        if htype == House.Type.ROW:
            if not is_right:
                context.draw_text(.9 - padding, row_or_column + .5, str(value),
                                  verticalalignment='center', horizontalalignment='right', **args)
            else:
                context.draw_text(10.1 + padding, row_or_column + .5, str(value),
                                  verticalalignment='center', horizontalalignment='left', **args)
        else:
            if not is_right:
                context.draw_text(row_or_column + .5, .9 - padding, str(value),
                                  verticalalignment='bottom', horizontalalignment='center', **args)
            else:
                context.draw_text(row_or_column + .5, 10.1 + padding, str(value),
                                  verticalalignment='top', horizontalalignment='center', **args)

    @staticmethod
    def draw_outline(context, squares: Sequence[Square], *,
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
            context.plot(pts[:, 1], pts[:, 0], **args)

    @staticmethod
    def get_row_or_column(htype, row_column):
        if htype == House.Type.ROW:
            squares = [(row_column, i) for i in range(1, 10)]
        else:
            squares = [(i, row_column) for i in range(1, 10)]
        return squares

    check_elided: ClassVar[int] = 0
    check_called: ClassVar[int] = 0

    @staticmethod
    def check_only_if_changed(checker: CheckFunction) -> CheckFunction:
        saved_info: dict[Feature, Sequence[int]] = {}

        def called_function(self: Feature) -> bool:
            cells = cast(Sequence[Cell], getattr(self, 'cells'))
            if self in saved_info:
                generator = (-1 if cell.is_known else cell.bitmap for cell in cells)
                if all(x == y for x, y in zip(saved_info[self], generator)):
                    Feature.check_elided += 1
                    return False
            saved_info[self] = [-1 if cell.is_known else cell.bitmap for cell in cells]
            Feature.check_called += 1
            return checker(self)

        return called_function


@atexit.register
def print_counters():
    total = Feature.check_called + Feature.check_elided
    if total > 0:
        elision = 100.0 * Feature.check_elided / total
        print(f'Method feature.check() called {total} times; {Feature.check_elided} ({elision:.2f}%) were elided')


class MultiFeature(Feature):
    features: Sequence[Feature]

    def __init__(self, features: Sequence[Feature]):
        super().__init__()
        self.features = features

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        for feature in self.features:
            feature.initialize(grid)

    def reset(self) -> None:
        for feature in self.features:
            feature.reset()

    def get_neighbors(self, cell: Cell) -> Iterable[Cell]:
        for feature in self.features:
            yield from feature.get_neighbors(cell)

    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        for feature in self.features:
            yield from feature.get_neighbors_for_value(cell, value)

    def check(self) -> bool:
        return any(feature.check() for feature in self.features)

    def check_special(self) -> bool:
        return any(feature.check_special() for feature in self.features)

    def draw(self, context: DrawContext) -> None:
        for feature in self.features:
            feature.draw(context)
