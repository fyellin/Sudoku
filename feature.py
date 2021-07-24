from __future__ import annotations

import abc
import atexit
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence, Callable
from itertools import product
from typing import Any, ClassVar, Union, cast, Optional

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

    def __repr__(self) -> str:
        return self.name

    def __matmul__(self, square: Square) -> Cell:
        return self.grid.matrix[square]

    def neighbors_from_offsets(self, cell: Cell, offsets: Iterable[Square]) -> Iterable[Cell]:
        row, column = cell.index
        for dr, dc in offsets:
            if 1 <= row + dr <= 9 and 1 <= column + dc <= 9:
                yield self.grid.matrix[row + dr, column + dc]

    __DESCRIPTORS = dict(N=(-1, 0), S=(1, 0), E=(0, 1), W=(0, -1), NE=(-1, 1), NW=(-1, -1), SE=(1, 1), SW=(1, -1))

    @staticmethod
    def parse_squares(descriptor: Union[str, int, Sequence[Square]]) -> Sequence[Square]:
        if isinstance(descriptor, int):
            descriptor = str(descriptor)
        if not isinstance(descriptor, str):
            return descriptor
        descriptors = Feature.__DESCRIPTORS
        pieces = deque(piece.strip() for piece in descriptor.split(','))

        squares: list[Square] = []
        while pieces:
            if pieces[0][0] in "123456789":
                value = int(pieces.popleft())
                if value <= 9:
                    row, column = value, int(pieces.popleft())
                else:
                    row, column = divmod(value, 10)
                assert 1 <= row <= 9 and 1 <= column <= 9
                squares.append((row, column))
            else:
                dr, dc = descriptors[pieces.popleft().upper()]
                row, column = squares[-1]
                assert 1 <= row + dr <= 9 and 1 <= column + dc <= 9
                squares.append((row + dr, column + dc))
        return squares

    @staticmethod
    def parse_square(descriptor: Union[str, int, Square]) -> Square:
        temp = Feature.parse_squares(descriptor)
        assert len(temp) == 1
        return temp[0]

    class_count: ClassVar[dict[Any, int]] = defaultdict(int)

    def get_default_feature_name(self):
        klass = self.__class__
        Feature.class_count[klass] += 1
        return f'{klass.__name__} #{Feature.class_count[klass]}'

    @staticmethod
    def get_house_squares(htype, index):
        if htype == House.Type.ROW:
            return [(index, i) for i in range(1, 10)]
        if htype == House.Type.COLUMN:
            return [(i, index) for i in range(1, 10)]
        if htype == House.Type.BOX:
            q, r = divmod(index - 1, 3)
            start_row = 3 * q + 1
            start_column = 3 * r + 1
            return [(row, column)
                    for row in range(start_row, start_row + 3)
                    for column in range(start_column, start_column + 3)]
        assert False, f'Bad argument {htype}'

    @staticmethod
    def box_for_square(square) -> int:
        row, column = square
        return 3 * ((row - 1) // 3) + ((column - 1) // 3) + 1

    @staticmethod
    def all_squares() -> Iterable[Square]:
        return cast(Iterable[Square], product(range(1, 10), repeat=2))

    @classmethod
    def is_neighborly(cls):
        return (cls.get_neighbors, cls.get_neighbors_for_value) != \
               (Feature.get_neighbors, Feature.get_neighbors_for_value)

    @classmethod
    def is_checking(cls):
        return (cls.check, cls.check_special) != (Feature.check, Feature.check_special)

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


if __name__ == '__main__':
    pass
