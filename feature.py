from __future__ import annotations

import abc
import atexit
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from itertools import product, zip_longest
from typing import ClassVar, Optional, Union, cast

from cell import Cell, CellValue, House
from draw_context import DrawContext
from grid import Grid

Square = tuple[int, int]
SquaresParseable = Union[str, int, Sequence[Square]]
CheckFunction = Callable[['Feature'], bool]


class Feature(abc.ABC):
    name: str
    grid: Grid

    __prefix_count: ClassVar[dict[str, int]] = defaultdict(int)

    def __init__(self, *, name: Optional[str] = None, prefix: Optional[str] = None) -> None:
        if not name:
            prefix = prefix or self.__class__.__name__.removesuffix("Feature")
            self.__prefix_count[prefix] += 1
            name = f'{prefix} #{self.__prefix_count[prefix]}'
        self.name = name

    def initialize(self, grid) -> None:
        self.grid = grid

    def start(self) -> None:
        pass

    def get_neighbors(self, cell: Cell) -> Iterable[Cell]:
        return ()

    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        return ()

    def check(self) -> bool:
        return False

    def check_special(self) -> bool:
        return False

    # noinspection PyMethodMayBeStatic
    def get_strong_pairs(self, _cell_value: CellValue) -> Iterable[CellValue]:
        """
        cell1=value1 and cell2=value2 are a strong pair if at least one of them is always True.
        For now, we only generate results in which exactly one of them is always True."""
        """"""
        return ()

    def get_weak_pairs(self, _cell_value: CellValue) -> Iterable[CellValue]:
        """
        cell1=value1 and cell2=value2 are a weak pair if both of them cannot be simultaneously True.
        If either of them is true, then the other must be False.
        """
        return ()

    # noinspection PyMethodMayBeStatic
    def get_chain_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        """
        Exactly one of cell1=value and cell2=value is true.  These are both strong and weak.
        """
        # In the lack of better information, we can just intersection strong_pairs and weak_pairs.  Most features
        # don't even bother having strong pairs, so find that first.
        strong_pairs = set(self.get_strong_pairs(cell_value))
        if strong_pairs:
            yield from (w for w in self.get_weak_pairs(cell_value) if w in strong_pairs)

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
    def parse_squares(descriptor: SquaresParseable) -> Sequence[Square]:
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
                piece = pieces.popleft().upper()
                if piece == 'R':
                    squares = [(col, 10 - row) for row, col in squares]
                elif piece == 'T':
                    squares = [(col, row) for row, col in squares]
                else:
                    dr, dc = descriptors[piece]
                    row, column = squares[-1]
                    assert 1 <= row + dr <= 9 and 1 <= column + dc <= 9
                    squares.append((row + dr, column + dc))
        return squares

    @staticmethod
    def parse_square(descriptor: SquaresParseable) -> Square:
        temp = Feature.parse_squares(descriptor)
        assert len(temp) == 1
        return temp[0]

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
    def has_neighbor_method(cls):
        return cls.get_neighbors != Feature.get_neighbors or \
            cls.get_neighbors_for_value != Feature.get_neighbors_for_value

    @classmethod
    def has_check_method(cls):
        return cls.check != Feature.check or cls.check_special != Feature.check_special

    @classmethod
    def has_any_pair_method(cls):
        return cls.get_weak_pairs != Feature.get_weak_pairs or cls.get_strong_pairs != Feature.get_strong_pairs or \
            cls.get_chain_pairs != Feature.get_chain_pairs

    check_elided: ClassVar[int] = 0
    check_called: ClassVar[int] = 0

    @staticmethod
    def cells_changed_since_last_invocation(location: list[int], cells: Sequence[Cell]) -> bool:
        generator = (-1 if cell.is_known else cell.bitmap for cell in cells)
        if all(x == y for x, y in zip_longest(location, generator)):
            Feature.check_elided += 1
            return False
        else:
            Feature.check_called += 1
            location[:] = (-1 if cell.is_known else cell.bitmap for cell in cells)
            return True


@atexit.register
def print_counters():
    total = Feature.check_called + Feature.check_elided
    if total > 0:
        elision = 100.0 * Feature.check_elided / total
        print(f'Method feature.check() called {total} times; {Feature.check_elided} ({elision:.2f}%) were elided')


if __name__ == '__main__':
    pass
