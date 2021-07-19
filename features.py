from __future__ import annotations

import abc
import datetime
import functools
from collections import deque, defaultdict
from collections.abc import Iterable, Sequence, Mapping
from itertools import permutations, combinations, product, tee, groupby, combinations_with_replacement
from typing import Optional, ClassVar, Any, Union

from cell import Cell, House, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square, MultiFeature
from grid import Grid
from possibilities_feature import PossibilitiesFeature, GroupedPossibilitiesFeature


class KnightsMoveFeature(Feature):
    """No two squares within a knight's move of each other can have the same value."""
    OFFSETS = [(dr, dc) for dx in (-1, 1) for dy in (-2, 2) for (dr, dc) in ((dx, dy), (dy, dx))]

    def get_neighbors(self, cell: Cell) -> Iterable[Cell]:
        return self.neighbors_from_offsets(cell, self.OFFSETS)


class KingsMoveFeature(Feature):
    """No two pieces within a king's move of each other can have the same value."""
    OFFSETS = [(dr, dc) for dr in (-1, 1) for dc in (-1, 1)]

    def get_neighbors(self, cell: Cell) -> Iterable[Cell]:
        return self.neighbors_from_offsets(cell, self.OFFSETS)


class QueensMoveFeature(Feature):
    OFFSETS = [(dr, dc) for delta in range(1, 9) for dr in (-delta, delta) for dc in (-delta, delta)]
    values: set[int]

    def __init__(self, values: set[int] = frozenset({9})):
        super().__init__()
        self.values = values

    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        if value in self.values:
            return self.neighbors_from_offsets(cell, self.OFFSETS)
        else:
            return ()


class TaxicabFeature(Feature):
    """Two squares with the same value cannot have "value" as the taxicab distance between them."""
    taxis: set[int]

    def __init__(self, taxis: Sequence[int] = ()):
        super().__init__()
        self.taxis = set(taxis) if taxis else set(range(1, 10))

    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        if value in self.taxis:
            offsets = self.__get_offsets_for_value(value)
            return self.neighbors_from_offsets(cell, offsets)
        else:
            return ()

    @staticmethod
    @functools.lru_cache()
    def __get_offsets_for_value(value: int) -> Sequence[Square]:
        result = [square for i in range(0, value)
                  for square in [(i - value, i), (i, value - i), (value - i, -i), (-i, i - value)]]
        return result


class MagicSquareFeature(PossibilitiesFeature):
    """There is a magic square within the grid"""
    POSSIBILITIES = ((2, 7, 6, 9, 5, 1, 4, 3, 8), (2, 9, 4, 7, 5, 3, 6, 1, 8),
                     (8, 3, 4, 1, 5, 9, 6, 7, 2), (8, 1, 6, 3, 5, 7, 4, 9, 2),
                     (4, 3, 8, 9, 5, 1, 2, 7, 6), (6, 1, 8, 7, 5, 3, 2, 9, 4),
                     (6, 7, 2, 1, 5, 9, 8, 3, 4), (4, 9, 2, 3, 5, 7, 8, 1, 6),)

    center: Square
    color: str

    def __init__(self, center: Square = (5, 5), *, dr: int = 1, dc: int = 1, color: str = 'lightblue'):
        center_x, center_y = center
        squares = [(center_x + dr * dx, center_y + dc * dy) for dx, dy in product((-1, 0, 1), repeat=2)]
        super().__init__(squares, name=f'magic square at {center}')
        self.color = color
        self.center = center

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        return self.POSSIBILITIES

    def draw(self, context: DrawContext) -> None:
        context.draw_rectangles(self.squares, facecolor=self.color)


class AdjacentRelationshipFeature(Feature, abc.ABC):
    """
    Adjacent squares must fulfill some relationship.

    The squares have an order, so this relationship does not need to be symmetric.  (I.e. a thermometer)
    """
    squares: Sequence[Square]
    cells: Sequence[Cell]
    cyclic: bool
    handle_reset: bool

    triples: Sequence[tuple[Optional[Cell], Cell, Optional[Cell]]]
    color: Optional[str]

    def __init__(self, squares: Union[Sequence[Square], str], *,
                 name: Optional[str] = None, cyclic: bool = False, color: Optional[str] = 'gold'):
        super().__init__(name=name)
        self.squares = self.parse_squares(squares)
        self.cyclic = cyclic
        self.color = color

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        self.cells = [grid.matrix[x] for x in self.squares]
        self.triples = [
            ((self.cells[-1] if self.cyclic else None), self.cells[0], self.cells[1]),
            *[(self.cells[i - 1], self.cells[i], self.cells[i + 1]) for i in range(1, len(self.cells) - 1)],
            (self.cells[-2], self.cells[-1], (self.cells[0] if self.cyclic else None))]

    @abc.abstractmethod
    def match(self, digit1: int, digit2: int) -> bool: ...

    @Feature.check_only_if_changed
    def check(self) -> bool:
        for previous_cell, cell, next_cell in self.triples:
            if cell.is_known:
                continue
            impossible_values = {value for value in cell.possible_values
                                 if self.__is_impossible_value(value, previous_cell, cell, next_cell)}
            if impossible_values:
                print(f"{self}: No appropriate value in adjacent cells")
                Cell.remove_values_from_cells([cell], impossible_values)
                return True
        return False

    def __is_impossible_value(self, value: int,
                              previous_cell: Optional[Cell], cell: Cell, next_cell: Optional[Cell]) -> bool:
        previous_match = next_match = set(range(1, 10))
        if previous_cell:
            previous_match = {value2 for value2 in previous_cell.possible_values if self.match(value2, value)}
            if cell.is_neighbor(previous_cell):
                previous_match.discard(value)
        if next_cell:
            next_match = {value2 for value2 in next_cell.possible_values if self.match(value, value2)}
            if cell.is_neighbor(next_cell):
                next_match.discard(value)
        if not previous_match or not next_match:
            return True
        elif previous_cell and next_cell and previous_cell.is_neighbor(next_cell) \
                and len(previous_match) == 1 and len(next_match) == 1 and previous_match == next_match:
            return True
        return False

    def draw(self, context: DrawContext) -> None:
        if self.color:
            context.draw_line(self.squares, closed=self.cyclic, color=self.color, linewidth=5)


class AllValuesPresentFeature(Feature):
    """Verifies that within a set of squares, all values from 1 to 9 are present.  There should be nine or more
    squares.

    You should probably be using a SnakeFeature if there are exactly nine squares, as other more complicated logic
    is available if there is precisely one of each number.
    """
    squares: Sequence[Square]
    cells: Sequence[Cell]

    def __init__(self, squares: Sequence[Square]):
        super().__init__()
        assert len(squares) >= 9
        self.squares = squares

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        self.cells = [grid.matrix[x] for x in self.squares]

    def check(self) -> bool:
        known_cell_values = {cell.known_value for cell in self.cells if cell.is_known}
        unknown_cell_values = [value for value in range(1, 10) if value not in known_cell_values]
        unknown_cells = {cell for cell in self.cells if not cell.is_known}
        result = False
        for value in unknown_cell_values:
            cells = [cell for cell in unknown_cells if value in cell.possible_values]
            assert len(cells) >= 1
            if len(cells) == 1:
                cells[0].set_value_to(value)
                print(f'Hidden Single: Ring = {value} must be {cells[0]}')
                result = True
        return result


def _draw_thermometer(squares: Sequence[Square], color: str, context: DrawContext) -> None:
    context.draw_line(squares, color=color, linewidth=10)
    row, column = squares[0]
    context.draw_circle((column + .5, row + .5), radius=.3, fill=True, facecolor=color)


class Thermometer1Feature(AdjacentRelationshipFeature):
    """
    A sequence of squares that must monotonically increase.

    If slow is set, then this is a "slow" thermometer, and two adjacent numbers can be the same.  Typically,
    thermometers must be strictly monotonic.

    This implementation uses "adjacency"
    """
    def __init__(self, thermometer: Union[Sequence[Square], str], *,
                 name: Optional[str] = None, color: str = 'lightgrey') -> None:
        super().__init__(thermometer, name=name, color=color)

    def match(self, digit1: int, digit2: int) -> bool:
        return digit1 < digit2

    def draw(self, context: DrawContext) -> None:
        _draw_thermometer(self.squares, self.color, context)


class Thermometer2Feature(PossibilitiesFeature):
    """
    A sequence of squares that must monotonically increase.
    This is implemented as a subclass of Possibilities Feature.  Not sure which implementation is better.
    """
    color: str

    def __init__(self, thermometer: Union[Sequence[Square], str], *,
                 name: Optional[str] = None, color: str = 'lightgrey'):
        super().__init__(thermometer, name=name)
        self.color = color

    def draw(self, context: DrawContext) -> None:
        _draw_thermometer(self.squares, self.color, context)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        return combinations(range(1, 10), len(self.squares))


class Thermometer3Feature(GroupedPossibilitiesFeature):
    """
    A sequence of squares that must monotonically increase.
    This is implemented as a subclass of Possibilities Feature.  Not sure which implementation is better.
    """
    color: str

    def __init__(self, thermometer: Union[Sequence[Square], str],
                 name: Optional[str] = None, color: str = 'lightgrey'):
        super().__init__(thermometer, name=name)
        self.color = color

    def draw(self, context: DrawContext) -> None:
        _draw_thermometer(self.squares, self.color, context)

    def get_possibilities(self) -> Iterable[tuple[set[int], ...]]:
        length = len(self.squares)
        if length > 2:
            for permutation in combinations(range(2, 9), length - 2):
                yield (set(range(1, permutation[0])),
                       *({x} for x in permutation),
                       set(range(permutation[-1] + 1, 10)))
        else:
            for i in range(1, 9):
                yield {i}, set(range(i + 1, 10))


class ThermometerFeature(Thermometer3Feature):
    pass


class ThermometerAsLessThanFeature(ThermometerFeature):
    """A Thermometer of two squares, where we draw a < sign between them"""
    def __init__(self, thermometer: Union[Sequence[Square], str], *, name: Optional[str] = None) -> None:
        super().__init__(thermometer, name=name)
        assert len(self.squares) == 2

    def draw(self, context: DrawContext) -> None:
        assert len(self.squares) == 2
        (r1, c1), (r2, c2) = self.squares
        y, x = (r1 + r2) / 2, (c1 + c2) / 2
        dy, dx = (r2 - r1), (c2 - c1)
        context.draw_text(x + .5, y + .5, '>' if (dy == 1 or dx == -1) else '<',
                          verticalalignment='center', horizontalalignment='center',
                          rotation=(90 if dx == 0 else 0),
                          fontsize=20, weight='bold')


class SlowThermometerFeature(Thermometer1Feature):
    """A thermometer in which the digits only need to be ≤ rather than <"""
    def match(self, digit1: int, digit2: int) -> bool:
        return digit1 <= digit2


class BoxOfNineFeature(Feature):
    """A set of nine squares where each number is used exactly once."""
    show: bool
    line: bool
    color: Optional[str]

    squares: Sequence[Square]

    def __init__(self, squares: Union[str, Sequence[Square]], *,
                 show: bool = True, line: bool = True, color: Optional[str] = None):
        super().__init__()
        self.squares = self.parse_squares(squares)
        assert len(self.squares) == 9
        self.show = show
        self.line = line
        self.color = color

    @staticmethod
    def major_diagonal() -> BoxOfNineFeature:
        return BoxOfNineFeature([(i, i) for i in range(1, 10)])

    @staticmethod
    def minor_diagonal() -> BoxOfNineFeature:
        return BoxOfNineFeature([(10 - i, i) for i in range(1, 10)])

    @staticmethod
    def disjoint_groups() -> Sequence[BoxOfNineFeature]:
        return [BoxOfNineFeature(list((r + dr, c + dc) for dr in (0, 3, 6) for dc in (0, 3, 6)), show=False)
                for r in (1, 2, 3) for c in (1, 2, 3)]

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        cells = [grid.matrix[square] for square in self.squares]
        grid.houses.append(House(House.Type.EXTRA, 0, cells))

    def draw(self, context: DrawContext) -> None:
        if not self.show:
            return
        if self.line:
            context.draw_line(self.squares, color=(self.color or 'lightgrey'), linewidth=5)
        else:
            color = self.color or 'blue'
            for row, column in self.squares:
                context.draw_circle((column + .5, row + .5), radius=.1, fill=True, facecolor=color)


class LimitedValuesFeature(Feature):
    """A set of squares that initially contain only a limited set of values"""
    squares: Sequence[Square]
    values: Sequence[int]
    color: Optional[str]

    def __init__(self, squares: Union[Sequence[Square], str], values: Sequence[int], *,
                 color: Optional[str] = None):
        super().__init__()
        self.squares = self.parse_squares(squares)
        self.values = values
        self.color = color

    def reset(self) -> None:
        cells = [self @ x for x in self.squares]
        Cell.keep_values_for_cell(cells, set(self.values), show=False)

    def draw(self, context: DrawContext) -> None:
        if self.color:
            context.draw_rectangles(self.squares, color=self.color)


class OddsAndEvensFeature(MultiFeature):
    """Certain squares must be even; certain squares must be odd"""
    odds: LimitedValuesFeature
    evens: LimitedValuesFeature

    def __init__(self, odds: Union[Sequence[Square], str] = (), evens: Union[Sequence[Square], str] = ()):
        self.odds = LimitedValuesFeature(odds, (1, 3, 5, 7, 9))
        self.evens = LimitedValuesFeature(evens, (2, 4, 6, 8))
        super().__init__([self.odds, self.evens])

    def draw(self, context: DrawContext) -> None:
        for row, column in self.evens.squares:
            context.draw_rectangle((column + .1, row + .1), width=.8, height=.8, color='lightgray', fill=True)
        for row, column in self.odds.squares:
            context.draw_circle((column + .5, row + .5), radius=.4, color='lightgray', fill=True)


class AbstractMateFeature(Feature, abc.ABC):
    """Handles messages involving a square and its mates"""
    this_square: Square
    this_cell: Cell
    possible_mates: Sequence[Cell]
    done: bool

    def __init__(self, square: Square):
        super().__init__()
        self.this_square = square

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        self.this_cell = self @ self.this_square
        self.possible_mates = list(self.get_mates(self.this_cell))
        self.done = False

    def get_mates(self, cell: Cell) -> Iterable[Cell]:
        return self.neighbors_from_offsets(cell, KnightsMoveFeature.OFFSETS)

    def check(self) -> bool:
        if self.done:
            return False
        if self.this_cell.is_known:
            assert self.this_cell.known_value is not None
            return self._check_value_known(self.this_cell.known_value)
        else:
            return self._check_value_not_known()

    @abc.abstractmethod
    def _check_value_known(self, value: int) -> bool:
        """Handle the case of this cell having a known value"""
        ...

    @abc.abstractmethod
    def _check_value_not_known(self) -> bool:
        """Handle the case of this cell not having a value yet"""
        ...


class SameValueAsExactlyOneMateFeature(AbstractMateFeature):
    """The square must have the same value as exactly one of its mates"""
    def _check_value_known(self, value: int) -> bool:
        # We must make sure that the known value has exactly one mate
        count = sum(1 for cell in self.possible_mates if cell.is_known and cell.known_value == value)
        mates = [cell for cell in self.possible_mates if not cell.is_known and value in cell.possible_values]
        assert count < 2
        if count == 1:
            self.done = True
            if mates:
                print(f'Cell {self.this_cell} can only have one mate')
                Cell.remove_value_from_cells(mates, value)
                return True
            return False
        elif len(mates) == 1:
            print(f'Cell {self.this_cell} only has one possible mate')
            mates[0].set_value_to(value, show=True)
            self.done = True
            return True
        return False

    def _check_value_not_known(self) -> bool:
        # The only possible values for this cell are those values for which it can have one mate.
        impossible_values = set()
        for value in self.this_cell.possible_values:
            count = sum(1 for cell in self.possible_mates if cell.is_known and cell.known_value == value)
            mates = [cell for cell in self.possible_mates if not cell.is_known and value in cell.possible_values]
            if count >= 2 or (count == 0 and not mates):
                impossible_values.add(value)
        if impossible_values:
            print(f'Cell {self.this_cell} must have a mate value')
            Cell.remove_values_from_cells([self.this_cell], impossible_values)
            return True
        return False


class SameValueAsMateFeature(AbstractMateFeature):
    """The square must have the same value as at least one of its mates"""
    def _check_value_known(self, value: int) -> bool:
        if any(cell.is_known and cell.known_value == value for cell in self.possible_mates):
            # We didn't change anything, but we've verified that this guy has a mate
            self.done = True
            return False
        mates = [cell for cell in self.possible_mates if not cell.is_known and value in cell.possible_values]
        assert len(mates) >= 1
        if len(mates) == 1:
            print(f'Cell {self.this_cell} has only one possible mate')
            mates[0].\
                set_value_to(value, show=True)
            self.done = True
            return True
        return False

    def _check_value_not_known(self) -> bool:
        legal_values = SmallIntSet.union(*(cell.possible_values for cell in self.possible_mates))
        if not self.this_cell.possible_values <= legal_values:
            print(f'Cell {self.this_cell} must have a mate')
            Cell.keep_values_for_cell([self.this_cell], legal_values)
            return True
        return False


class LittlePrincessFeature(Feature):
    """The taxicab distance between two like values cannot be that value"""
    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        offsets = self.__get_offsets_for_value(value)
        return self.neighbors_from_offsets(cell, offsets)

    @staticmethod
    @functools.lru_cache
    def __get_offsets_for_value(value: int) -> Sequence[Square]:
        return [(dr, dc) for delta in range(1, value)
                for dr in (-delta, delta) for dc in (-delta, delta)]


class AlternativeBoxesFeature(Feature):
    """Don't use the regular nine boxes, but instead use 9 alternative boxes"""
    squares: Sequence[Sequence[Square]]

    def __init__(self, pattern: Union[str, Sequence[str]]) -> None:
        super().__init__()
        if isinstance(pattern, str):
            # We can use a string of 81 characters, each one 1-9, identifying the box it belongs to
            assert len(pattern) == 81
            info: Sequence[list[Square]] = [list() for _ in range(10)]
            for (row, column), letter in zip(self.all_squares(), pattern):
                assert '1' <= letter <= '9'
                info[int(letter)].append((row, column))
            for i in range(1, 10):
                assert len(info[i]) == 9
            self.squares = info[1:]
        else:
            # Alternatively, we pass along a list of 9 sequences of squares
            assert len(pattern) == 9
            self.squares = [self.parse_squares(item) for item in pattern]
            for box in self.squares:
                assert len(box) == 9
            all_squares = [square for box in self.squares for square in box]
            assert len(set(all_squares)) == 81

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        grid.delete_normal_boxes()
        boxes = [House(House.Type.BOX, i + 1,
                       [grid.matrix[square] for square in self.squares[i]])
                 for i in range(len(self.squares))]
        grid.houses.extend(boxes)

    def draw(self, context: DrawContext) -> None:
        colors = ('lightcoral', "violet", "bisque", "lightgreen", "lightgray", "yellow", "skyblue",
                  "pink", "purple")
        for square, color in zip(self.squares, colors):
            self.draw_outline(context, square, inset=.1, color='black')


class SandwichFeature(GroupedPossibilitiesFeature):
    """Specify the total sum of the squares between the 1 and the 9."""
    htype: House.Type
    row_column: int
    total: int

    @staticmethod
    def all(htype: House.Type, totals: Sequence[Optional[int]]) -> Sequence[SandwichFeature]:
        """Used to set sandwiches for an entire row or column.   A none indicates missing"""
        return [SandwichFeature(htype, rc, total) for rc, total in enumerate(totals, start=1) if total is not None]

    def __init__(self, htype: House.Type, row_column: int, total: int):
        name = f'Sandwich {htype.name.title()} #{row_column}'
        squares = self.get_house_squares(htype, row_column)
        self.htype = htype
        self.row_column = row_column
        self.total = total
        super().__init__(squares, name=name, compressed=True)

    def get_possibilities(self) -> Iterable[tuple[set[int], ...]]:
        return self._get_possibilities(self.total)

    @classmethod
    def _get_possibilities(cls, total: int) -> Iterable[tuple[set[int], ...]]:
        for length in range(0, 8):
            for values in combinations((2, 3, 4, 5, 6, 7, 8), length):
                if sum(values) == total:
                    non_values = set(range(2, 9)) - set(values)
                    non_values_length = 7 - length
                    temp = deque([{1, 9}, *([set(values)] * length), {1, 9}, *([non_values] * non_values_length)])
                    for i in range(0, non_values_length + 1):
                        yield tuple(temp)
                        temp.rotate(1)

    ONE_AND_NINE = SmallIntSet((1, 9))

    def draw(self, context: DrawContext) -> None:
        self.draw_outside(context, self.total, self.htype, self.row_column, fontsize=20, weight='bold')
        if not context.get(self.__class__):
            context[self.__class__] = True
            special = [cell.index for cell in self.grid.cells if cell.possible_values.isdisjoint(self.ONE_AND_NINE)]
            context.draw_rectangles(special, color='lightgreen')


class SandwichXboxFeature(PossibilitiesFeature):
    htype: House.Type
    row_column: int
    value: int
    is_right: bool

    def __init__(self, htype: House.Type, row_column: int, value: int, right: bool = False) -> None:
        name = f'Skyscraper {htype.name.title()} #{row_column}'
        squares = self.get_house_squares(htype, row_column)
        self.htype = htype
        self.row_column = row_column
        self.value = value
        self.is_right = right
        super().__init__(squares, name=name)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        result = self._get_all_possibilities()[self.value]
        if not self.is_right:
            return result
        else:
            return (item[::-1] for item in result)

    @staticmethod
    @functools.lru_cache(None)
    def _get_all_possibilities() -> Mapping[int, Sequence[tuple[int, ...]]]:
        result: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        start = datetime.datetime.now()
        for values in permutations(range(1, 10)):
            index1 = values.index(1)
            index2 = values.index(9)
            if index2 < index1:
                index2, index1 = index1, index2
            sandwich = sum([values[index] for index in range(index1 + 1, index2)])
            xbox = sum([values[index] for index in range(values[0])])
            if sandwich == xbox:
                result[sandwich].append(values)
        end = datetime.datetime.now()
        print(f'Initialization = {end - start}.')
        return result

    def draw(self, context: DrawContext) -> None:
        args = dict(fontsize=20, weight='bold')
        self.draw_outside(context, self.value, self.htype, self.row_column, is_right=self.is_right, **args)


class SameValueFeature(PossibilitiesFeature):
    def __init__(self, *squares: Square, name: Optional[str] = None) -> None:
        assert len(squares) > 1
        super().__init__(squares, name=name)

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        mapper: dict[Cell, frozenset[Cell]] = getattr(grid, "SameValueFeature", None)
        if not mapper:
            mapper = {cell: frozenset([cell]) for cell in grid.cells}
            setattr(grid, "SameValueFeature", mapper)
        clone_group = frozenset.union(*(mapper[cell] for cell in self.cells))
        for cell in clone_group:
            mapper[cell] = clone_group

    def reset(self) -> None:
        super().reset()
        mapper: dict[Cell, frozenset[Cell]] = getattr(self.grid, "SameValueFeature", None)
        if not mapper:
            return
        try:
            clone_groups = set(group for group in mapper.values() if len(group) > 1)
            for clone_group in clone_groups:
                neighbors = frozenset.union(*(cell.neighbors for cell in clone_group))
                for cell in clone_group:
                    assert cell not in neighbors
                    cell.neighbors = neighbors
        finally:
            delattr(self.grid, "SameValueFeature")

    def get_possibilities(self) -> list[tuple[int, ...]]:
        count = len(self.squares)
        return [(i,) * count for i in range(1, 10)]


class PalindromeFeature(MultiFeature):
    squares: Sequence[Square]
    color: Optional[str]

    def __init__(self, squares: Union[Sequence[Square], str], color: Optional[str] = None):
        self.squares = squares = self.parse_squares(squares)
        self.color = color
        count = len(squares) // 2
        features = [SameValueFeature((r1, c1), (r2, c2), name=f'r{r1}c{c1}=r{r2}c{c2}')
                    for (r1, c1), (r2, c2) in zip(squares[:count], squares[::-1])]
        super().__init__(features)

    def draw(self, context: DrawContext) -> None:
        context.draw_line(self.squares, color=self.color or 'blue')


class CloneBoxFeature(MultiFeature):
    def __init__(self, index1: int, index2: int):
        squares1 = self.get_house_squares(House.Type.BOX, index1)
        squares2 = self.get_house_squares(House.Type.BOX, index2)
        features = [SameValueFeature(*pair) for pair in zip(squares1, squares2)]
        super().__init__(features)


class XVFeature(AdjacentRelationshipFeature):
    total: Optional[int]
    non_total: Optional[set[int]]

    def __init__(self, squares: tuple[Square, Square], *,
                 total: Optional[int] = None, non_total: Optional[set[int]] = None):
        super().__init__(squares, name=f'{squares[0]}/{squares[1]}')
        self.total = total
        self.non_total = non_total

    def match(self, digit1: int, digit2: int) -> bool:
        if self.total:
            return digit1 + digit2 == self.total
        else:
            return digit1 + digit2 not in self.non_total

    def draw(self, context: DrawContext) -> None:
        if self.total:
            args = {}
            (r1, c1), (r2, c2) = self.squares
            character = 'X' if self.total == 10 else 'XV' if self.total == 15 else 'V'
            context.draw_text((c1 + c2 + 1) / 2, (r1 + r2 + 1) / 2, character,
                              verticalalignment='center', horizontalalignment='center', **args)

    @classmethod
    def setup(cls, *,
              across: Mapping[int, Union[Sequence[Square], str]],
              down: Mapping[int, Union[Sequence[Square], str]],
              all_listed: bool = True,
              all_values: Optional[set[int]] = None) -> MultiFeature:
        across = {total: cls.parse_squares(squares) for total, squares in across.items()}
        down = {total: cls.parse_squares(squares) for total, squares in down.items()}
        features = []
        features.extend(XVFeature(((row, column), (row, column + 1)), total=total)
                        for total, squares in across.items()
                        for row, column in squares)
        features.extend(XVFeature(((row, column), (row + 1, column)), total=total)
                        for total, squares in down.items()
                        for row, column in squares)
        if all_listed:
            if all_values is None:
                all_values = set(across.keys()).union(set(down.keys()))
            acrosses_seen = {square for total, squares in across.items() for square in squares}
            downs_seen = {square for total, squares in down.items() for square in squares}
            features.extend(XVFeature(((row, column), (row, column + 1)), non_total=all_values)
                            for row, column in product(range(1, 10), range(1, 9))
                            if (row, column) not in acrosses_seen)
            features.extend(XVFeature(((row, column), (row + 1, column)), non_total=all_values)
                            for row, column in product(range(1, 9), range(1, 10))
                            if (row, column) not in downs_seen)
        return MultiFeature(features)

    def __str__(self) -> str:
        if self.total:
            return f'<{self.squares[0]}+{self.squares[1]}={self.total}>'
        else:
            return f'<{self.squares[0]}+{self.squares[1]}!={self.non_total}>'


class KropkeDotFeature(MultiFeature):
    squares: Sequence[Square]
    is_black: bool

    def __init__(self, squares: Union[str, tuple[Square]], *, color: str) -> None:
        assert color == 'white' or color == 'black'
        self.is_black = color == 'black'
        self.squares = squares = self.parse_squares(squares)

        chunks = [tuple(items) for _, items in groupby(squares, Feature.box_for_square)]
        prev = None
        features = []
        for chunk in chunks:
            if prev:
                features.append(self._KropkeInBoxFeature((prev, chunk[0]), black=self.is_black))
            if len(chunk) > 1:
                features.append(self._KropkeInBoxFeature(chunk, black=self.is_black))
            prev = chunk[-1]
        super().__init__(features)

    def draw(self, context: DrawContext) -> None:
        (iter1, iter2) = tee(self.squares)
        next(iter2, None)
        for (y1, x1), (y2, x2) in zip(iter1, iter2):
            context.draw_circle(((x1 + x2 + 1) / 2, (y1 + y2 + 1) / 2),
                                radius=.2, fill=self.is_black, color='black')

    class _KropkeInBoxFeature(PossibilitiesFeature):
        is_black: bool

        def __init__(self, squares: [tuple[Square]], *, black) -> None:
            (r0, c0), (r1, c1) = squares[0], squares[-1]
            super().__init__(squares, name=f'Kropke r{r0}c{c0}-r{r1}c{c1}')
            self.is_black = black

        def get_possibilities(self) -> list[tuple[int, ...]]:
            count = len(self.squares)
            assert count >= 2
            if not self.is_black or count >= 3:
                items = (1, 2, 4, 8) if self.is_black else list(range(1, 10))
                for i in range(len(items) - count + 1):
                    yield items[i:i + count]
                    yield items[i:i + count][::-1]
            else:
                for i in (1, 2, 3, 4):
                    yield i, 2 * i
                    yield 2 * i, i


class NonConsecutiveFeature(MultiFeature):
    def __init__(self) -> None:
        super().__init__([self._Comparer(self.get_house_squares(htype, i), name=f'{htype.name} #{i}')
                          for htype in [House.Type.ROW, House.Type.COLUMN]
                          for i in range(1, 10)])

    def draw(self, context: DrawContext) -> None:
        pass

    class _Comparer(AdjacentRelationshipFeature):
        def __init__(self, squares: Sequence[Square], *, name: str):
            # color = None so we don't need to override draw
            super().__init__(squares, name=name, color=None)

        def match(self, digit1: int, digit2: int) -> bool:
            return abs(digit1 - digit2) != 1


class KillerCageFeature(PossibilitiesFeature):
    """The values in the cage must all be different.  They must sum to the total"""
    total: int

    def __init__(self, total: int, squares: Union[Sequence[Square], str]):
        self.total = total
        super().__init__(squares)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        count = len(self.squares)
        for values in permutations(range(1, 10), count - 1):
            last_value = self.total - sum(values)
            if 1 <= last_value <= 9 and last_value not in values:
                yield *values, last_value

    def draw(self, context: DrawContext) -> None:
        self.draw_outline(context, self.squares)
        row, column = min(self.squares)
        context.draw_text(column + .2, row + .2, str(self.total),
                          verticalalignment='top', horizontalalignment='left', fontsize=10, weight='bold')


class ArrowFeature(PossibilitiesFeature):
    """The sum of the values in the arrow must equal the digit in the head of the array"""
    def __init__(self, squares: Union[Sequence[Square], str]):
        super().__init__(squares, neighbors=True)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        return self._get_possibilities(len(self.squares) - 1)

    @staticmethod
    @functools.cache
    def _get_possibilities(count):
        return [(total, *values) for values in product(range(1, 10), repeat=count)
                for total in [sum(values)]
                if total < 10]

    def draw(self, context: DrawContext) -> None:
        y, x = self.squares[0]
        context.draw_circle((x + .5, y+.5), radius=.5, fill=False, color='black')
        context.draw_line(self.squares)


class BetweenLineFeature(PossibilitiesFeature):
    """The values in the middle of the arrow must be strictly in between the values of the two endpoints"""
    def __init__(self, squares: Union[Sequence[Square], str]):
        super().__init__(squares, neighbors=True)

    @staticmethod
    def between(square1: Square, square2: Square) -> BetweenLineFeature:
        (r1, c1), (r2, c2) = square1, square2
        dr, dc = r2 - r1, c2 - c1
        distance = max(abs(dr), abs(dc))
        dr, dc = dr // distance, dc // distance
        squares = [square1]
        while squares[-1] != square2:
            r1, c1 = r1 + dr, c1 + dc
            squares.append((r1, c1))
        return BetweenLineFeature(squares)

    def get_possibilities(self) -> Iterable[tuple[set[int], ...]]:
        for low in range(1, 8):
            for high in range(low + 2, 10):
                for middle in product(range(low + 1, high), repeat=len(self.squares) - 2):
                    yield low, *middle, high
                    yield high, *middle, low

    def draw(self, context: DrawContext) -> None:
        y0, x0 = self.squares[0]
        y1, x1 = self.squares[-1]
        context.draw_circle((x0 + .5, y0+.5), radius=.4, fill=True, color='lightgray')
        context.draw_circle((x1 + .5, y1+.5), radius=.4, fill=True, color='lightgray')
        context.draw_line(self.squares)


class ExtremesFeature(MultiFeature):
    """Reds must be larger than all of its neighbors.  Greens must be smaller than all of its neighbors"""
    reds: Sequence[Square]
    greens: Sequence[Square]

    def __init__(self, *, reds: Union[str, Sequence[Square]] = (), greens: Union[str, Sequence[Square]] = ()) -> None:
        self.reds = self.parse_squares(reds)
        self.greens = self.parse_squares(greens)
        super().__init__([
            *[self._Comparer(square, high=True) for square in self.reds],
            *[self._Comparer(square, high=False) for square in self.greens],
        ])

    def draw(self, context: DrawContext) -> None:
        for color, squares in (('#FCA0A0', self.reds), ('#B0FEB0', self.greens)):
            for y, x in squares:
                context.draw_rectangle((x, y), width=1, height=1, color=color, fill=True)

    class _Comparer(PossibilitiesFeature):
        high: bool

        def __init__(self, square: Square, high: bool):
            squares = [square, *self._orthogonal_neighbors(square)]
            self.high = high
            super().__init__(squares, neighbors=True)

        def get_possibilities(self) -> list[tuple[int, ...]]:
            count = len(self.squares) - 1
            for center in range(1, 10):
                outside_range = range(1, center) if self.high else range(center + 1, 10)
                for outside in product(outside_range, repeat=count):
                    yield center, *outside

        @staticmethod
        def _orthogonal_neighbors(square):
            row, column = square
            return [(r, c) for r, c in ((row + 1, column), (row - 1, column), (row, column + 1), (row, column - 1))
                    if 1 <= r <= 9 and 1 <= c <= 9]

        def __str__(self) -> str:
            r, c = self.squares[0]
            return f'<r{r}c{c} {"high" if self.high else "low"}>'


class LittleKillerFeature(PossibilitiesFeature):
    """Typically done via a diagonal.  The sum of the diagonal must total a specific value"""
    ranges: ClassVar[Sequence[range]] = (None, range(1, 9 + 1), range(3, 17 + 1), range(6, 24 + 1))
    ranges_dict: ClassVar[Any] = None
    total: int
    start: Square
    direction: Square

    def __init__(self, total: int, start: Square, direction: Square):
        self.total = total
        self.start = start
        self.direction = direction
        row, column = start
        dr, dc = direction
        squares = []
        while 1 <= row <= 9 and 1 <= column <= 9:
            squares.append((row, column))
            row, column = row + dr, column + dc
        if not self.ranges_dict:
            ranges_dict = defaultdict(list)
            for count in (1, 2, 3):
                for values in permutations(range(1, 10), count):
                    ranges_dict[count, sum(values)].append(values)
            self.ranges_dict = ranges_dict

        super().__init__(squares)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        # Find the clumps, where clumps are the squares that are in a single box
        clumps = [list(squares)
                  for box, squares in groupby(self.squares, key=self.box_for_square)]
        length = len(clumps)
        # Possible values that each clump can have
        values = [self.ranges[len(clump)] for clump in clumps]
        left_values, right_values = values[0:length // 2], values[length // 2:]

        right_map = defaultdict(list)
        for right_value_list in product(*right_values):
            right_map[sum(right_value_list)].append(right_value_list)

        for left_value_list in product(*left_values):
            for right_value_list in right_map[self.total - sum(left_value_list)]:
                value_list = [*left_value_list, *right_value_list]
                value_items = [self.ranges_dict[len(clump), value] for clump, value in zip(clumps, value_list)]
                for list_of_list in product(*value_items):
                    yield tuple(x for lst in list_of_list for x in lst)

    def draw(self, context: DrawContext) -> None:
        (y, x), (dy, dx) = self.start, self.direction
        context.draw_text(x - dx + .5, y - dy + .5, str(self.total),
                          verticalalignment='center', horizontalalignment='center',
                          fontsize=25, color='black', weight='light')
        context.arrow(x - dx + .5, y - dy + .5, .5 * dx, .5 * dy,
                      length_includes_head=True,
                      head_width=.2, head_length=.2)


class QuadrupleFeature(PossibilitiesFeature):
    values: Sequence[int]

    def __init__(self, *, top_left: Square, values: Sequence[int]):
        row, column = top_left
        squares = [(row, column), (row, column + 1), (row + 1, column + 1), (row + 1, column)]
        self.values = values
        super().__init__(squares, neighbors=True, duplicates=True)

    def get_possibilities(self) -> list[tuple[int, ...]]:
        for extra in combinations_with_replacement(range(1, 10), 4 - len(self.values)):
            sequence = (*self.values, *extra)
            yield from permutations(sequence)

    def draw(self, context: DrawContext) -> None:
        y, x = self.squares[0]
        context.draw_circle((x + 1, y + 1), radius=.2, fill=False)
        text = ' '.join(str(x) for x in self.values)
        if len(self.values) >= 3:
            text = text[0:3] + '\n' + text[4:]
        context.draw_text(x + 1, y + 1, text,
                          fontsize=10,
                          verticalalignment='center', horizontalalignment='center', color='black')

    def __str__(self) -> str:
        row, column = self.squares[0]
        return f'Quad {"".join(str(value) for value in sorted(self.values))}@r{row}c{column}'


class RenbanFeature(PossibilitiesFeature):
    color: Optional[str]

    def __init__(self, squares: Union[Sequence[Square], str], color: Optional[str] = None):
        super().__init__(squares)
        self.color = color

    def get_possibilities(self) -> list[tuple[set[int], ...]]:
        count = len(self.squares)
        for i in range(1, 11 - count):
            yield from permutations(range(i, i + count))

    def draw(self, context: DrawContext) -> None:
        context.draw_line(self.squares, color=self.color or 'lightgrey', linewidth=5)


class MessageFeature(MultiFeature):
    mapping: Mapping[str, Sequence[Square]]

    def __init__(self, letters: str, squares: Union[Sequence[Square], str]):
        mapping: dict[str, list[Square]] = defaultdict(list)
        for letter, square in zip(letters, self.parse_squares(squares)):
            mapping[letter].append(square)
        assert len(mapping) == 9   # We'll fix this later, maybe
        self.mapping = mapping

        features = [
            BoxOfNineFeature(squares=[squares[0] for squares in mapping.values()], show=False),
            *[SameValueFeature(*squares, name=f'Letter "{letter}"')
              for letter, squares in mapping.items() if len(squares) > 1],
        ]
        super().__init__(features)

    def draw(self, context: DrawContext) -> None:
        for letter, squares in self.mapping.items():
            for y, x in squares:
                context.draw_text(x, y, letter,
                                  weight='bold',
                                  fontsize=10,
                                  verticalalignment='top', horizontalalignment='left')


class HelperFeature(Feature):
    round: int

    def __init__(self, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.round = 0

    def check_special(self) -> bool:
        self.round += 1
        method = getattr(self, f'round_{self.round}', None)
        if method:
            print(f'Looking at {self} for Round=#{self.round}')
            method()
            return True
        else:
            return False
