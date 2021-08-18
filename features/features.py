from __future__ import annotations

import functools
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations_with_replacement, groupby, permutations, product
from typing import Any, ClassVar, Optional

from cell import Cell, House, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from grid import Grid
from tools.itertool_recipes import pairwise
from .possibilities_feature import PossibilitiesFeature, AdjacentRelationshipFeature, FullGridAdjacencyFeature


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

    def start(self) -> None:
        self.cells = [self @ x for x in self.squares]

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


class BoxOfNineFeature(Feature):
    """A set of nine squares where each number is used exactly once."""
    show: bool
    line: bool
    color: Optional[str]

    squares: Sequence[Square]

    def __init__(self, squares: SquaresParseable, *,
                 name: Optional[str] = None, prefix: Optional[str] = None,
                 show: bool = True, line: bool = True, color: Optional[str] = None):
        super().__init__(name=name, prefix=prefix)
        self.squares = self.parse_squares(squares)
        assert len(self.squares) == 9
        self.show = show
        self.line = line
        self.color = color

    @staticmethod
    def major_diagonal(**kwargs: Any) -> BoxOfNineFeature:
        return BoxOfNineFeature([(i, i) for i in range(1, 10)], name="Major Diagonal", **kwargs)

    @staticmethod
    def minor_diagonal(**kwargs: Any) -> BoxOfNineFeature:
        return BoxOfNineFeature([(10 - i, i) for i in range(1, 10)], name="Minor Diagonal", **kwargs)

    @staticmethod
    def disjoint_groups() -> Sequence[BoxOfNineFeature]:
        return [BoxOfNineFeature(list((r + dr, c + dc) for dr in (0, 3, 6) for dc in (0, 3, 6)), show=False)
                for r in (1, 2, 3) for c in (1, 2, 3)]

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        cells = [self @ square for square in self.squares]
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
    values: SmallIntSet
    color: Optional[str]

    def __init__(self, squares: SquaresParseable, values: Sequence[int], *,
                 name: Optional[str] = None, prefix: Optional[str] = None,
                 color: Optional[str] = None):
        super().__init__(name=name, prefix=prefix)
        self.squares = self.parse_squares(squares)
        self.values = SmallIntSet(values)
        self.color = color

    @classmethod
    def odds_and_evens(cls, *, odds: SquaresParseable = (), evens: SquaresParseable = ()) -> Sequence[Feature]:
        odds = cls.parse_squares(odds)
        evens = cls.parse_squares(evens)

        def draw_function(context) -> None:
            for row, column in evens:
                context.draw_rectangle((column + .1, row + .1), width=.8, height=.8, color='lightgray', fill=True)
            for row, column in odds:
                context.draw_circle((column + .5, row + .5), radius=.4, color='lightgray', fill=True)

        return [
            *([LimitedValuesFeature(odds, (1, 3, 5, 7, 9), name="Odds")] if odds else []),
            *([LimitedValuesFeature(evens, (2, 4, 6, 8), name="Evens")] if evens else []),
            DrawOnlyFeature(draw_function),
        ]

    def start(self) -> None:
        cells = [self @ x for x in self.squares]
        Cell.keep_values_for_cell(cells, self.values, show=False)

    def draw(self, context: DrawContext) -> None:
        if self.color:
            context.draw_rectangles(self.squares, color=self.color)


class AlternativeBoxesFeature(Feature):
    """Don't use the regular nine boxes, but instead use 9 alternative boxes"""
    house_squares: Sequence[Sequence[Square]]

    def __init__(self, pattern: str | Sequence[SquaresParseable]) -> None:
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
            self.house_squares = info[1:]
        else:
            # Alternatively, we pass along a list of 9 sequences of squares
            assert len(pattern) == 9
            self.house_squares = [self.parse_squares(item) for item in pattern]
            for box in self.house_squares:
                assert len(box) == 9
            all_squares = [square for box in self.house_squares for square in box]
            assert len(set(all_squares)) == 81

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        grid.delete_normal_boxes()
        boxes = [House(House.Type.BOX, i + 1, [grid.matrix[square] for square in self.house_squares[i]])
                 for i in range(len(self.house_squares))]
        grid.houses.extend(boxes)

    def draw(self, context: DrawContext) -> None:
        context.draw_normal_boxes = False
        for squares in self.house_squares:
            context.draw_outline(squares, inset=0, color='black', linestyle='solid', linewidth=3)


class PalindromeFeature(Feature):
    squares: Sequence[tuple]
    color: str

    def __init__(self, squares: SquaresParseable, color: Optional[str] = None) -> None:
        super().__init__()
        self.squares = self.parse_squares(squares)
        self.color = color or 'blue'

    def start(self) -> None:
        same_value_handler = self.grid.same_value_handler
        cells = [self @ square for square in self.squares]
        count = len(cells) // 2
        for cell1, cell2 in zip(cells[:count], cells[::-1]):
            same_value_handler.make_cells_same_value(cell1, cell2, name=f'{cell1}={cell2}')

    def draw(self, context: DrawContext) -> None:
        context.draw_line(self.squares, color=self.color)


class CloneBoxFeature(Feature):
    index1: int
    index2: int

    def __init__(self, index1: int, index2: int) -> None:
        super().__init__()
        self.index1 = index1
        self.index2 = index2

    def start(self):
        same_value_handler = self.grid.same_value_handler
        squares1 = self.get_house_squares(House.Type.BOX, self.index1)
        squares2 = self.get_house_squares(House.Type.BOX, self.index2)
        for square1, square2 in zip(squares1, squares2):
            cell1, cell2 = self @ square1, self @ square2
            same_value_handler.make_cells_same_value(cell1, cell2, name=f'{cell1}={cell2})')


class XVFeature(Feature):
    values: dict[tuple[Square, Square], int]
    all_listed: bool
    all_totals: frozenset[int]

    def __init__(self, *,
                 across: Mapping[int, SquaresParseable],
                 down: Mapping[int, SquaresParseable],
                 all_listed: bool = True,
                 all_values: Optional[set[int]] = None):
        super().__init__(name="XVFeature")
        across = {total: Feature.parse_squares(squares) for total, squares in across.items()}
        down = {total: Feature.parse_squares(squares) for total, squares in down.items()}

        values = {((row, column), (row, column + 1)): total
                  for total, squares in across.items() for row, column in squares}
        values |= {((row, column), (row + 1, column)): total
                   for total, squares in down.items() for row, column in squares}
        self.values = values
        self.all_listed = all_listed
        self.all_totals = frozenset(all_values) if all_values else frozenset(across.keys()) | frozenset(down.keys())

    def start(self):
        features: list[Feature] = [self._Helper(squares, total=total) for squares, total in self.values.items()]
        if self.all_listed:
            features.extend(self._Helper(pair, not_total=self.all_totals)
                            for row, column in product(range(1, 10), range(1, 9))
                            for pair in [((row, column), (row, column + 1))]
                            if pair not in self.values)
            features.extend(self._Helper(pair, not_total=self.all_totals)
                            for row, column in product(range(1, 9), range(1, 10))
                            for pair in [((row, column), (row + 1, column))]
                            if pair not in self.values)
        for feature in features:
            feature.initialize(self.grid)
            feature.start()

    CHARACTER_MAP = {5: 'V', 10: 'X', 15: 'XV'}

    def draw(self, context: DrawContext) -> None:
        for ((y1, x1), (y2, x2)), total in self.values.items():
            character = self.CHARACTER_MAP.get(total) or str(total)
            context.draw_text((x1 + x2 + 1) / 2, (y1 + y2 + 1) / 2, character, va='center', ha='center')

    class _Helper(AdjacentRelationshipFeature):
        total: Optional[int]
        not_total: Optional[frozenset[int]]

        def __init__(self, squares: tuple[Square, Square], *,
                     total: Optional[int] = None, not_total: Optional[frozenset[int]] = None):
            super().__init__(squares)
            self.total = total
            self.not_total = not_total

        def match(self, i: int, j: int) -> bool:
            return i + j == self.total if self.total is not None else i + j not in self.not_total


class KropkeDotFeature(AdjacentRelationshipFeature):
    is_black: bool

    def __init__(self, squares: SquaresParseable, *, color: str) -> None:
        super().__init__(squares)
        assert color == 'white' or color == 'black'
        self.is_black = (color == 'black')

    def match(self, i: int, j: int) -> bool:
        if self.is_black:
            return i == 2 * j or j == 2 * i
        else:
            return abs(i - j) == 1

    def draw(self, context: DrawContext) -> None:
        for (y1, x1), (y2, x2) in pairwise(self.squares):
            context.draw_circle(((x1 + x2 + 1) / 2, (y1 + y2 + 1) / 2), radius=.2, fill=self.is_black, color='black')


class AdjacentNotConsecutiveFeature(FullGridAdjacencyFeature):
    def __init__(self):
        super().__init__(name="Adjacent≠")

    def match(self, i, j):
        return abs(i - j) != 1


class KillerCageFeature(PossibilitiesFeature):
    """The values in the cage must all be different.  They must sum to the total"""
    total: Optional[int]

    def __init__(self, total: Optional[int], squares: SquaresParseable, *, name: Optional[str] = None):
        squares = self.parse_squares(squares)
        r, c = squares[0]
        name = name or f'KillerCage={total}@r{r}c{c}'
        self.total = total
        super().__init__(squares, name=name)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        count = len(self.squares)
        if self.total is not None:
            for values in permutations(range(1, 10), count - 1):
                last_value = self.total - sum(values)
                if 1 <= last_value <= 9 and last_value not in values:
                    yield *values, last_value
        else:
            yield from permutations(range(1, 10), count)

    def draw(self, context: DrawContext) -> None:
        context.draw_outline(self.squares)
        row, column = min(self.squares)
        context.draw_text(column + .2, row + .2, str(self.total),
                          va='top', ha='left', fontsize=10, weight='bold')


class FakeKillerCageFeature(Feature):
    squares: Sequence[Square]
    show_total: bool
    color: Optional[str]

    DEFAULT_FACE_COLOR = '#a89dbc'

    def __init__(self, squares: SquaresParseable, show_total: bool = True,
                 color: Optional[str] = None):
        super().__init__()
        self.squares = Feature.parse_squares(squares)
        self.show_total = show_total
        if color == "default":
            color = self.DEFAULT_FACE_COLOR
        self.color = color

    def draw(self, context: DrawContext) -> None:
        if self.color is not None:
            context.draw_rectangles(self.squares, facecolor=self.color)
        context.draw_outline(self.squares)
        if self.show_total:
            if all((self @ square).is_known for square in self.squares):
                total = sum((self @ square).known_value for square in self.squares)
                row, column = min(self.squares)
                context.draw_text(column + .2, row + .2, str(total),
                                  va='top', ha='left', color='blue', fontsize=10, weight='bold')
                if context.done:
                    print(f'KillerCageFeature({total}, {self.squares})')


class ArrowSumFeature(PossibilitiesFeature):
    """The sum of the values in the arrow must equal the digit in the head of the array"""
    def __init__(self, squares: SquaresParseable):
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
        context.draw_circle((x + .5, y+.5), radius=.35, fill=False, color='gray')
        context.draw_line(self.squares)
        (y0, x0), (y1, x1) = self.squares[-2], self.squares[-1]
        context.draw_arrow(x0 + .5, y0 + .5, x1 - x0, y1 - y0)


class ExtremeEndpointsFeature(PossibilitiesFeature):
    """The values in the middle of the arrow must be strictly in between the values of the two endpoints"""
    def __init__(self, squares: SquaresParseable):
        super().__init__(squares, neighbors=True)

    @staticmethod
    def between(square1: Square, square2: Square) -> ExtremeEndpointsFeature:
        (r1, c1), (r2, c2) = square1, square2
        dr, dc = r2 - r1, c2 - c1
        distance = max(abs(dr), abs(dc))
        dr, dc = dr // distance, dc // distance
        squares = [square1]
        while squares[-1] != square2:
            r1, c1 = r1 + dr, c1 + dc
            squares.append((r1, c1))
        return ExtremeEndpointsFeature(squares)

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


class LocalMinOrMaxFeature(Feature):
    """Reds must be larger than all of its neighbors.  Greens must be smaller than all of its neighbors"""
    reds: Sequence[Square]
    greens: Sequence[Square]

    def __init__(self, *, reds: SquaresParseable = (), greens: SquaresParseable = ()) -> None:
        super().__init__(name="LocalMinMax")
        self.reds = Feature.parse_squares(reds)
        self.greens = Feature.parse_squares(greens)

    def start(self):
        features = [
            *[self._LocalMinMaxFeature(square, high=True) for square in self.reds],
            *[self._LocalMinMaxFeature(square, high=False) for square in self.greens],
        ]
        for feature in features:
            feature.initialize(self.grid)
            feature.start()

    def draw(self, context: DrawContext) -> None:
        for color, squares in (('#FCA0A0', self.reds), ('#B0FEB0', self.greens)):
            for y, x in squares:
                context.draw_rectangle((x, y), width=1, height=1, color=color, fill=True)

    class _LocalMinMaxFeature(PossibilitiesFeature):
        high: bool

        def __init__(self, square: Square, high: bool):
            squares = [square, *self.__orthogonal_neighbors(square)]
            self.high = high
            r, c = square
            name = f'{"High" if high else "Low"}@r{r}c{c}'
            super().__init__(squares, name=name, neighbors=True)

        def get_possibilities(self) -> list[tuple[int, ...]]:
            count = len(self.squares) - 1
            for center in range(1, 10):
                outside_range = range(1, center) if self.high else range(center + 1, 10)
                for outside in product(outside_range, repeat=count):
                    yield center, *outside

        @staticmethod
        def __orthogonal_neighbors(square):
            row, column = square
            return [(r, c) for r, c in ((row + 1, column), (row - 1, column), (row, column + 1), (row, column - 1))
                    if 1 <= r <= 9 and 1 <= c <= 9]


class LittleKillerFeature(PossibilitiesFeature):
    """Typically done via a diagonal.  The sum of the diagonal must total a specific value"""
    ranges: ClassVar[Sequence[range]] = (None, range(1, 9 + 1), range(3, 17 + 1), range(6, 24 + 1))
    ranges_dict: ClassVar[Any] = None
    total: int
    direction: Square

    def __init__(self, total: int, start: Square | str, direction: Square | str):
        self.total = total
        self.direction = dr, dc = self.parse_direction(direction)
        row, column = self.parse_square(start)
        squares = []
        while 1 <= row <= 9 and 1 <= column <= 9:
            squares.append((row, column))
            row, column = row + dr, column + dc
        super().__init__(squares)
        if not self.ranges_dict:
            ranges_dict = defaultdict(list)
            for count in (1, 2, 3):
                for values in permutations(range(1, 10), count):
                    ranges_dict[count, sum(values)].append(values)
            self.ranges_dict = ranges_dict

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
        (y, x), (dy, dx) = self.squares[0], self.direction
        context.draw_text(x - dx + .5, y - dy + .5, str(self.total),
                          va='center', ha='center', fontsize=25, color='black', weight='light')
        context.draw_arrow(x - dx + .5, y - dy + .5, .5 * dx, .5 * dy)


class ValuesAroundIntersectionFeature(PossibilitiesFeature):
    """Up to four numbers are in an intersection.  The values must be surrounding the intersection"""
    values: Sequence[int]

    def __init__(self, *, top_left: Square | str, values: Sequence[int]):
        top_left = self.parse_square(top_left)
        row, column = top_left
        squares = [(row, column), (row, column + 1), (row + 1, column + 1), (row + 1, column)]
        self.values = values
        name = f'Quad {"".join(str(value) for value in sorted(self.values))}@r{row}c{column}'
        super().__init__(squares, name=name, neighbors=True, duplicates=True)

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
                          fontsize=10, color='black', va='center', ha='center')


class RenbanFeature(PossibilitiesFeature):
    color: Optional[str]

    def __init__(self, squares: SquaresParseable, color: Optional[str] = None):
        super().__init__(squares)
        self.color = color

    def get_possibilities(self) -> list[tuple[set[int], ...]]:
        count = len(self.squares)
        for i in range(1, 11 - count):
            yield from permutations(range(i, i + count))

    def draw(self, context: DrawContext) -> None:
        context.draw_line(self.squares, color=self.color or 'lightgrey', linewidth=5)


class MessageFeature(Feature):
    mapping: dict[str, list[Square]]
    box_of_nine_feature: BoxOfNineFeature

    def __init__(self, letters: str, squares: SquaresParseable) -> None:
        super().__init__(name="message")
        mapping: dict[str, list[Square]] = defaultdict(list)
        for letter, square in zip(letters, Feature.parse_squares(squares)):
            mapping[letter].append(square)
        assert len(mapping) == 9   # We'll fix this later, maybe
        self.mapping = mapping
        self.box_of_nine_feature = BoxOfNineFeature(squares=[squares[0] for squares in mapping.values()], show=False)

    def initialize(self, grid) -> None:
        super().initialize(grid)
        self.box_of_nine_feature.initialize(grid)

    def start(self):
        self.box_of_nine_feature.start()
        for letter, squares in self.mapping.items():
            cells = [self @ square for square in squares]
            self.grid.same_value_handler.make_cells_same_value(*cells, name=f"Letter {letter}")

    def draw(self, context: DrawContext) -> None:
        for letter, squares in self.mapping.items():
            for y, x in squares:
                context.draw_text(x, y, letter, weight='bold', fontsize=10, va='top', ha='left')
        if context.done and context.result:
            # Print the values of the letters in numeric order
            value_map = {(self @ squares[0]).known_value: letter for letter, squares in self.mapping.items()}
            pieces = [f'{value}={value_map[value]}' for value in sorted(value_map)]
            print('Message:', ', '.join(pieces))


class ArithmeticFeature(PossibilitiesFeature):
    total: Optional[int]
    operation: str
    info: str

    @classmethod
    def get(cls, square: Square | str, info: str) -> Feature:
        square = Feature.parse_square(square)
        match = re.match(r'([?]|\d+)([-+x/]|)', info)
        digits, symbol = match.groups()
        if not symbol:
            values = [int(x) for x in digits]
            return ValuesAroundIntersectionFeature(top_left=square, values=values)
        else:
            total = None if digits == '?' else int(digits)
            return ArithmeticFeature(top_left=square, operation=symbol, total=total, info=info)

    def __init__(self, top_left: Square, operation: str, total: Optional[int], info: str) -> None:
        r, c = top_left
        self.total = total
        self.operation = operation
        self.info = info
        squares = ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
        super().__init__(squares, name=f'{info}@r{r}c{c}', neighbors=True)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        total, operation = self.total, self.operation
        op = {'+': lambda x, y: x + y,
              'x': lambda x, y: x * y,
              '-': lambda x, y: abs(x - y),
              '/': lambda x, y: max(x, y) // min(x, y) if max(x, y) % min(x, y) == 0 else None,
              }[operation]

        for a, b, c, d in product(range(1, 10), repeat=4):
            if a != b and c != d and a != c and b != d:
                value1, value2 = op(a, d), op(b, c)
                if value1 == value2 and value1 is not None and (total is None or total == value1):
                    yield a, b, c, d

    def draw(self, context: DrawContext):
        y, x = self.squares[0]
        context.draw_circle((x + 1, y + 1), radius=.3, fill=False)
        context.draw_text(x + 1, y + 1, self.info, fontsize=12, color='black', weight='bold', va='center', ha='center')


class DrawOnlyFeature(Feature):
    drawer: Callable[[DrawContext], None]

    def __init__(self, drawer: Callable[[DrawContext], None]) -> None:
        super().__init__()
        self.drawer = drawer

    def draw(self, context: DrawContext) -> None:
        self.drawer(context)


class SimonSaysFeature(Feature):
    """
    For those cases where I need help.  Subclass this method and define methods named round_1, round_2,
    etc to do work that we couldn't otherwise figure out.
    """
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
