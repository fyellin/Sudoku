from __future__ import annotations

import abc
import functools
from collections import defaultdict
from collections.abc import Iterable, Sequence, Mapping
from itertools import permutations, product, tee, groupby, combinations_with_replacement
from typing import Optional, ClassVar, Any, Union

from cell import Cell, House, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square, MultiFeature
from grid import Grid
from features.possibilities_feature import PossibilitiesFeature
from features.same_value_feature import SameValueFeature


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
    def major_diagonal(**kwargs: Any) -> BoxOfNineFeature:
        return BoxOfNineFeature([(i, i) for i in range(1, 10)], **kwargs)

    @staticmethod
    def minor_diagonal(**kwargs: Any) -> BoxOfNineFeature:
        return BoxOfNineFeature([(10 - i, i) for i in range(1, 10)], **kwargs)

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
    values: SmallIntSet
    color: Optional[str]

    def __init__(self, squares: Union[Sequence[Square], str], values: Sequence[int], *,
                 color: Optional[str] = None):
        super().__init__()
        self.squares = self.parse_squares(squares)
        self.values = SmallIntSet(values)
        self.color = color

    def reset(self) -> None:
        cells = [self @ x for x in self.squares]
        Cell.keep_values_for_cell(cells, self.values, show=False)

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


class PalindromeFeature(MultiFeature):
    squares: Sequence[Square]
    color: Optional[str]

    def __init__(self, squares: Union[Sequence[Square], str], color: Optional[str] = None):
        self.squares = squares = self.parse_squares(squares)
        self.color = color
        count = len(squares) // 2
        features = [SameValueFeature((square1, square2))
                    for square1, square2 in zip(squares[:count], squares[::-1])]
        super().__init__(features)

    def draw(self, context: DrawContext) -> None:
        context.draw_line(self.squares, color=self.color or 'blue')


class CloneBoxFeature(MultiFeature):
    def __init__(self, index1: int, index2: int):
        squares1 = self.get_house_squares(House.Type.BOX, index1)
        squares2 = self.get_house_squares(House.Type.BOX, index2)
        features = [SameValueFeature(pair) for pair in zip(squares1, squares2)]
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


class AdjacentNotConsecutiveFeature(MultiFeature):
    def __init__(self) -> None:
        super().__init__([self._NotConsecutiveRowOrColumn(self.get_house_squares(htype, i), name=f'{htype.name} #{i}')
                          for htype in [House.Type.ROW, House.Type.COLUMN]
                          for i in range(1, 10)])

    def draw(self, context: DrawContext) -> None:
        pass

    class _NotConsecutiveRowOrColumn(AdjacentRelationshipFeature):
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


class ArrowSumFeature(PossibilitiesFeature):
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


class ExtremeEndpointsFeature(PossibilitiesFeature):
    """The values in the middle of the arrow must be strictly in between the values of the two endpoints"""
    def __init__(self, squares: Union[Sequence[Square], str]):
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


class LocalMinOrMaxFeature(MultiFeature):
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


class ValuesAroundIntersectionFeature(PossibilitiesFeature):
    """Up to four numbers are in an intersection.  The values must be surrounding the intersection"""
    values: Sequence[int]

    def __init__(self, *, top_left: Square, values: Sequence[int]):
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
                          fontsize=10, color='black',
                          verticalalignment='center', horizontalalignment='center')


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
            *[SameValueFeature(squares, name=f'Letter "{letter}"')
              for letter, squares in mapping.items() if len(squares) > 1],
        ]
        super().__init__(features)

    def draw(self, context: DrawContext) -> None:
        if context.done and context.result:
            # Print the values of the letters in numeric order
            value_map = {self @ squares[0]: letter for letter, squares in self.mapping.items()}
            pieces = [f'{value}={value_map[value]}' for value in sorted(value_map)]
            print('Message:', ', '.join(pieces))
        for feature in self.features[1:]:
            feature.draw(context)
        for letter, squares in self.mapping.items():
            for y, x in squares:
                context.draw_text(x, y, letter,
                                  weight='bold',
                                  fontsize=10,
                                  verticalalignment='top', horizontalalignment='left')


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
