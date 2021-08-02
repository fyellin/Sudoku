from __future__ import annotations

import abc
import datetime
import itertools
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from tools.dancing_links import DancingLinks


class Sudoku:
    initial_grid: Mapping[tuple[int, int], int]
    constraints: dict[tuple[int, int, int], list[str]]
    optional_constraints: set[str]
    deletions: set[tuple[int, int, int]]
    features: Sequence[Feature]

    def solve(self, puzzle: str, features: Sequence[Feature], *, show: bool = False) -> None:
        self.initial_grid = {(row, column): int(letter)
                             for (row, column), letter in zip(itertools.product(range(1, 10), repeat=2), puzzle)
                             if '1' <= letter <= '9'}

        self.constraints = {
            (row, column, value): [f"V{row}{column}", f"R{row}={value}", f"C{column}={value}", f"B{box}={value}"]
            for row, column in itertools.product(range(1, 10), repeat=2)
            for box in [row - (row - 1) % 3 + (column - 1) // 3]
            for value in range(1, 10)}
        self.optional_constraints = set()
        self.features = features
        self.deletions = {(row, column, valuex)
                          for (row, column), value in self.initial_grid.items()
                          for valuex in range(1, 10) if valuex != value}

        for feature in features:
            feature.update_constraints(self)

        for key in self.deletions:
            self.constraints.pop(key)

        if show:
            self.draw_grid([(row, column, value) for ((row, column), value) in self.initial_grid.items()])
            return

        for row, constraints in self.constraints.items():
            assert len(constraints) == len(set(constraints))

        links = DancingLinks(self.constraints, optional_constraints=self.optional_constraints,
                             row_printer=self.draw_grid)
        links.solve(recursive=False)

    def draw_grid(self, results: Sequence[tuple[int, int, int]]) -> None:
        if not all(feature.check_solution(results) for feature in self.features):
            return

        figure, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=100)

        # set (1,1) as the top-left corner, and (max_column, max_row) as the bottom right.
        axes.axis([1, 10, 10, 1])
        axes.axis('equal')
        axes.axis('off')
        figure.tight_layout()

        for feature in self.features:
            feature.pre_print()

        # Draw the bold outline
        for x in range(1, 11):
            width = 3 if x in (1, 4, 7, 10) else 1
            axes.plot([x, x], [1, 10], linewidth=width, color='black')
            axes.plot([1, 10], [x, x], linewidth=width, color='black')

        given = dict(fontsize=13, color='black', weight='heavy')
        found = dict(fontsize=12, color='blue', weight='normal')
        for row, column, value in results:
            args = given if (row, column) in self.initial_grid else found
            axes.text(column + .5, row + .5, str(value),
                      va='center', ha='center', **args)

        for feature in self.features:
            feature.post_print(results)

        temp = ''.join(str(value) for _, _, value in sorted(results))
        print(f"'{temp}'")
        plt.show()

    def not_both(self, triple1: tuple[int, int, int], triple2: tuple[int, int, int], name: str = '') -> None:
        row1, column1, value1 = triple1
        row2, column2, value2 = triple2
        constraint = f'{name}:r{row1}c{column1}={value1};r{row2}c{column2}={value2}'
        self.optional_constraints.add(constraint)
        self.constraints[triple1].append(constraint)
        self.constraints[triple2].append(constraint)


class Feature(abc.ABC):
    index: int
    counter: ClassVar[int] = 0

    def __init__(self) -> None:
        Feature.counter += 1
        self.index = Feature.counter

    @abc.abstractmethod
    def update_constraints(self, sudoku: Sudoku) -> None: ...

    def pre_print(self) -> None:
        pass

    def post_print(self, results: Sequence[tuple[int, int, int]]) -> None:
        pass

    def check_solution(self, _results: Sequence[tuple[int, int, int]]) -> bool:
        return True

    @staticmethod
    def draw_line(points: Sequence[tuple[int, int]], *, closed: bool = False, **kwargs: Any) -> None:
        ys = [row + .5 for row, _ in points]
        xs = [column + .5 for _, column in points]
        if closed:
            ys.append(ys[0])
            xs.append(xs[0])
        plt.plot(xs, ys, **{'color': 'black', **kwargs})


class ChessFeature(Feature):
    king: bool
    knight: bool

    def __init__(self, *, king: bool = False, knight: bool = False) -> None:
        super().__init__()
        self.king = king
        self.knight = knight

    def update_constraints(self, sudoku: Sudoku) -> None:
        for row, column in itertools.product(range(1, 10), repeat=2):
            neighbors: set[tuple[int, int]] = set()
            if self.knight:
                neighbors.update(self.knights_move(row, column))
            if self.king:
                neighbors.update(self.kings_move(row, column))
            # For each "neighbor", we add an optional constraint so that either this cell can have a value or the
            # neighbor can have the value.  This ensures that at most one of them will have the specified value
            for row2, column2 in neighbors:
                if (row, column) < (row2, column2):
                    for value in range(1, 10):
                        sudoku.not_both((row, column, value), (row2, column2, value), 'C')

    @staticmethod
    def knights_move(row: int, column: int) -> Sequence[tuple[int, int]]:
        return [((row + dr), (column + dc))
                for dx, dy in itertools.product((1, -1), (2, -2))
                for dr, dc in ((dx, dy), (dy, dx))
                if 1 <= row + dr <= 9 and 1 <= column + dc <= 9]

    @staticmethod
    def kings_move(row: int, column: int) -> Sequence[tuple[int, int]]:
        return [((row + dr), (column + dc))
                for dr, dc in itertools.product((-1, 1), repeat=2)
                if 1 <= row + dr <= 9 and 1 <= column + dc <= 9]


class AdjacentFeature(Feature):
    def update_constraints(self, sudoku: Sudoku) -> None:
        for row, column in itertools.product(range(1, 10), repeat=2):
            for row2, column2 in self.adjacent_move(row, column):
                if (row, column) < (row2, column2):
                    for value in range(1, 9):
                        sudoku.not_both((row, column, value), (row2, column2, value + 1))
                    for value in range(2, 10):
                        sudoku.not_both((row, column, value), (row2, column2, value - 1))

    @staticmethod
    def adjacent_move(row: int, column: int) -> Sequence[tuple[int, int]]:
        return [((row + dr), (column + dc))
                for dx in (-1, 1)
                for dr, dc in ((dx, 0), (0, dx))
                if 1 <= row + dr <= 9 and 1 <= column + dc <= 9]


class ThermometerFeature(Feature):
    thermometer: Sequence[tuple[int, int]]

    def __init__(self, thermometer: Sequence[tuple[int, int]]):
        super().__init__()
        self.thermometer = thermometer

    def update_constraints(self, sudoku: Sudoku) -> None:
        length = len(self.thermometer)
        span = 10 - length  # number of values each element in thermometer can have
        for minimum, (row, column) in enumerate(self.thermometer, start=1):
            maximum = minimum + span - 1
            sudoku.deletions.update((row, column, value) for value in range(1, 10)
                                    if not minimum <= value <= maximum)
        prefix = f'T{self.index}'
        for (index1, (row1, col1)), (index2, (row2, col2)) in itertools.combinations(enumerate(self.thermometer), 2):
            for value1, value2 in itertools.product(range(1, 10), repeat=2):
                if value2 < value1 + (index2 - index1):
                    sudoku.not_both((row1, col1, value1), (row2, col2, value2), prefix)

    def pre_print(self) -> None:
        self.draw_line(self.thermometer, color='lightgrey', linewidth=5)
        row, column = self.thermometer[0]
        plt.gca().add_patch(plt.Circle((column + .5, row + .5), radius=.3, fill=True, facecolor='lightgrey'))


class SnakeFeature(Feature):
    snake: Sequence[tuple[int, int]]
    closed: bool
    color: str

    def __init__(self, snake: Sequence[tuple[int, int]], *, closed: bool = False, color: str = 'lightgrey'):
        super().__init__()
        self.snake = snake
        self.closed = closed
        self.color = color

    def update_constraints(self, sudoku: Sudoku) -> None:
        for value in range(1, 10):
            constraint = f"Snake{self.index}={value}"
            # If the snake has length 9, each of the constraints must be fulfilled.  Otherwise, they're optional.
            if len(self.snake) < 9:
                sudoku.optional_constraints.add(constraint)
            for row, column in self.snake:
                sudoku.constraints[row, column, value].append(constraint)

    def pre_print(self) -> None:
        self.draw_line(self.snake, color=self.color, linewidth=5, closed=self.closed)


class MagicSquareFeature(Feature):
    center: tuple[int, int]

    def __init__(self, center: tuple[int, int] = (5, 5)):
        super().__init__()
        self.center = center

    def update_constraints(self, sudoku: Sudoku) -> None:
        deltas = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
        values = (2, 9, 4, 3, 8, 1, 6, 7)

        center_x, center_y = self.center
        squares = [(center_x + delta_x, center_y + delta_y) for delta_x, delta_y in deltas]

        # The center must be five, the corners must be even, and the others must be odd.
        deletions = sudoku.deletions
        deletions.update((center_x, center_y, value) for value in range(1, 10) if value != 5)
        deletions.update((row, column, value) for (row, column) in squares[0::2] for value in (1, 3, 5, 7, 9))
        deletions.update((row, column, value) for (row, column) in squares[1::2] for value in (2, 4, 5, 6, 8))

        prefix = f'MS{self.index}'
        for (i1, (row1, col1)), (i2, (row2, col2)) in itertools.combinations(enumerate(squares), 2):
            delta = i2 - i1
            for value_index in (0, 2, 4, 6):
                value1 = values[(value_index + i1) % 8]
                value2a = values[(value_index + i1 + delta) % 8]
                value2b = values[(value_index + i1 - delta) % 8]
                for value2 in range(1, 10):
                    if value2 != value2a and value2 != value2b:
                        sudoku.not_both((row1, col1, value1), (row2, col2, value2), prefix)


class GermanSnakeFeature(Feature):
    cup: Sequence[tuple[int, int]]

    def __init__(self, cup: Sequence[tuple[int, int]]):
        super().__init__()
        self.cup = cup

    def update_constraints(self, sudoku: Sudoku) -> None:
        for row, column in self.cup:
            sudoku.deletions.add((row, column, 5))
        for ((row1, col1), (row2, col2)) in zip(self.cup, self.cup[1:]):
            for value1, value2 in itertools.product(range(1, 10), repeat=2):
                if abs(value1 - value2) < 5:
                    sudoku.not_both((row1, col1, value1), (row2, col2, value2))

    def pre_print(self) -> None:
        self.draw_line(self.cup)


class TaxicabFeature(Feature):
    def update_constraints(self, sudoku: Sudoku) -> None:
        for row1, col1, row2, col2 in itertools.product(range(1, 10), repeat=4):
            if (row1, col1) < (row2, col2):
                delta = abs(row1 - row2) + abs(col1 - col2)
                if delta <= 9:
                    sudoku.not_both((row1, col1, delta), (row2, col2, delta))


class MarvolosRingFeature(Feature):
    circle = ((2, 4), (2, 5), (2, 6), (3, 7), (4, 8), (5, 8), (6, 8), (7, 7),
              (8, 6), (8, 5), (8, 4), (7, 3), (6, 2), (5, 2), (4, 2), (3, 3))

    def update_constraints(self, sudoku: Sudoku) -> None:
        for ((row1, col1), (row2, col2)) in zip(self.circle, self.circle[1:] + self.circle[:1]):
            for value1, value2 in itertools.product(range(1, 10), repeat=2):
                if value1 + value2 not in (1, 4, 8, 9, 16):
                    sudoku.not_both((row1, col1, value1), (row2, col2, value2), 'c')

    def check_solution(self, results: Sequence[tuple[int, int, int]]) -> bool:
        values_in_circle = {value for (row, column, value) in results if (row, column) in self.circle}
        return len(values_in_circle) == 9

    def pre_print(self) -> None:
        plt.gca().add_patch(plt.Circle((5.5, 5.5), radius=3, fill=False, facecolor='black'))


class ContainsTextFeature(Feature):
    row: int
    text: Sequence[int]

    def __init__(self, row: int, text: Sequence[int]) -> None:
        super().__init__()
        self.row = row
        self.text = text

    def update_constraints(self, sudoku: Sudoku) -> None:
        span = 10 - len(self.text)
        not_text = [x for x in range(1, 10) if x not in self.text]
        for index, digit in enumerate(self.text, start=1):
            min_column = index
            max_column = index + span
            sudoku.deletions.update((self.row, column, digit) for column in range(1, 10)
                                    if not min_column <= column <= max_column)

        for (index1, digit1), (index2, digit2) in itertools.combinations(enumerate(self.text, start=1), 2):
            delta = index2 - index1
            for column1, column2 in itertools.permutations(range(1, 10), 2):
                if column2 != column1 + delta:
                    sudoku.not_both((self.row, column1, digit1), (self.row, column2, digit2), 'A')

        for column in range(1, span + 1):
            # text goes between column and column + len(text) - 1
            for index, digit in enumerate(self.text):
                for column2 in range(column, column + len(self.text)):
                    if column2 != column:
                        for digit2 in not_text:
                            sudoku.not_both((self.row, column + index, digit), (self.row, column2, digit2), 'B')


class DrawCirclesFeature(Feature):
    circles: Sequence[tuple[int, int]]

    def __init__(self, circles: Sequence[tuple[int, int]]):
        super().__init__()
        self.circles = circles

    def update_constraints(self, sudoku: Sudoku) -> None:
        pass

    def pre_print(self) -> None:
        for row, column in self.circles:
            plt.gca().add_patch(plt.Circle((column + .5, row + .5), radius=.5, fill=False, facecolor='black'))

    def post_print(self, results: Sequence[tuple[int, int, int]]) -> None:
        mapping = {(row, column): value for (row, column, value) in results}
        grid = ['.'] * 81
        for row, column in self.circles:
            value = mapping.get((row, column), '.')
            grid[(row - 1) * 9 + (column - 1)] = str(value)
        print(''.join(grid))


class EvenFeature(Feature):
    evens: Sequence[tuple[int, int]]

    def __init__(self, evens:  Sequence[tuple[int, int]]):
        super().__init__()
        self.evens = evens

    def update_constraints(self, sudoku: Sudoku) -> None:
        sudoku.deletions.update((row, column, value) for (row, column) in self.evens for value in (1, 3, 5, 7, 9))


class CheckEggFeature(Feature):
    eggs: Sequence[list[tuple[int, int]]]

    def __init__(self, pattern: str) -> None:
        super().__init__()
        assert len(pattern) == 81
        info: Sequence[list[tuple[int, int]]] = [list() for _ in range(10)]
        for (row, column), letter in zip(itertools.product(range(1, 10), repeat=2), pattern):
            if '0' <= letter <= '9':
                info[int(letter)].append((row, column))
        for i in range(0, 9):
            assert len(info[i]) == i
        self.eggs = info

    def update_constraints(self, sudoku: Sudoku) -> None:
        for size in range(1, 10):
            for (row, column) in self.eggs[size]:
                sudoku.deletions.update((row, column, value) for value in range(size + 1, 10))
            for value in range(1, size + 1):
                constraint = f'Snake{size}{value}'
                for (row, column) in self.eggs[size]:
                    sudoku.constraints[row, column, value].append(constraint)

    def pre_print(self) -> None:
        cells = {cell for size in range(1, 10) for cell in self.eggs[size]}
        for row, column in itertools.product(range(1, 10), repeat=2):
            if (row, column) not in cells:
                plt.gca().add_patch(plt.Rectangle((column, row), 1, 1, facecolor='lightblue'))


class PlusFeature(Feature):
    squares: Sequence[tuple[int, int]]
    puzzles: Sequence[str]

    def __init__(self, squares: Sequence[tuple[int, int]], puzzles: Sequence[str]) -> None:
        super().__init__()
        self.squares = squares
        self.puzzles = puzzles

    def update_constraints(self, sudoku: Sudoku) -> None:
        for row, column in self.squares:
            value = self.__get_value(row, column)
            sudoku.deletions.update((row, column, valuex) for valuex in range(1, 10) if valuex != value)

    def __get_value(self, row: int, column: int) -> int:
        index = (row - 1) * 9 + (column - 1)
        value = sum(int(puzzle[index]) for puzzle in self.puzzles) % 9
        if value == 0:
            value = 9
        return value

    def pre_print(self) -> None:
        for row, column in self.squares:
            plt.plot((column + .2, column + .8), (row + .5, row + .5), color='lightgrey', linewidth=3)
            plt.plot((column + .5, column + .5), (row + .2, row + .8), color='lightgrey', linewidth=3)


class ColorFeature(Feature):
    setup: Mapping[tuple[int, int], str]
    color_map: Mapping[str, str]
    plus_feature: PlusFeature

    def __init__(self, grid: str, color_map: str, puzzles: Sequence[str]) -> None:
        super().__init__()
        self.setup = {(row, column): letter
                      for (row, column), letter in zip(itertools.product(range(1, 10), repeat=2), grid)
                      if letter != '.' and letter != '+'}
        pluses = [(row, column)
                  for (row, column), letter in zip(itertools.product(range(1, 10), repeat=2), grid)
                  if letter == '+']
        self.color_map = dict(zip(color_map, puzzles))
        self.plus_feature = PlusFeature(pluses, puzzles)

    def update_constraints(self, sudoku: Sudoku) -> None:
        self.plus_feature.update_constraints(sudoku)
        for (row, column), letter in self.setup.items():
            puzzle = self.color_map[letter]
            index = (row - 1) * 9 + (column - 1)
            value = int(puzzle[index])
            sudoku.deletions.update((row, column, valuex) for valuex in range(1, 10) if valuex != value)

    CIRCLES = dict(r="lightcoral", p="violet", o="bisque", g="lightgreen", G="lightgray", y="yellow", b="skyblue")

    def pre_print(self) -> None:
        self.plus_feature.pre_print()
        axis = plt.gca()
        for (row, column), letter in self.setup.items():
            axis.add_patch(plt.Circle((column + .5, row + .5), radius=.4, fill=True,
                                      color=self.CIRCLES[letter]))
        # noinspection PyTypeChecker
        axis.add_patch(FancyBboxPatch((2.3, 5.3), 6.4, 0.4, boxstyle='round, pad=0.2', fill=False))


class LittlePrincessFeature(Feature):
    def update_constraints(self, sudoku: Sudoku) -> None:
        squares = [(row, column) for row, column in itertools.product(range(1, 10), repeat=2)]
        for (r1, c1), (r2, c2) in itertools.combinations(squares, 2):
            dr, dc = abs(r1 - r2), abs(c1 - c2)
            if dr == dc:
                for value in range(dr + 1, 10):
                    sudoku.not_both((r1, c1, value), (r2, c2, value))


def merge(p1: str, p2: str) -> str:
    assert len(p1) == len(p2) == 81
    assert(p1[i] == '.' or p2[i] == '.' or p1[i] == p2[i] for i in range(81))
    result = ((y if x == '.' else x) for x, y in zip(p1, p2))
    return ''.join(result)


class ThermosSudoku(Sudoku):
    thermometers = [[(2, 2), (1, 3), (1, 4), (1, 5), (2, 6)],
                    [(2, 2), (3, 1), (4, 1), (5, 1), (6, 2)],
                    [(2, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8)],
                    [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (6, 8)],
                    [(3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 7)],
                    [(4, 2), (5, 2), (6, 3), (7, 4), (8, 4)],
                    [(1, 7), (1, 8)],
                    [(8, 8), (8, 9)],
                    [(8, 2), (8, 3)]]

    @staticmethod
    def run() -> None:
        sudoku = ThermosSudoku()
        puzzle = ' ' * 81
        features = [ThermometerFeature(thermometer) for thermometer in ThermosSudoku.thermometers]
        # features = [ ()]
        sudoku.solve(puzzle, features)


def puzzle1() -> None:
    # XUZZ = "123456789123456789123456789123456789123456789123456789123456789123456789123456789"
    puzzle = "...6.1.....4...2...1.....6.1.......2....8....6.......4.7.....9...1...4.....1.2..3"
    texts = [(3, 1, 8), *[(x, 9) for x in range(1, 9)]]
    features: list[Feature] = [ContainsTextFeature(i, text) for i, text in enumerate(texts, start=1)]
    features.append(DrawCirclesFeature([(5, 8), (6, 8), (8, 4)]))
    Sudoku().solve(puzzle, features)


def puzzle2() -> None:
    previo = '...........................................5........3.............7..............'
    puzzle = '.9...16....................8............9............8....................16...8.'
    features = [MarvolosRingFeature(),
                DrawCirclesFeature([(1, 5), (2, 5), (4, 1), (4, 7), (5, 9), (7, 1)])]
    Sudoku().solve(merge(previo, puzzle), features)


def puzzle3() -> None:
    previo = '....4........6.............8.....9..........6.........2..........................'
    puzzle = '..9...7...5.....3.7.4.....9.............5.............5.....8.1.3.....9...7...5..'

    evens = [(2, 3), (3, 2), (3, 3), (1, 4), (1, 5), (1, 6)]
    evens = evens + [(column, 10 - row) for row, column in evens]
    evens = evens + [(10 - row, 10 - column) for row, column in evens]

    features = [SnakeFeature([(3, 6), (3, 5), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6), (7, 5), (7, 4)]),
                EvenFeature(evens),
                DrawCirclesFeature([(3, 6), (4, 2), (6, 3), (6, 7)])
                ]
    Sudoku().solve(merge(puzzle, previo), features)


def puzzle4() -> None:
    previo = '.......................1....4..................5...1.............................'
    puzzle = '...............5.....6.....' + (54 * '.')

    cup1 = ((1, 2), (2, 2), (3, 3), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (8, 3))
    cup2 = tuple((row, 10 - column) for row, column in cup1)

    features = [GermanSnakeFeature(cup1),
                GermanSnakeFeature(cup2),
                ChessFeature(knight=True),
                DrawCirclesFeature([(1, 3), (2, 1), (2, 7), (5, 3), (5, 6), (7, 1), (7, 2), (8, 7), (9, 3)])
                ]

    Sudoku().solve(merge(puzzle, previo), features)


def puzzle5() -> None:
    previo = '..7......3.....5......................3..8............15.............9....9......'
    puzzle = '......3...1...............72.........................2..................8........'
    diadem = SnakeFeature([(4, 2), (2, 1), (3, 3), (1, 4), (3, 5), (1, 6), (3, 7), (2, 9), (4, 8)],
                          color='lightblue', closed=True)
    thermometers = [ThermometerFeature([(row, column) for row in (9, 8, 7, 6, 5, 4)]) for column in (2, 4, 6, 8)]
    circles = DrawCirclesFeature([(3, 5), (5, 8), (6, 3), (7, 3), (9, 3)])
    features = [diadem, *thermometers, circles]
    Sudoku().solve(merge(puzzle, previo), features)


def puzzle6() -> None:
    previo = '......................5....................6...1........2.................9......'
    puzzle = '......75.2.....4.9.....9.......2.8..5...............3........9...7......4........'
    snakey = '3...777773.5.77...3.5...22...555.....4...8888.4.6.8..8.4.6.88...4.6...1....666...'
    features = [CheckEggFeature(snakey)]
    Sudoku().solve(merge(puzzle, previo), features, show=False)


def puzzle7() -> None:
    puzzles = [
        '925631847364578219718429365153964782249387156687215934472853691531796428896142573',   # Diary, Red
        '398541672517263894642987513865372941123894756974156238289435167456718329731629485',   # Ring, Purple
        '369248715152769438784531269843617952291854376675392184526973841438125697917486523',   # Locket, Orangeish
        '817325496396487521524691783741952638963148257285763149158279364632814975479536812',   # Cup, Yellow
        '527961384318742596694853217285619473473528169961437852152396748746285931839174625',   # Crown, Blue
        '196842753275361489384759126963125847548937261721684935612578394837496512459213678',   # Snake, Green
    ]
    pluses = [(1, 1), (1, 9), (2, 4), (2, 6), (3, 3), (3, 7), (4, 2), (4, 4), (4, 6), (4, 8), (5, 3)]
    pluses = pluses + [(10 - row, 10 - column) for row, column in pluses]
    puzzle = '......................7..1.....8.................6.....3..5......................'
    feature = PlusFeature(pluses, puzzles)
    Sudoku().solve(puzzle, [feature], show=False)


def puzzle8() -> None:
    puzzles = [
        '925631847364578219718429365153964782249387156687215934472853691531796428896142573',   # Diary, Red
        '398541672517263894642987513865372941123894756974156238289435167456718329731629485',   # Ring, Purple
        '369248715152769438784531269843617952291854376675392184526973841438125697917486523',   # Locket, Orangeish
        '817325496396487521524691783741952638963148257285763149158279364632814975479536812',   # Cup, Yellow
        '527961384318742596694853217285619473473528169961437852152396748746285931839174625',   # Crown, Blue
        '196842753275361489384759126963125847548937261721684935612578394837496512459213678',   # Snake, Green
        '213845679976123854548976213361782945859314762724569381632458197185297436497631528',   # Enigma, Gray
    ]

    grid = '+.y..o+.+...Gb.......p...r.+..+b.b+...........+g.g+..+.o...........ry...+.+g..g.+'
    features = [
        ColorFeature(grid, 'rpoybgG', puzzles),
        SnakeFeature([(i, i) for i in range(1, 10)]),
        SnakeFeature([(10 - i, i) for i in range(1, 10)]),
    ]
    Sudoku().solve('.'*81, features, show=False)


def little_princess() -> None:
    puzzle = '.......6...8..........27......6.8.1....4..........9..............7...............'
    Sudoku().solve(puzzle, features=[LittlePrincessFeature()])


def main() -> None:
    start = datetime.datetime.now()
    little_princess()
    end = datetime.datetime.now()
    print(end - start)


if __name__ == '__main__':
    main()
