import itertools
import math
from typing import Iterable, List, Mapping, Sequence, Set, cast

from cell import Cell, House, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from features.chess_move import KnightsMoveFeature
from features.features import AllValuesPresentFeature, BoxOfNineFeature, LimitedValuesFeature
from features.possibilities_feature import AdjacentRelationshipFeature, HousePossibilitiesFeature
from features.thermometer import ThermometerFeature
from grid import Grid
from human_sudoku import Sudoku


class MalvoloRingFeature(AdjacentRelationshipFeature):
    SQUARES = ((2, 4), (2, 5), (2, 6), (3, 7), (4, 8), (5, 8), (6, 8), (7, 7),
               (8, 6), (8, 5), (8, 4), (7, 3), (6, 2), (5, 2), (4, 2), (3, 3))

    def __init__(self) -> None:
        super().__init__(self.SQUARES)

    def match(self, i: int, j: int) -> bool:
        return i + j in (4, 8, 9, 16)

    def draw(self, context: DrawContext) -> None:
        radius = math.hypot(2.5, 1.5)
        context.draw_circle((5.5, 5.5), radius=radius, fill=False, facecolor='black')

    @classmethod
    def get_all_features(cls) -> list[Feature]:
        return [MalvoloRingFeature(), AllValuesPresentFeature(cls.SQUARES)]


class GermanSnakeFeature(AdjacentRelationshipFeature):
    def __init__(self, squares: SquaresParseable, prefix: str) -> None:
        super().__init__(squares, prefix=prefix)

    def match(self, i: int, j: int) -> bool:
        return abs(i - j) >= 5


class ContainsTextFeature(HousePossibilitiesFeature):
    text: Sequence[int]
    text_length: int

    """A row that must contain certain digits consecutively"""
    def __init__(self, row: int, text: Sequence[int]) -> None:
        super().__init__(House.Type.ROW, row, prefix="Text Row")
        self.text = text
        self.text_length = len(self.text)

    def match(self, permutation: Sequence[int]) -> bool:
        text, text_length = self.text, self.text_length
        text0 = text[0]
        index = permutation.index(text0)
        return index <= 9 - text_length and \
            all(permutation[index + i] == text[i] for i in range(1, text_length))


class SnakesEggFeature(Feature):
    """A snake egg, where eggs of size n have all the digits from 1-n"""

    class Egg(House):
        def __init__(self, index: int, cells: Sequence[Cell]) -> None:
            super().__init__(House.Type.EGG, index, cells)
            self.unknown_values = SmallIntSet(range(1, len(self.cells) + 1))
            Cell.remove_values_from_cells(self.cells, set(range(len(self.cells) + 1, 10)))

    squares: Sequence[List[Square]]

    def __init__(self, pattern: str) -> None:
        super().__init__()
        assert len(pattern) == 81
        info: Sequence[List[Square]] = [list() for _ in range(10)]
        for (row, column), letter in zip(itertools.product(range(1, 10), repeat=2), pattern):
            if '0' <= letter <= '9':
                info[int(letter)].append((row, column))
        for i in range(0, 9):
            assert len(info[i]) == i
        self.squares = info

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        eggs = [self.Egg(i, [self @ square for square in self.squares[i]]) for i in range(1, 9)]
        grid.houses.extend(eggs)

    def draw(self, context: DrawContext) -> None:
        # Find all squares that aren't in one of the eggs.
        snake: Set[Square] = set(cast(Iterable[Square], itertools.product(range(1, 10), repeat=2)))
        snake.difference_update(cell for size in range(1, 10) for cell in self.squares[size])
        context.draw_rectangles(cast(Sequence[Square], snake), facecolor='lightblue')


class PlusFeature(Feature):
    squares: Sequence[Square]
    puzzles: Sequence[str]

    def __init__(self, squares: Sequence[Square], puzzles: Sequence[str]) -> None:
        super().__init__()
        self.squares = squares
        self.puzzles = puzzles

    def start(self) -> None:
        super().start()
        for row, column in self.squares:
            value = self.__get_value(row, column)
            (self @ (row, column)).set_value_to(value)

    def __get_value(self, row: int, column: int) -> int:
        index = (row - 1) * 9 + (column - 1)
        value = sum(int(puzzle[index]) for puzzle in self.puzzles) % 9
        if value == 0:
            value = 9
        return value

    def draw(self, context: DrawContext) -> None:
        for row, column in self.squares:
            context.plot((column + .2, column + .8), (row + .5, row + .5), color='lightgrey', linewidth=3)
            context.plot((column + .5, column + .5), (row + .2, row + .8), color='lightgrey', linewidth=3)


class ColorFeature(Feature):
    setup: Mapping[Square, str]
    color_map: Mapping[str, str]

    def __init__(self, grid: str, color_map: str, puzzles: Sequence[str]) -> None:
        super().__init__()
        self.setup = {(row, column): letter
                      for (row, column), letter in zip(self.all_squares(), grid)
                      if letter != '.' and letter != '+'}
        self.color_map = dict(zip(color_map, puzzles))

    def start(self) -> None:
        super().start()
        for (row, column), letter in self.setup.items():
            puzzle = self.color_map[letter]
            index = (row - 1) * 9 + (column - 1)
            value = int(puzzle[index])
            self.grid.matrix[row, column].set_value_to(value)

    CIRCLES = dict(r="lightcoral", p="violet", o="bisque", g="lightgreen", G="lightgray", y="yellow", b="skyblue")

    def draw(self, context: DrawContext) -> None:
        for (row, column), letter in self.setup.items():
            context.draw_circle((column + .5, row + .5), radius=.4, fill=True, color=self.CIRCLES[letter])
        # noinspection PyTypeChecker
        context.add_fancy_bbox((2.3, 5.3), width=6.4, height=0.4, boxstyle='round, pad=0.2', fill=False)


class DrawCircleFeature(Feature):
    squares: Sequence[Square]

    def __init__(self, squares: Sequence[Square]):
        super().__init__()
        self.squares = squares

    def draw(self, context: DrawContext) -> None:
        for row, column in self.squares:
            context.draw_circle((column + .5, row + .5), radius=.5, fill=False, color='blue')


class DrawBoxFeature(Feature):
    squares: Sequence[Square]

    def __init__(self, squares: Sequence[Square]):
        super().__init__()
        self.squares = squares

    def draw(self, context: DrawContext) -> None:
        context.draw_outline(self.squares)


def merge(p1: str, p2: str) -> str:
    assert len(p1) == len(p2) == 81
    assert(p1[i] == '.' or p2[i] == '.' or p1[i] == p2[i] for i in range(81))
    result = ((y if x == '.' else x) for x, y in zip(p1, p2))
    return ''.join(result)


def puzzle1() -> None:
    # INFO = "123456789123456789123456789123456789123456789123456789123456789123456789123456789"
    puzzle = "...6.1.....4...2...1.....6.1.......2....8....6.......4.7.....9...1...4.....1.2..3"
    texts = [(3, 1, 8), *[(x, 9) for x in range(1, 9)]]
    features: List[Feature] = [ContainsTextFeature(i, text) for i, text in enumerate(texts, start=1)]
    sudoku = Sudoku()
    sudoku.solve(puzzle, features=features)


def puzzle2() -> None:
    previous: str = '...........................................5........3.............7..............'
    puzzle = '.9...16....................8............9............8....................16...8.'
    puzzle = merge(puzzle, previous)
    sudoku = Sudoku()
    sudoku.solve(puzzle, features=MalvoloRingFeature.get_all_features())


def puzzle3() -> None:
    previous = '....4........6.............8.....9..........6.........2..........................'
    puzzle = '..9...7...5.....3.7.4.....9.............5.............5.....8.1.3.....9...7...5..'

    evens = [(2, 3), (3, 2), (3, 3), (1, 4), (1, 5), (1, 6)]
    evens = evens + [(column, 10 - row) for row, column in evens]
    evens = evens + [(10 - row, 10 - column) for row, column in evens]

    features = [BoxOfNineFeature([(3, 6), (3, 5), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6), (7, 5), (7, 4)]),
                LimitedValuesFeature(evens, (2, 4, 6, 8)),
                ]
    sudoku = Sudoku()
    sudoku.solve(merge(puzzle, previous), features=features)


def puzzle4() -> None:
    previous = '.......................1....4..................5...1.............................'
    puzzle = '...............5.....6.....' + (54 * '.')
    puzzle = merge(puzzle, previous)
    info1 = ((1, 2), (2, 2), (3, 3), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (8, 3))
    info2 = tuple((row, 10-column) for (row, column) in info1)
    sudoku = Sudoku()
    sudoku.solve(puzzle, features=[
        GermanSnakeFeature(info1, "Left"),
        GermanSnakeFeature(info2, "Right"),
        KnightsMoveFeature()
    ])


def puzzle5() -> None:
    previous = '..7......3.....5......................3..8............15.............9....9......'
    puzzle = '......3...1...............72.........................2..................8........'
    diadem = BoxOfNineFeature([(4, 2), (2, 1), (3, 3), (1, 4), (3, 5), (1, 6), (3, 7), (2, 9), (4, 8)])
    thermometers = [ThermometerFeature([(row, column) for row in (9, 8, 7, 6, 5, 4)], name=name)
                    for column in (2, 4, 6, 8)
                    for name in [f'Thermometer #{column // 2}']]
    features = [diadem, *thermometers]
    sudoku = Sudoku()
    sudoku.solve(merge(puzzle, previous), features=features)


def puzzle6() -> None:
    previous = '......................5....................6...1........2.................9......'
    puzzle = '......75.2.....4.9.....9.......2.8..5...............3........9...7......4........'
    snake_eggs = '3...777773.5.77...3.5...22...555.....4...8888.4.6.8..8.4.6.88...4.6...1....666...'
    Sudoku().solve(merge(puzzle, previous), features=[SnakesEggFeature(snake_eggs)])


def puzzle7() -> None:
    puzzles = [
        '925631847364578219718429365153964782249387156687215934472853691531796428896142573',   # Diary, Red
        '398541672517263894642987513865372941123894756974156238289435167456718329731629485',   # Ring, Purple
        '369248715152769438784531269843617952291854376675392184526973841438125697917486523',   # Locket, Orange(ish)
        '817325496396487521524691783741952638963148257285763149158279364632814975479536812',   # Cup, Yellow
        '527961384318742596694853217285619473473528169961437852152396748746285931839174625',   # Crown, Blue
        '196842753275361489384759126963125847548937261721684935612578394837496512459213678',   # Snake, Green
    ]
    pluses = [(1, 1), (1, 9), (2, 4), (2, 6), (3, 3), (3, 7), (4, 2), (4, 4), (4, 6), (4, 8), (5, 3)]
    pluses = pluses + [(10 - row, 10 - column) for row, column in pluses]
    puzzle = '......................7..1.....8.................6.....3..5......................'
    Sudoku().solve(puzzle, features=[(PlusFeature(pluses, puzzles))])


def puzzle8() -> None:
    puzzles = [
        '925631847364578219718429365153964782249387156687215934472853691531796428896142573',   # Diary, Red
        '398541672517263894642987513865372941123894756974156238289435167456718329731629485',   # Ring, Purple
        '369248715152769438784531269843617952291854376675392184526973841438125697917486523',   # Locket, Orange(ish)
        '817325496396487521524691783741952638963148257285763149158279364632814975479536812',   # Cup, Yellow
        '527961384318742596694853217285619473473528169961437852152396748746285931839174625',   # Crown, Blue
        '196842753275361489384759126963125847548937261721684935612578394837496512459213678',   # Snake, Green
        '213845679976123854548976213361782945859314762724569381632458197185297436497631528',   # Enigma, Gray
    ]

    grid = '+.y..o+.+...Gb.......p...r.+..+b.b+...........+g.g+..+.o...........ry...+.+g..g.+'
    pluses = [square for square, letter in zip(Feature.all_squares(), grid) if letter == '+']

    features = [
        # noinspection SpellCheckingInspection
        ColorFeature(grid, 'rpoybgG', puzzles),
        PlusFeature(pluses, puzzles),
        BoxOfNineFeature.major_diagonal(),
        BoxOfNineFeature.minor_diagonal(),
    ]
    Sudoku().solve('.'*81, features=features)


def main() -> None:
    puzzle1()
    puzzle2()
    puzzle3()
    puzzle4()
    puzzle5()
    puzzle6()
    puzzle7()
    puzzle8()


if __name__ == '__main__':
    main()
