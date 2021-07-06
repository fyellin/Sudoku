import datetime
import itertools
import math
from typing import Sequence, Tuple, List, Mapping, Iterable, Set, Optional, cast

from cell import Cell, House
from feature import Feature, Square
from features import KnightsMoveFeature, PossibilitiesFeature, MagicSquareFeature, \
    AdjacentRelationshipFeature, AllValuesPresentFeature, ThermometerFeature, SnakeFeature, LimitedValuesFeature, \
    SameValueAsExactlyOneMateFeature, SameValueAsMateFeature, LittlePrincessFeature, \
    AlternativeBoxesFeature, SlowThermometerFeature, SandwichFeature, KingsMoveFeature, \
    QueensMoveFeature, SandwichXboxFeature, PalindromeFeature, XVFeature, NonConsecutiveFeature
from grid import Grid
from human_sudoku import Sudoku
from draw_context import DrawContext
from skyscraper_feature import SkyscraperFeature


class MalvoloRingFeature(Feature):
    SQUARES = ((2, 4), (2, 5), (2, 6), (3, 7), (4, 8), (5, 8), (6, 8), (7, 7),
               (8, 6), (8, 5), (8, 4), (7, 3), (6, 2), (5, 2), (4, 2), (3, 3))

    class Adjacency(AdjacentRelationshipFeature):
        def __init__(self, squares: Sequence[Square]) -> None:
            super().__init__(squares, name="Malvolo Ring", cyclic=True)

        def match(self, digit1: int, digit2: int) -> bool:
            return digit1 + digit2 in (4, 8, 9, 16)

    features: Sequence[Feature]
    special: Cell

    def __init__(self) -> None:
        self.features = [self.Adjacency(self.SQUARES), AllValuesPresentFeature(self.SQUARES)]

    def initialize(self, grid: Grid) -> None:
        self.special = grid.matrix[2, 4]
        for feature in self.features:
            feature.initialize(grid)

    def reset(self, grid: Grid) -> None:
        for feature in self.features:
            feature.reset(grid)

    def check(self) -> bool:
        return any(feature.check() for feature in self.features)

    def check_special(self) -> bool:
        """A temporary hack that it's not worth writing the full logic for.  If we set this value to 4,
           then it will start a cascade such that no variable on the ring can have a value of 2. """
        if len(self.special.possible_values) == 2:
            print("Danger, danger")
            self.special.set_value_to(2)
            return True
        return False

    def draw(self, context: DrawContext) -> None:
        radius = math.hypot(2.5, 1.5)
        context.draw_circle((5.5, 5.5), radius=radius, fill=False, facecolor='black')


class GermanSnakeFeature(AdjacentRelationshipFeature):
    """A sequence of squares that must differ by 5 or more"""
    def __init__(self, name: str, snake:  Sequence[Square]):
        super().__init__(snake, name=name)
        self.snake = snake

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        print("No Fives in a German Snake")
        Cell.remove_values_from_cells(self.cells, {5}, show=False)

    def match(self, digit1: int, digit2: int) -> bool:
        return abs(digit1 - digit2) >= 5


class ContainsTextFeature(PossibilitiesFeature):
    text: Sequence[int]

    """A row that must contain certain digits consecutively"""
    def __init__(self, row: int, text: Sequence[int]) -> None:
        super().__init__([(row, column) for column in range(1, 10)], name=f'Text Row {row}')
        self.text = text

    def get_possibilities(self) -> Iterable[Tuple[Set[int], ...]]:
        unused_digits = {digit for digit in range(1, 10) if digit not in self.text}
        text_template = [{v} for v in self.text]
        template = [unused_digits] * len(unused_digits)
        for text_position in range(0, len(unused_digits) + 1):
            line = (*template[0:text_position], *text_template, *template[text_position:])
            yield line


class LimitedKnightsMove(KnightsMoveFeature):
    BAD_INDICES = {(row, column) for row in (4, 5, 6) for column in (4, 5, 6)}

    def get_neighbors(self, cell: Cell) -> Iterable[Cell]:
        index = cell.index
        if index in self.BAD_INDICES:
            return
        for neighbor in super().get_neighbors(cell):
            if neighbor.index not in self.BAD_INDICES:
                yield neighbor


class SnakesEggFeature(Feature):
    """A snake egg, where eggs of size n have all the digits from 1-n"""

    class Egg(House):
        def __init__(self, index: int, cells: Sequence[Cell]) -> None:
            super().__init__(House.Type.EGG, index, cells)

        def reset(self) -> None:
            super().reset()
            self.unknown_values = set(range(1, len(self.cells) + 1))
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
        eggs = [self.Egg(i, [grid.matrix[square] for square in self.squares[i]]) for i in range(1, 9)]
        grid.houses.extend(eggs)

    def draw(self, context: DrawContext) -> None:
        # Find all squares that aren't in one of the eggs.
        snake: Set[Square] = set(itertools.product(range(1, 10), range(1, 10)))
        snake.difference_update(cell for size in range(1, 10) for cell in self.squares[size])
        context.draw_rectangles(cast(Sequence[Square], snake), facecolor='lightblue')


class Pieces44(Feature):
    """Eggs that contain the numbers 2-9, but no 1"""
    class Egg(House):
        def __init__(self, index: int, cells: Sequence[Cell]) -> None:
            super().__init__(House.Type.EGG, index, cells)

        def reset(self) -> None:
            super().reset()
            self.unknown_values = set(range(2, 10))
            Cell.remove_values_from_cells(self.cells, {1}, show=False)

    eggs: Sequence[List[Square]]

    def __init__(self, pattern: str) -> None:
        super().__init__()
        assert len(pattern) == 81
        info: Sequence[List[Square]] = [list() for _ in range(10)]
        for (row, column), letter in zip(itertools.product(range(1, 10), repeat=2), pattern):
            if '1' <= letter <= '7':
                info[int(letter)].append((row, column))
        for i in range(1, 8):
            assert len(info[i]) == 8
        self.eggs = info[1:8]

    def initialize(self, grid: Grid) -> None:
        eggs = [self.Egg(i + 1, [grid.matrix[square] for square in self.eggs[i]]) for i in range(len(self.eggs))]
        grid.houses.extend(eggs)

    def draw(self, context: DrawContext) -> None:
        colors = ('lightcoral', "violet", "bisque", "lightgreen", "lightgray", "yellow", "skyblue")
        for color, squares in zip(colors, self.eggs):
            context.draw_rectangles(squares, facecolor=color)


class PlusFeature(Feature):
    squares: Sequence[Square]
    puzzles: Sequence[str]

    def __init__(self, squares: Sequence[Square], puzzles: Sequence[str]) -> None:
        self.squares = squares
        self.puzzles = puzzles

    def reset(self, grid: Grid) -> None:
        for row, column in self.squares:
            value = self.__get_value(row, column)
            grid.matrix[row, column].set_value_to(value)

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

    def reset(self, grid: Grid) -> None:
        self.plus_feature.reset(grid)
        for (row, column), letter in self.setup.items():
            puzzle = self.color_map[letter]
            index = (row - 1) * 9 + (column - 1)
            value = int(puzzle[index])
            grid.matrix[row, column].set_value_to(value)

    CIRCLES = dict(r="lightcoral", p="violet", o="bisque", g="lightgreen", G="lightgray", y="yellow", b="skyblue")

    def draw(self, context: DrawContext) -> None:
        self.plus_feature.draw(context)
        for (row, column), letter in self.setup.items():
            context.draw_circle((column + .5, row + .5), radius=.4, fill=True, color=self.CIRCLES[letter])
        # noinspection PyTypeChecker
        context.add_fancy_bbox((2.3, 5.3), width=6.4, height=0.4, boxstyle='round, pad=0.2', fill=False)


class DrawCircleFeature(Feature):
    squares: Sequence[Square]

    def __init__(self, squares: Sequence[Square]):
        self.squares = squares

    def draw(self, context: DrawContext) -> None:
        for row, column in self.squares:
            context.draw_circle((column + .5, row + .5), radius=.5, fill=False, color='blue')


class DoubleSumFeature(PossibilitiesFeature):
    row_column: int
    htype: House.Type
    total: Optional[int]
    ptotal: int

    def __init__(self, htype: House.Type, row_column: int, ptotal: int, total: Optional[int] = None):
        name = f'DoubleSum {htype.name.title()} #{row_column}'
        squares = self.get_row_or_column(htype, row_column)
        self.row_column = row_column
        self.htype = htype
        self.total = total
        self.ptotal = ptotal
        super().__init__(squares, name=name, compressed=True)

    def get_possibilities(self) -> Iterable[Tuple[Set[int], ...]]:
        total = self.total
        ptotal = self.ptotal
        for item1, item2 in itertools.permutations(range(1, 10), 2):
            if total and item1 + item2 != total:
                continue
            item3_possibilities = [item1] if item1 == 1 else [item2] if item1 == 2 \
                else [x for x in range(1, 10) if x not in {item1, item2}]
            for item3 in item3_possibilities:
                item4_possibilities = [item1] if item2 == 1 else [item2] if item2 == 2 \
                    else [x for x in range(1, 10) if x not in {item1, item2, item3}]
                item4 = ptotal - item3
                if item4 not in item4_possibilities:
                    continue
                other_values = set(range(1, 10)) - {item1, item2, item3, item4}
                temp = [{item1}, {item2}] + [other_values] * 7
                temp[item1 - 1] = {item3}
                temp[item2 - 1] = {item4}
                yield tuple(temp)

    def draw(self, context: DrawContext) -> None:
        args = {'fontsize': '15'}
        if self.total:
            self.draw_outside(context, f'{self.total}', self.htype, self.row_column, padding=.6, **args)
        self.draw_outside(context, f'{self.ptotal}', self.htype, self.row_column, **args)


class DrawBoxFeature(Feature):
    squares: Sequence[Square]

    def __init__(self, squares: Sequence[Square]):
        self.squares = squares

    def draw(self, context: DrawContext) -> None:
        self.draw_outline(context, self.squares)


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
    sudoku.solve(puzzle, features=[MalvoloRingFeature()])


def puzzle3() -> None:
    previous = '....4........6.............8.....9..........6.........2..........................'
    puzzle = '..9...7...5.....3.7.4.....9.............5.............5.....8.1.3.....9...7...5..'

    evens = [(2, 3), (3, 2), (3, 3), (1, 4), (1, 5), (1, 6)]
    evens = evens + [(column, 10 - row) for row, column in evens]
    evens = evens + [(10 - row, 10 - column) for row, column in evens]

    features = [SnakeFeature([(3, 6), (3, 5), (4, 4), (5, 4), (5, 5), (5, 6), (6, 6), (7, 5), (7, 4)]),
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
        GermanSnakeFeature("Left", info1),
        GermanSnakeFeature("Right", info2), KnightsMoveFeature()
    ])


def puzzle5() -> None:
    previous = '..7......3.....5......................3..8............15.............9....9......'
    puzzle = '......3...1...............72.........................2..................8........'
    diadem = SnakeFeature([(4, 2), (2, 1), (3, 3), (1, 4), (3, 5), (1, 6), (3, 7), (2, 9), (4, 8)])
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
    features = [
        # noinspection SpellCheckingInspection
        ColorFeature(grid, 'rpoybgG', puzzles),
        SnakeFeature.major_diagonal(),
        SnakeFeature.minor_diagonal(),
    ]
    Sudoku().solve('.'*81, features=features)


def magic_squares() -> None:
    puzzle = ('.' * 17) + "1" + ('.' * 54) + '.6.......'
    features = [
        MagicSquareFeature((2, 6)),

        MagicSquareFeature((4, 2)),
        MagicSquareFeature((6, 8)),
        MagicSquareFeature((8, 4)),
    ]
    sudoku = Sudoku()
    sudoku.solve(puzzle, features=features)


def run_thermometer() -> None:
    thermometers = [[(2, 2), (1, 3), (1, 4), (1, 5), (2, 6)],
                    [(2, 2), (3, 1), (4, 1), (5, 1), (6, 2)],
                    [(2, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8)],
                    [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (6, 8)],
                    [(3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 7)],
                    [(4, 2), (5, 2), (6, 3), (7, 4), (8, 4)],
                    [(1, 7), (1, 8)],
                    [(8, 8), (8, 9)],
                    [(8, 2), (8, 3)]]
    puzzle = ' ' * 81
    features = [ThermometerFeature(thermometer) for thermometer in thermometers]
    sudoku = Sudoku()
    sudoku.solve(puzzle, features=features)
    sudoku.draw_grid()


def thermometer_magic() -> None:
    thermometers = [
        [(6 - r, 1) for r in range(1, 6)],
        [(6 - r, r) for r in range(1, 6)],
        [(1, 10 - r) for r in range(1, 6)],
        [(10 - r, 9) for r in range(1, 6)],
        [(10 - r, 4 + r) for r in range(1, 6)],
        [(9, 6 - r) for r in range(1, 6)]]
    features = [
        MagicSquareFeature(dr=4, dc=4, color='lightblue'),
        *[ThermometerFeature(squares) for squares in thermometers]
    ]
    puzzle = ("." * 18) + '.....2...' + ('.' * 27) + '...8.....' + ('.' * 18)
    Sudoku().solve(puzzle, features=features)


def you_tuber() -> None:
    puzzle = '...........12986...3.....7..8.....2..1.....6..7.....4..9.....8...54823..........'
    features = [
        LimitedKnightsMove(),
        *[SameValueAsExactlyOneMateFeature((row, column)) for row in (4, 5, 6) for column in (4, 5, 6)]
    ]
    sudoku = Sudoku()
    sudoku.solve(puzzle, features=features)


def little_princess() -> None:
    puzzle = '.......6...8..........27......6.8.1....4..........9..............7...............'
    Sudoku().solve(puzzle, features=[LittlePrincessFeature()])


def puzzle44() -> None:
    puzzle = "........8...........7............2................9....................5....36..."
    pieces = '1112.333.1.2223.33122.2233.111....44.5.64444..566.44..55.6677775556..77..566...77'
    Sudoku().solve(puzzle, features=[KnightsMoveFeature(), Pieces44(pieces)])


def puzzle_alice(*, show: bool = False) -> None:
    # puzzle = "......... 3......8. ..4...... ......... 2...9...7 ......... ......5.. .1......6 ........."
    puzzle = "......... 3....6.8. ..4...... ......... 2...9...7 ......... ......5.. .1......6 ........."  # 18:30

    pieces = "122222939112122333911123333441153666445555696497758966447958886447559886777778889"
    features = [AlternativeBoxesFeature(pieces),
                *(SameValueAsMateFeature((r, c)) for r in range(1, 10) for c in range(1, 10))
                ]
    puzzle = puzzle.replace(' ', '')
    Sudoku().solve(puzzle, features=features, show=show)


def slow_thermometer_puzzle1() -> None:
    puzzle = '.' * 81
    thermometers = [
        [(4, 5), (5, 5), (6, 6), (5, 6), (4, 6), (3, 7), (2, 7), (1, 6), (1, 5), (1, 4), (2, 3), (3, 3), (4, 4)],
        [(4, 5), (5, 5), (6, 6), (5, 7), (4, 7), (3, 8), (2, 8)],
        [(2, 2), (2, 1), (1, 1), (1, 2)],
        [(1, 7), (1, 8), (2, 9), (3, 9), (4, 8), (5, 8)],
        [(6, 4),  (5, 4),  (4, 3), (3, 2)],
        [(5, 3), (4, 2), (4, 1), (5, 2), (5, 1), (6, 1)],
        [(6, 8), (6, 9), (5, 9), (4, 9)],
        [(8, 4), (9, 3), (8, 2), (8, 3), (7, 4), (6, 3), (7, 3), (7, 2)],
        [(7, 6), (7, 7), (7, 8), (7, 9), (8, 8), (9, 8), (9, 7), (8, 6), (7, 5)]
    ]
    thermometers = [SlowThermometerFeature(thermometer) for thermometer in thermometers]
    Sudoku().solve(puzzle, features=thermometers)


def slow_thermometer_puzzle2() -> None:
    puzzle = '.' * 72 + ".....1..."
    thermos = [
        "2,4,N,W,S,S,E,SE",
        "2,7,N,W,S",
        "4,6,N,NW",
        "4,7,N,SE,SE",
        "4,2,SW,E,SW,E,SW,E,SW,E,SW",
        "5,4,SE,E",
        "6,4,E,E",
        "7,3,S,S",
        "9,5,NW,S",
        "9,6,N",
        "9,6,NW",
        "6,7,E,SW,W,W,W,NW",
        "6,9,W,SW,W,W,W,NW",
        "8,8,NW",
        "8,8,W,SE,W"
    ]
    thermometers = [SlowThermometerFeature(line, color='lightblue') for line in thermos]
    Sudoku().solve(puzzle, features=thermometers)


def thermometer_07_23() -> None:
    puzzle = ".....................9.............5...............3.................8.......9..."
    thermos = [
        "1,1,SE,SE,SE,SW,SW",
        "1,9,SW,SW,SW,NW,NW",
        "9,1,NE,NE,NE,SE,SE",
        "9,9,NW,NW,NW,NE,NE"
    ]
    thermometers = [ThermometerFeature(line, color='lightgray') for line in thermos]
    Sudoku().solve(puzzle, features=thermometers)


def double_sum_puzzle(*, show: bool = False) -> None:

    class CheckSpecialFeature(Feature):
        cells: Sequence[Cell]

        def initialize(self, grid: Grid) -> None:
            self.cells = [grid.matrix[1, 6], grid.matrix[2, 6]]

        def check_special(self) -> bool:
            if len(self.cells[0].possible_values) == 4:
                print("Danger.  Danger")
                Cell.keep_values_for_cell(self.cells, {3, 7})
                return True
            return False

    features = [
        DoubleSumFeature(House.Type.ROW, 1, 6),
        DoubleSumFeature(House.Type.ROW, 4, 10, 10),
        DoubleSumFeature(House.Type.ROW, 5, 10, 9),
        DoubleSumFeature(House.Type.ROW, 6, 10, 10),
        DoubleSumFeature(House.Type.ROW, 7, 10, 10),
        DoubleSumFeature(House.Type.ROW, 9, 9, 11),

        DoubleSumFeature(House.Type.COLUMN, 1, 16),
        DoubleSumFeature(House.Type.COLUMN, 3, 13, 13),
        DoubleSumFeature(House.Type.COLUMN, 4, 12, 11),
        DoubleSumFeature(House.Type.COLUMN, 5, 9),
        DoubleSumFeature(House.Type.COLUMN, 6, 10, 10),
        DoubleSumFeature(House.Type.COLUMN, 7, 11, 15),
        DoubleSumFeature(House.Type.COLUMN, 8, 11, 9),

        CheckSpecialFeature(),
    ]
    Sudoku().solve('.' * 81, features=features, show=show)


def puzzle_hunt(*, show: bool = False) -> None:
    puzzle = "...48...7.8.5..6...9.....3.4...2.3..1...5...2..8..7......8.3.7...5...1.39...15.4."
    features = [SnakeFeature.major_diagonal(), SnakeFeature.minor_diagonal()]
    Sudoku().solve(puzzle, features=features, show=show)


def sandwich_07_28(*, show: bool = False) -> None:
    class LiarsSandwichFeature(SandwichFeature):
        def get_possibilities(self) -> Iterable[Tuple[Set[int], ...]]:
            yield from self._get_possibilities(self.total - 1)
            yield from self._get_possibilities(self.total + 1)

    puzzle = "..6................1...........1.....4.........9...2.....................7......8"
    features = [
        *[LiarsSandwichFeature(House.Type.ROW, row, total)
          for row, total in enumerate((5, 8, 5, 16, 12, 7, 5, 3, 1), start=1)],
        LiarsSandwichFeature(House.Type.COLUMN, 1, 5),
        LiarsSandwichFeature(House.Type.COLUMN, 5, 4),
        LiarsSandwichFeature(House.Type.COLUMN, 9, 5),
    ]
    Sudoku().solve(puzzle, features=features, show=show)


def skyscraper_07_29(*, show: bool = False) -> None:
    basement = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)]
    basement = basement + [(row, 10 - column) for row, column in basement]
    basement = basement + [(10 - row, column) for row, column in basement]

    features = [
        SkyscraperFeature(House.Type.ROW, 2, 2, None, basement=basement),
        SkyscraperFeature(House.Type.ROW, 3, None, 2, basement=basement),
        SkyscraperFeature(House.Type.ROW, 5, None, 5, basement=basement),
        SkyscraperFeature(House.Type.ROW, 6, 5, None, basement=basement),
        SkyscraperFeature(House.Type.ROW, 8, None, 5, basement=basement),
        SkyscraperFeature(House.Type.ROW, 9, 2, 2, basement=basement),

        SkyscraperFeature(House.Type.COLUMN, 2, 5, None, basement=basement),
        SkyscraperFeature(House.Type.COLUMN, 3, 2, None, basement=basement),
        SkyscraperFeature(House.Type.COLUMN, 4, 2, None, basement=basement),
        SkyscraperFeature(House.Type.COLUMN, 5, 5, 5, basement=basement),
        SkyscraperFeature(House.Type.COLUMN, 6, 2, 5, basement=basement),
        SkyscraperFeature(House.Type.COLUMN, 8, 5, None, basement=basement),
        SkyscraperFeature(House.Type.COLUMN, 9, 2, 2, basement=basement),
    ]

    Sudoku().solve('.' * 81, features=features, show=show)


def puzzle_07_30(*, show: bool = False) -> None:
    features = [
        SandwichXboxFeature(House.Type.ROW, 3, 16),
        SandwichXboxFeature(House.Type.ROW, 4, 10, right=True),
        SandwichXboxFeature(House.Type.COLUMN, 3, 30),
        SandwichXboxFeature(House.Type.COLUMN, 4, 3),
        SandwichXboxFeature(House.Type.COLUMN, 7, 17),
        KingsMoveFeature(),
        QueensMoveFeature(),
    ]
    puzzle = "." * 63 + '.5.......' + '.' * 9
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_07_30_simon(*, show: bool = False) -> None:
    thermos = [
        "2,1,NE,S,NE,S,NE",
        "2,4,NE,S,NE,S,NE",
        "2,7,NE,S",
        "4,3,W,S,E,E,N",
        "4,7,W,S,E,E,N",
        "7,5,E,N,NW",
        "8,3,S,E,E,E",
        "9,8,N,E,N"
    ]
    thermometers = [ThermometerFeature(line, color='lightblue') for line in thermos]
    nada = "........."
    puzzle = nada + "......3.." + nada * 6 + ".......3."
    Sudoku().solve(puzzle, features=thermometers, show=show)


def puzzle_08_02(*, show: bool = False) -> None:
    thermos = [
        "2,1,SE,SE,NE,N,NE",
        "2,9,SW,SW,NW,N,NW",
        "6,1,N,N,N",
        "3,9,S,S,S",
        "8,1,E,E",
        "8,7,E,E",
        "5,6,SW,NW"
    ]
    features = [
        MagicSquareFeature(),
        KingsMoveFeature(),
        *[ThermometerFeature(line) for line in thermos],
    ]
    Sudoku().solve('.' * 81, features=features, show=show)


def puzzle_08_06(*, show: bool = False) -> None:
    offsets1 = [(dr, dc) for dx in (-1, 1) for dy in (-2, 2) for (dr, dc) in ((dx, dy), (dy, dx))]
    offsets2 = [(dr, dc) for delta in range(1, 9) for dr in (-delta, delta) for dc in (-delta, delta)]
    offsets = offsets1 + offsets2

    class MyFeature(SameValueAsMateFeature):
        def get_mates(self, cell: Cell, grid: Grid) -> Iterable[Cell]:
            return self.neighbors_from_offsets(grid, cell, offsets)

    features = [MyFeature((i, j)) for i, j in itertools.product(range(1, 10), repeat=2)]
    puzzle = "39.1...822.....5.....4.....6..2.....1....4.........3............6...3..551.....64"
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_08_07(*, show: bool = False) -> None:
    thermos = [
        "1,1,S,S,S",
        "9,1,N,N,N",
        "1,4,E,E,E,E,E,S",
        "2,4,E,E,E",
        "5,5,E,E,E,E",
        "6,5,SW",
        "9,4,E,E,E,E,E"
    ]
    features = [
        SnakeFeature.major_diagonal(),
        SnakeFeature.minor_diagonal(),
        *[ThermometerFeature(line) for line in thermos],
        # FoobarFeature()
    ]
    Sudoku().solve('.'*81, features=features, show=show)


def puzzle_08_12(*, show: bool = False) -> None:
    thermos = [
        "5,5,nw,nw,n,ne",
        "5,5,nw,nw,sw,sw,s",
        "5,5,ne,ne,n,nw",
        "5,5,ne,ne,se,se,s",
        "5,5,s,s,sw,w,nw",
        "5,5,s,s,se,e,ne",
        "1,1,e", "1,1,s",
        "2,9,n,w",
        "8,1,s,e",
        "9,9,w", "9,9,n",
        "4,3,ne",
        "3,5,e,se,s",
        "6,7,sw,w,w",
        "6,3,n"
    ]
    thermometers = [ThermometerFeature(line) for line in thermos]
    puzzle = "." * 63 + "....8...." + "." * 9
    Sudoku().solve(puzzle, features=thermometers, show=show)


def puzzle_08_15(*, show: bool = False) -> None:
    puzzle = "....1...4........5.............................1.....8........75.3....6.....3...."
    odds = [(3, 2), (3, 4), (3, 6), (3, 7), (3, 8),
            (4, 1), (4, 2), (4, 4), (4, 8),
            (5, 2), (5, 4), (5, 5), (5, 6), (5, 8),
            (6, 2), (6, 5), (6, 8),
            (7, 2), (7, 5), (7, 8)]
    features = [KingsMoveFeature(),
                LimitedValuesFeature(odds, (1, 3, 5, 7, 9), color='lightgray')
                ]
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_08_26(*, show: bool = False) -> None:
    class PrimeRing(AdjacentRelationshipFeature):
        def __init__(self, squares):
            super().__init__(squares, name="Prime", cyclic=True, color='red')

        def match(self, digit1: int, digit2: int) -> bool:
            return digit1 + digit2 in {2, 3, 5, 7, 11, 13, 17}

    columns = (21, 25, 11, 0, 35, 23, 13, 4, 18)
    rows = (13, 13, 6, 9, 0, 29, 2, 13, 2)
    features = [
        ThermometerFeature("3,7,S,S,S,S"),
        PrimeRing("2,2,E,E,S,E,E,N,E,E,S,S,W,S,S,E,S,S,W,W,N,W,W,S,W,W,N,N,E,N,N,W,N"),
        *[SandwichFeature(House.Type.ROW, row, total) for row, total in enumerate(rows, start=1)],
        *[SandwichFeature(House.Type.COLUMN, col, total) for col, total in enumerate(columns, start=1)],
    ]
    Sudoku().solve(' ' * 81, features=features, show=show)


def puzzle_08_31(*, show: bool = False) -> None:
    thermos = ["1,5,SW,SW,E,S",
               "1,8,W,W",
               "3,8,SW,S,SE,E",
               "7,3,NW,NW",
               "9,1,E,E",
               "9,8,NW,SW,NW,N,N,N,N"
               ]
    thermometers = [ThermometerFeature(line)
                    for i, line in enumerate(thermos, start=1)]
    snake_squares = [thermometer.squares[0] for thermometer in thermometers]
    snake_squares.extend(((2, 2), (4, 1), (7, 2)))
    snake = SnakeFeature(snake_squares, line=False)
    puzzle = ".....8....................9.................6.....4.................6.......7.9.."
    Sudoku().solve(puzzle, features=[*thermometers, snake], show=show)


def puzzle_09_03(*, show: bool = False) -> None:
    columns = (11, 0, 17, 6, 22, 0, 10, 35, 9)
    rows = (27, 3, 0, 16, 16, 19, 5, 13, 0)
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79}
    prime_squares = [square for value, square in enumerate(itertools.product(range(1, 10), range(1, 10)), start=1)
                     if value in primes]
    features = [
        *[SandwichFeature(House.Type.ROW, row, total) for row, total in enumerate(rows, start=1)],
        *[SandwichFeature(House.Type.COLUMN, col, total) for col, total in enumerate(columns, start=1)],
        LimitedValuesFeature(prime_squares, (2, 3, 5, 7)),
        DrawCircleFeature(prime_squares)
    ]
    Sudoku().solve('1' + ' ' * 80, features=features, show=show)


def puzzle_09_04(*, show: bool = False) -> None:
    class IncreasingFeature(AdjacentRelationshipFeature):
        def __init__(self, square1: Square, square2: Square):
            super().__init__([square1, square2], name="<", color='purple')

        def match(self, digit1: int, digit2: int) -> bool:
            return digit2 == digit1 + 1

    features = [
        PalindromeFeature("1,2,E,E,S,S", color="blue"),
        PalindromeFeature("1,8,W,W,S,S", color="blue"),
        PalindromeFeature("1,5,S,S,S,W,W,W", color="red"),
        PalindromeFeature("2,7,S,S,E,E", color="red"),
        PalindromeFeature("7,1,E,E,N,N", color="blue"),
        PalindromeFeature("7,5,E,E,S,S", color="blue"),
        PalindromeFeature("7,2,E,E,S,S", color="red"),
        PalindromeFeature("7,8,W,W,S,S", color="red"),
        IncreasingFeature((7, 2), (7, 3)),
        IncreasingFeature((7, 6), (7, 7))
    ]
    puzzle = "8....6..9..........3......42...............8.............5......1...............7"
    # noinspection SpellCheckingInspection
    extras = "XXXXXXX--..6X".replace("X", "---").replace("-", "...")
    puzzle = merge(puzzle, extras)
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_09_05(*, show: bool = False) -> None:
    class DeltaFeature(AdjacentRelationshipFeature):
        delta: int

        def __init__(self, square: Square, delta: int, is_right: bool):
            row, column = square
            square2 = (row, column + 1) if is_right else (row + 1, column)
            self.delta = delta
            super().__init__([square, square2], name="d")

        def draw(self, context: DrawContext) -> None:
            (r1, c1), (r2, c2) = self.squares
            context.draw_text((c1 + c2 + 1)/2, (r1 + r2 + 1)/2, str(self.delta),
                              verticalalignment='center', horizontalalignment='center',
                              fontsize=15, weight='bold', color='red')

        def match(self, digit1: int, digit2: int) -> bool:
            return abs(digit1 - digit2) == self.delta

    features = [
        DeltaFeature((2, 3), 1, True),
        DeltaFeature((2, 6), 1, True),
        DeltaFeature((3, 3), 7, True),
        DeltaFeature((3, 6), 7, True),
        DeltaFeature((4, 3), 5, True),
        DeltaFeature((4, 6), 5, True),
        DeltaFeature((6, 2), 1, True),
        DeltaFeature((6, 7), 2, True),
        DeltaFeature((1, 5), 8, False),
        DeltaFeature((2, 1), 4, False),
        DeltaFeature((2, 9), 4, False),
        DeltaFeature((3, 2), 6, False),
        DeltaFeature((3, 8), 6, False),
        DeltaFeature((4, 5), 3, False),
        DeltaFeature((7, 1), 1, False),
        DeltaFeature((7, 9), 2, False),
        DeltaFeature((8, 2), 1, False),
        DeltaFeature((8, 8), 2, False),

    ]
    # noinspection SpellCheckingInspection
    puzzle = "XXXXXX.921.738.-3.9-X".replace("X", "---").replace("-", "...")
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_09_06(*, show: bool = False) -> None:
    class CamelJumpFeature(SameValueAsMateFeature):
        OFFSETS = [(dr, dc) for dx in (-1, 1) for dy in (-3, 3) for (dr, dc) in ((dx, dy), (dy, dx))]

        def get_mates(self, cell: Cell, grid: Grid) -> Iterable[Cell]:
            return self.neighbors_from_offsets(grid, cell, self.OFFSETS)

        def draw(self, context: DrawContext) -> None:
            if self.done:
                self.draw_outline(context, [self.this_square], linestyle="-")

    features = [CamelJumpFeature(square) for square in itertools.product(range(1, 10), range(1, 10))]
    puzzle = "........9.....85..7...2.1..35...............6.96.....7...........1.7.9......452.."
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_09_10(*, show: bool = False) -> None:
    puzzle = "....9.....14..5..8......3.4.2.3..74...6.......81.24....6.9..........69.28.9..1..3"
    Sudoku().solve(puzzle, features=(), show=show)


def puzzle_09_15(*, show: bool = False) -> None:
    puzzle = "-----.5.3.8.2.2.5.3.6.9.9.4.6.1.-".replace('-', '.........')
    features = XVFeature.setup(
        down={5: [(1, 3), (1, 5), (1, 7), (2, 2), (2, 4), (2, 6), (2, 8), (3, 3), (3, 5), (3, 7)]},
        across={}
    )
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_09_16(*, show: bool = False) -> None:
    puzzle = "529784361............2......4....2..361529784..2....3......2............784361529"
    features = [SnakeFeature.major_diagonal(), SnakeFeature.minor_diagonal()]
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_09_17_2(*, show: bool = False) -> None:
    puzzle = "..8..964.9..1.6........7....3.....5.7.2...4.3.6.....8...18..9.....4....5.832..1.."
    Sudoku().solve(puzzle, features=(), show=show)


def puzzle_09_20(*, show: bool = False) -> None:
    # noinspection SpellCheckingInspection
    puzzle = "XXXXX-1..-XXX".replace("X", "---").replace("-", "...")
    features = [
        *SandwichFeature.all(House.Type.ROW, [10, 19, 25, 28, 17, 3, 23, 6, 7]),
        *SandwichFeature.all(House.Type.COLUMN, [18, 8, 21, 18, 13, 27, 25, 13, 3]),
        KnightsMoveFeature()
    ]
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_09_21(*, show: bool = False) -> None:
    class Multiplication(PossibilitiesFeature):
        def __init__(self, row, column) -> None:
            squares = [(row, column), (row, column + 1), (row + 1, column), (row + 1, column + 1)]
            super().__init__(squares, name=f"Square{row}{column}", neighbors=True)

        def get_possibilities(self) -> List[Tuple[Set[int], ...]]:
            for x, y in itertools.product(range(1, 10), repeat=2):
                if x <= y:
                    z = x * y
                    if z >= 11 and z % 10 != 0:
                        yield [{x, y}, {x, y}, {z // 10}, {z % 10}]

        def draw(self, context: DrawContext) -> None:
            context.draw_rectangles(self.squares, color='lightgray')

    puzzle = "X..7-6...5.-.8.-..9-X-5..-.6.-.9...1-2..X".replace("X", "---").replace("-", "...")
    features = [
        Multiplication(1, 1), Multiplication(1, 8), Multiplication(3, 3), Multiplication(6, 6),
        Multiplication(8, 1), Multiplication(8, 8)
    ]
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_10_17(*, show: bool = False) -> None:
    # noinspection SpellCheckingInspection
    puzzle = "XXXXX3.9.4.1.6.9.4.5.3.8.7.6.5.4X".replace("X", "---").replace("-", "...")
    features = NonConsecutiveFeature.setup()
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_2021_01_21(*, show: bool = False) -> None:
    puzzle = "1..4..6...2..5..9...3..6..87..1..9...9..2..4...4..3..55..8..2...8..7..3...7..9..1"
    features = ()
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_2021_03_15(*, show: bool = False) -> None:
    puzzle = "...........3.1.4..1..5.9.......2.65.35.....89.79.3.......2.3..8..4.6.2..........."
    features = ()
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_2021_05_12(*, show: bool = False) -> None:
    puzzle = "4.......2..5.829...2.....3...8.1....56..9..78....6.5...1.....6...615.7..3.......4"
    features = ()
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_2021_05_24(*, show: bool = False) -> None:
    lines = ["1,1,E", "1,3,SW", "1,6,S", "1,7,E", "2,3,NE", "2,5,N", "2,9,N", "3,1,N", "3,3,NE", "3,4,SE", "3,5,SE",
             "3,6,NE", "3,7,SE", "3,8,SE", "3,9,NW", "4,1,E", "4,3,NW", "4,7,SW", "5,1,E", "5,3,NE", "5,7,SW", "5,9,W",
             "6,3,NE", "6,4,NE", "6,7,SE", "6,9,W", "7,1,SE", "7,2,NW", "7,3,NW", "7,4,SW", "7,6,W", "7,7,SW", "7,9,S",
             "8,1,S", "8,7,SW", "9,3,W", "9,4,N", "9,5,N", "9,7,NE", "9,9,W"]
    thermometers = [ThermometerFeature(line) for line in lines]
    Sudoku().solve(' ' * 81, features=thermometers, show=show)


if __name__ == '__main__':
    start = datetime.datetime.now()
    puzzle8()
    end = datetime.datetime.now()
    print(end - start)
