import datetime
import itertools
from collections.abc import Sequence, Iterable
from typing import Optional, cast

from cell import Cell, House, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square
from features.chess_move import LittlePrincessFeature, KnightsMoveFeature, KingsMoveFeature, QueensMoveFeature
from features.features import MagicSquareFeature, AlternativeBoxesFeature, BoxOfNineFeature, \
    AdjacentRelationshipFeature, LimitedValuesFeature, XVFeature, AdjacentNotConsecutiveFeature, SimonSaysFeature, \
    OddsAndEvensFeature, ValuesAroundIntersectionFeature, RenbanFeature, KillerCageFeature, ExtremeEndpointsFeature
from features.possibilities_feature import GroupedPossibilitiesFeature, CombinedPossibilitiesFeature, \
    PossibilitiesFeature
from features.same_value_as_mate_feature import SameValueAsMateFeature
from features.sandwich_feature import SandwichFeature, SandwichXboxFeature
from features.thermometer import ThermometerFeature, SlowThermometerFeature
from grid import Grid
from human_sudoku import Sudoku
from features.skyscraper_feature import SkyscraperFeature


class Pieces44(Feature):
    """Eggs that contain the numbers 2-9, but no 1"""
    class Egg(House):
        def __init__(self, index: int, cells: Sequence[Cell]) -> None:
            super().__init__(House.Type.EGG, index, cells)

        def reset(self) -> None:
            super().reset()
            self.unknown_values = SmallIntSet(range(2, 10))
            Cell.remove_values_from_cells(self.cells, {1}, show=False)

    eggs: Sequence[list[Square]]

    def __init__(self, pattern: str) -> None:
        super().__init__()
        assert len(pattern) == 81
        info: Sequence[list[Square]] = [list() for _ in range(10)]
        for (row, column), letter in zip(itertools.product(range(1, 10), repeat=2), pattern):
            if '1' <= letter <= '7':
                info[int(letter)].append((row, column))
        for i in range(1, 8):
            assert len(info[i]) == 8
        self.eggs = info[1:8]

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        eggs = [self.Egg(i + 1, [grid.matrix[square] for square in self.eggs[i]]) for i in range(len(self.eggs))]
        grid.houses.extend(eggs)

    def draw(self, context: DrawContext) -> None:
        colors = ('lightcoral', "violet", "bisque", "lightgreen", "lightgray", "yellow", "skyblue")
        for color, squares in zip(colors, self.eggs):
            context.draw_rectangles(squares, facecolor=color)


class DrawCircleFeature(Feature):
    squares: Sequence[Square]

    def __init__(self, squares: Sequence[Square]):
        super().__init__()
        self.squares = squares

    def draw(self, context: DrawContext) -> None:
        for row, column in self.squares:
            context.draw_circle((column + .5, row + .5), radius=.5, fill=False, color='blue')


class DoubleSumFeature(GroupedPossibilitiesFeature):
    row_column: int
    htype: House.Type
    total: Optional[int]
    ptotal: int

    def __init__(self, htype: House.Type, row_column: int, ptotal: int, total: Optional[int] = None):
        name = f'DoubleSum {htype.name.title()} #{row_column}'
        squares = self.get_house_squares(htype, row_column)
        self.row_column = row_column
        self.htype = htype
        self.total = total
        self.ptotal = ptotal
        super().__init__(squares, name=name, compressed=True)

    def get_possibilities(self) -> Iterable[tuple[set[int], ...]]:
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
            super().initialize(grid)
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
    features = [BoxOfNineFeature.major_diagonal(), BoxOfNineFeature.minor_diagonal()]
    Sudoku().solve(puzzle, features=features, show=show)


def sandwich_07_28(*, show: bool = False) -> None:
    class LiarsSandwichFeature(SandwichFeature):
        def get_possibilities(self) -> Iterable[tuple[set[int], ...]]:
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
        def get_mates(self, cell: Cell) -> Iterable[Cell]:
            return self.neighbors_from_offsets(cell, offsets)

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
        BoxOfNineFeature.major_diagonal(),
        BoxOfNineFeature.minor_diagonal(),
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
    snake = BoxOfNineFeature(snake_squares, line=False)
    puzzle = ".....8....................9.................6.....4.................6.......7.9.."
    Sudoku().solve(puzzle, features=[*thermometers, snake], show=show)


def puzzle_09_03(*, show: bool = False) -> None:
    columns = (11, 0, 17, 6, 22, 0, 10, 35, 9)
    rows = (27, 3, 0, 16, 16, 19, 5, 13, 0)
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79}

    all_squares = cast(Iterable[Square], itertools.product(range(1, 10), range(1, 10)))
    prime_squares = [square for value, square in enumerate(all_squares, start=1) if value in primes]
    features = [
        *[SandwichFeature(House.Type.ROW, row, total) for row, total in enumerate(rows, start=1)],
        *[SandwichFeature(House.Type.COLUMN, col, total) for col, total in enumerate(columns, start=1)],
        LimitedValuesFeature(prime_squares, (2, 3, 5, 7)),
        DrawCircleFeature(prime_squares)
    ]
    Sudoku().solve('1' + ' ' * 80, features=features, show=show)


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

        def get_mates(self, cell: Cell) -> Iterable[Cell]:
            return self.neighbors_from_offsets(cell, self.OFFSETS)

        def draw(self, context: DrawContext) -> None:
            if self.done:
                self.draw_outline(context, [self.this_square], linestyle="-")

    all_squares = cast(Iterable[Square], itertools.product(range(1, 10), range(1, 10)))
    features = [CamelJumpFeature(square) for square in all_squares]
    puzzle = "........9.....85..7...2.1..35...............6.96.....7...........1.7.9......452.."
    Sudoku().solve(puzzle, features=features, show=show)


def puzzle_09_10(*, show: bool = False) -> None:
    puzzle = "....9.....14..5..8......3.4.2.3..74...6.......81.24....6.9..........69.28.9..1..3"
    Sudoku().solve(puzzle, features=(), show=show)


def puzzle_09_15(*, show: bool = False) -> None:
    puzzle = "-----.5.3.8.2.2.5.3.6.9.9.4.6.1.-".replace('-', '.........')
    feature = XVFeature.setup(
        down={5: [(1, 3), (1, 5), (1, 7), (2, 2), (2, 4), (2, 6), (2, 8), (3, 3), (3, 5), (3, 7)]},
        across={}
    )
    Sudoku().solve(puzzle, features=[feature], show=show)


def puzzle_09_16(*, show: bool = False) -> None:
    puzzle = "529784361............2......4....2..361529784..2....3......2............784361529"
    features = [BoxOfNineFeature.major_diagonal(), BoxOfNineFeature.minor_diagonal()]
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
    class Multiplication(GroupedPossibilitiesFeature):
        def __init__(self, row, column) -> None:
            squares = [(row, column), (row, column + 1), (row + 1, column), (row + 1, column + 1)]
            super().__init__(squares, name=f"Square{row}{column}", neighbors=True)

        def get_possibilities(self) -> list[tuple[set[int], ...]]:
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
    Sudoku().solve(puzzle, features=[AdjacentNotConsecutiveFeature()], show=show)


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


def puzzle_2021_07_06() -> tuple[str, Sequence[Feature]]:
    class Cheat (SimonSaysFeature):
        def round_1(self):
            Cell.remove_value_from_cells([self @ (7, 3)], 7)

        def round_2(self):
            (self @ (6, 4)).set_value_to(4)

    grid = "-.1.---.8.----.2.----------------".replace('-', '...')
    a, b, c, d, e, f, g, h, i, j = (1, 1), (3, 3), (5, 1), (5, 5), (5, 9), (7, 3), (7, 7), (9, 1), (9, 5), (9, 9)
    ends = [(a, b), (a, c), (b, c), (b, d), (c, d), (c, f), (c, h), (d, f), (d, i), (d, g), (e, g), (e, j),
            (f, h), (f, i), (g, i), (g, j), (h, i), (i, j)]
    features: list[Feature] = [ExtremeEndpointsFeature.between(x, y) for x, y in ends]
    features.append(Cheat())
    return grid, features


def puzzle_2021_07_10() -> tuple[str, Sequence[Feature]]:
    class MyHelperFeature(SimonSaysFeature):
        def round_1(self) -> None:
            Cell.remove_values_from_cells([self @ (3, 7)], {1, 2})
            Cell.keep_values_for_cell([self @ (6, 5), self @ (6, 8)], {1, 2})

    killers = [
        (7, "2,2,S,S"),
        (5, "2,6,E"),
        (6, "3,7,E"),
        (6, "4,6,E"),
        (23, "6,3,E,S"),
        (15, "6,9,S"),
        (17, "7,3,S"),
        (3, "9,6,E")]
    features: list[PossibilitiesFeature] = [KillerCageFeature(total, squares) for total, squares in killers]
    features[1:4] = [CombinedPossibilitiesFeature(features[1:4])]
    features.append(MyHelperFeature())
    return ' ' * 81, features


def puzzle_2021_07_11() -> tuple[str, Sequence[Feature]]:
    features = [
        OddsAndEvensFeature(evens=[(1, 4), (2, 2), (2, 9), (8, 1), (8, 1), (8, 8), (9, 6)]),
        BoxOfNineFeature("7,1,NE,E,NE,E,E,NE,E,NE"),
        BoxOfNineFeature("1,3,S,SE,SE,S,S,SE,SE,S"),
        ValuesAroundIntersectionFeature(top_left=(1, 1), values=(2, 3)),
        ValuesAroundIntersectionFeature(top_left=(1, 5), values=(5, 6)),
        ValuesAroundIntersectionFeature(top_left=(1, 8), values=(5, 6)),
        ValuesAroundIntersectionFeature(top_left=(4, 1), values=(2, 3)),
        ValuesAroundIntersectionFeature(top_left=(5, 8), values=(7, 8)),
        ValuesAroundIntersectionFeature(top_left=(8, 1), values=(4, 5)),
        ValuesAroundIntersectionFeature(top_left=(8, 4), values=(4, 5)),
        ValuesAroundIntersectionFeature(top_left=(8, 8), values=(7, 8)),
        RenbanFeature("1,5,E,E,S,S"),
        RenbanFeature("3,3,W,W,S,S"),
        RenbanFeature("7,3,S,S,E,E"),
        RenbanFeature("5,9,S,S,W,W")
    ]
    return '.' * 81, features


def main():
    start = datetime.datetime.now()
    grid, features = puzzle_2021_07_10()
    Sudoku().solve(grid, features=features, show=False, draw_verbose=True)
    end = datetime.datetime.now()
    print(end - start)


if __name__ == '__main__':
    main()
