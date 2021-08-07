import datetime
import itertools
from collections.abc import Iterable, Sequence
from typing import Optional, cast

from cell import Cell, House, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from features.chess_move import KingsMoveFeature, KnightsMoveFeature, LittlePrincessFeature, QueensMoveFeature
from features.features import AdjacentNotConsecutiveFeature, AdjacentRelationshipFeature, AlternativeBoxesFeature, \
    ArithmeticFeature, ArrowSumFeature, BoxOfNineFeature, DrawOnlyFeature, ExtremeEndpointsFeature, \
    KillerCageFeature, \
    LimitedValuesFeature, MagicSquareFeature, PalindromeFeature, RenbanFeature, ValuesAroundIntersectionFeature, \
    XVFeature
from features.possibilities_feature import HousePossibilitiesFeature, PossibilitiesFeature
from features.same_value_as_mate_feature import SameValueAsMateFeature
from features.sandwich_feature import SandwichFeature, SandwichXboxFeature
from features.skyscraper_feature import SkyscraperFeature
from features.thermometer import SlowThermometerFeature, ThermometerAsLessThanFeature, ThermometerFeature
from grid import Grid
from human_sudoku import Sudoku

BLANK_GRID = '.' * 81


class Pieces44(Feature):
    """Eggs that contain the numbers 2-9, but no 1"""
    class Egg(House):
        def __init__(self, index: int, cells: Sequence[Cell]) -> None:
            super().__init__(House.Type.EGG, index, cells)

        def start(self) -> None:
            super().start()
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


class DoubleSumFeature(HousePossibilitiesFeature):
    """The first two items in the row/column are indices, and the values they point at must total to ptotal.
    If total is also given, then the first two values must sum to this"""
    row_column: int
    htype: House.Type
    total: Optional[int]
    ptotal: int

    def __init__(self, htype: House.Type, row_column: int, ptotal: int, total: Optional[int] = None):
        super().__init__(htype, row_column, prefix="DoubleSum")
        self.total = total
        self.ptotal = ptotal

    def match(self, permutation: tuple[int, ...]) -> bool:
        a, b = permutation[0], permutation[1]
        return (self.total is None or a + b == self.total) and \
            permutation[a - 1] + permutation[b - 1] == self.ptotal

    def draw(self, context: DrawContext) -> None:
        args = {'fontsize': '10'}
        if self.total:
            context.draw_outside(f'{self.total}', self.htype, self.index, padding=.2,
                                 color='red', **args)
        context.draw_outside(f'{self.ptotal}', self.htype, self.index, **args)


def thermometer_magic() -> tuple[str, Sequence[Feature]]:
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
    return puzzle, features


def little_princess() -> tuple[str, Sequence[Feature]]:
    puzzle = '.......6...8..........27......6.8.1....4..........9..............7...............'
    return puzzle, [LittlePrincessFeature()]


def puzzle44() -> tuple[str, Sequence[Feature]]:
    puzzle = "........8...........7............2................9....................5....36..."
    pieces = '1112.333.1.2223.33122.2233.111....44.5.64444..566.44..55.6677775556..77..566...77'
    return puzzle, [KnightsMoveFeature(), Pieces44(pieces)]


def puzzle_alice() -> tuple[str, Sequence[Feature]]:
    # puzzle = "......... 3......8. ..4...... ......... 2...9...7 ......... ......5.. .1......6 ........."
    puzzle = "......... 3....6.8. ..4...... ......... 2...9...7 ......... ......5.. .1......6 ........."  # 18:30

    pieces = "122222939112122333911123333441153666445555696497758966447958886447559886777778889"
    features = [AlternativeBoxesFeature(pieces),
                *(SameValueAsMateFeature((r, c)) for r in range(1, 10) for c in range(1, 10))
                ]
    puzzle = puzzle.replace(' ', '')
    return puzzle, features


def slow_thermometer_puzzle1() -> tuple[str, Sequence[Feature]]:
    puzzle = BLANK_GRID
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
    return puzzle, thermometers


def slow_thermometer_puzzle2() -> tuple[str, Sequence[Feature]]:
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
    return puzzle, thermometers


def thermometer_07_23() -> tuple[str, Sequence[Feature]]:
    puzzle = ".....................9.............5...............3.................8.......9..."
    thermos = [
        "1,1,SE,SE,SE,SW,SW",
        "1,9,SW,SW,SW,NW,NW",
        "9,1,NE,NE,NE,SE,SE",
        "9,9,NW,NW,NW,NE,NE"
    ]
    thermometers = [ThermometerFeature(line, color='lightgray') for line in thermos]
    return puzzle, thermometers


def double_sum_puzzle() -> tuple[str, Sequence[Feature]]:
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
    ]
    return BLANK_GRID, features


def puzzle_hunt() -> tuple[str, Sequence[Feature]]:
    puzzle = "...48...7.8.5..6...9.....3.4...2.3..1...5...2..8..7......8.3.7...5...1.39...15.4."
    features = [BoxOfNineFeature.major_diagonal(), BoxOfNineFeature.minor_diagonal()]
    return puzzle, features


def sandwich_07_28() -> tuple[str, Sequence[Feature]]:
    class LiarsSandwichFeature(SandwichFeature):
        def match(self, permutation: tuple[int, ...]) -> bool:
            return abs(self.sandwich_sum(permutation) - self.total) == 1

    puzzle = "..6................1...........1.....4.........9...2.....................7......8"
    features = [
        *[LiarsSandwichFeature(House.Type.ROW, row, total)
          for row, total in enumerate((5, 8, 5, 16, 12, 7, 5, 3, 1), start=1)],
        LiarsSandwichFeature(House.Type.COLUMN, 1, 5),
        LiarsSandwichFeature(House.Type.COLUMN, 5, 4),
        LiarsSandwichFeature(House.Type.COLUMN, 9, 5),
    ]
    return puzzle, features


def skyscraper_07_29() -> tuple[str, Sequence[Feature]]:
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
    return BLANK_GRID, features


def puzzle_07_30() -> tuple[str, Sequence[Feature]]:
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
    return puzzle, features


def puzzle_07_30_simon() -> tuple[str, Sequence[Feature]]:
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
    return puzzle, thermometers


def puzzle_08_02() -> tuple[str, Sequence[Feature]]:
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
    return BLANK_GRID, features


def puzzle_08_06() -> tuple[str, Sequence[Feature]]:
    offsets1 = [(dr, dc) for dx in (-1, 1) for dy in (-2, 2) for (dr, dc) in ((dx, dy), (dy, dx))]
    offsets2 = [(dr, dc) for delta in range(1, 9) for dr in (-delta, delta) for dc in (-delta, delta)]
    offsets = offsets1 + offsets2

    class MyFeature(SameValueAsMateFeature):
        def get_mates(self, cell: Cell) -> Iterable[Cell]:
            return self.neighbors_from_offsets(cell, offsets)

    features = [MyFeature((i, j)) for i, j in itertools.product(range(1, 10), repeat=2)]
    puzzle = "39.1...822.....5.....4.....6..2.....1....4.........3............6...3..551.....64"
    return puzzle, features


def puzzle_08_07() -> tuple[str, Sequence[Feature]]:
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
    return BLANK_GRID, features


def puzzle_08_12() -> tuple[str, Sequence[Feature]]:
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
    return puzzle, thermometers


def puzzle_08_15() -> tuple[str, Sequence[Feature]]:
    puzzle = "....1...4........5.............................1.....8........75.3....6.....3...."
    odds = [(3, 2), (3, 4), (3, 6), (3, 7), (3, 8),
            (4, 1), (4, 2), (4, 4), (4, 8),
            (5, 2), (5, 4), (5, 5), (5, 6), (5, 8),
            (6, 2), (6, 5), (6, 8),
            (7, 2), (7, 5), (7, 8)]
    features = [KingsMoveFeature(),
                LimitedValuesFeature(odds, (1, 3, 5, 7, 9), color='lightgray')
                ]
    return puzzle, features


def puzzle_08_26() -> tuple[str, Sequence[Feature]]:
    PRIMES = {2, 3, 5, 7, 11, 13, 17}
    def create_prime_ring(squares: SquaresParseable) -> Sequence[Square]:
        return [
            *AdjacentRelationshipFeature.create(squares, prefix="Prime", cyclic=True,
                                                match=lambda i, j: i + j in PRIMES),
            DrawOnlyFeature(lambda context: context.draw_line(squares, closed=True, color='red', linewidth=5)),
        ]

    columns = (21, 25, 11, 0, 35, 23, 13, 4, 18)
    rows = (13, 13, 6, 9, 0, 29, 2, 13, 2)
    features = [
        ThermometerFeature("37,S,S,S,S"),
        create_prime_ring("22,E,E,S,E,E,N,E,E,S,S,W,S,S,E,S,S,W,W,N,W,W,S,W,W,N,N,E,N,N,W,N"),
        *[SandwichFeature(House.Type.ROW, row, total) for row, total in enumerate(rows, start=1)],
        *[SandwichFeature(House.Type.COLUMN, col, total) for col, total in enumerate(columns, start=1)],
    ]
    return BLANK_GRID, features


def puzzle_08_31() -> tuple[str, Sequence[Feature]]:
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
    return puzzle, [*thermometers, snake]


def puzzle_09_03() -> tuple[str, Sequence[Feature]]:
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
    return '1' + ' ' * 80, features


def puzzle_09_05() -> tuple[str, Sequence[Feature]]:
    class DeltaFeature:
        @classmethod
        def create(cls, square: Square, delta: int, is_right: bool):
            row, column = square
            square2 = (row, column + 1) if is_right else (row + 1, column)

            def draw(context):
                (r1, c1), (r2, c2) = (square, square2)
                context.draw_text((c1 + c2 + 1) / 2, (r1 + r2 + 1) / 2, str(delta),
                                  va='center', ha='center', fontsize=15, weight='bold', color='red')

            return [
                *AdjacentRelationshipFeature.create((square, square2), prefix='d',
                                                    match=lambda i, j: abs(i - j) == delta),
                DrawOnlyFeature(draw)
            ]

    features = [
        *DeltaFeature.create((2, 3), 1, True),
        *DeltaFeature.create((2, 6), 1, True),
        *DeltaFeature.create((3, 3), 7, True),
        *DeltaFeature.create((3, 6), 7, True),
        *DeltaFeature.create((4, 3), 5, True),
        *DeltaFeature.create((4, 6), 5, True),
        *DeltaFeature.create((6, 2), 1, True),
        *DeltaFeature.create((6, 7), 2, True),
        *DeltaFeature.create((1, 5), 8, False),
        *DeltaFeature.create((2, 1), 4, False),
        *DeltaFeature.create((2, 9), 4, False),
        *DeltaFeature.create((3, 2), 6, False),
        *DeltaFeature.create((3, 8), 6, False),
        *DeltaFeature.create((4, 5), 3, False),
        *DeltaFeature.create((7, 1), 1, False),
        *DeltaFeature.create((7, 9), 2, False),
        *DeltaFeature.create((8, 2), 1, False),
        *DeltaFeature.create((8, 8), 2, False),

    ]
    # noinspection SpellCheckingInspection
    puzzle = "XXXXXX.921.738.-3.9-X".replace("X", "---").replace("-", "...")
    return puzzle, features


def puzzle_09_06() -> tuple[str, Sequence[Feature]]:
    class CamelJumpFeature(SameValueAsMateFeature):
        OFFSETS = [(dr, dc) for dx in (-1, 1) for dy in (-3, 3) for (dr, dc) in ((dx, dy), (dy, dx))]

        def get_mates(self, cell: Cell) -> Iterable[Cell]:
            return self.neighbors_from_offsets(cell, self.OFFSETS)

        def draw(self, context: DrawContext) -> None:
            if self.done:
                context.draw_outline([self.this_square], linestyle="-")

    all_squares = cast(Iterable[Square], itertools.product(range(1, 10), range(1, 10)))
    features = [CamelJumpFeature(square) for square in all_squares]
    puzzle = "........9.....85..7...2.1..35...............6.96.....7...........1.7.9......452.."
    return puzzle, features


def puzzle_09_10() -> tuple[str, Sequence[Feature]]:
    puzzle = "....9.....14..5..8......3.4.2.3..74...6.......81.24....6.9..........69.28.9..1..3"
    return puzzle, ()


def puzzle_09_15() -> tuple[str, Sequence[Feature]]:
    puzzle = "-----.5.3.8.2.2.5.3.6.9.9.4.6.1.-".replace('-', '.........')
    features = XVFeature.create(
        down={5: [(1, 3), (1, 5), (1, 7), (2, 2), (2, 4), (2, 6), (2, 8), (3, 3), (3, 5), (3, 7)]},
        across={}
    )
    return puzzle, features


def puzzle_09_16() -> tuple[str, Sequence[Feature]]:
    puzzle = "529784361............2......4....2..361529784..2....3......2............784361529"
    features = [BoxOfNineFeature.major_diagonal(), BoxOfNineFeature.minor_diagonal()]
    return puzzle, features


def puzzle_09_17_2() -> tuple[str, Sequence[Feature]]:
    puzzle = "..8..964.9..1.6........7....3.....5.7.2...4.3.6.....8...18..9.....4....5.832..1.."
    return puzzle, ()


def puzzle_09_20() -> tuple[str, Sequence[Feature]]:
    # noinspection SpellCheckingInspection
    puzzle = "XXXXX-1..-XXX".replace("X", "---").replace("-", "...")
    features = [
        *SandwichFeature.create_all(House.Type.ROW, [10, 19, 25, 28, 17, 3, 23, 6, 7]),
        *SandwichFeature.create_all(House.Type.COLUMN, [18, 8, 21, 18, 13, 27, 25, 13, 3]),
        KnightsMoveFeature()
    ]
    return puzzle, features


def puzzle_09_21() -> tuple[str, Sequence[Feature]]:
    class Multiplication(PossibilitiesFeature):
        def __init__(self, row, column) -> None:
            squares = [(row, column), (row, column + 1), (row + 1, column), (row + 1, column + 1)]
            super().__init__(squares, name=f"Square{row}{column}", neighbors=True)

        def get_possibilities(self) -> list[tuple[int, ...]]:
            for x, y in itertools.product(range(2, 10), repeat=2):  # we now x,y ≠ 1
                q, r = divmod(x * y, 10)
                if 1 <= q <= 9 and 1 <= r <= 9:
                    yield x, y, q, r

        def draw(self, context: DrawContext) -> None:
            context.draw_rectangles(self.squares, color='lightgray')

    puzzle = "X..7-6...5.-.8.-..9-X-5..-.6.-.9...1-2..X".replace("X", "---").replace("-", "...")
    features = [
        Multiplication(1, 1), Multiplication(1, 8), Multiplication(3, 3), Multiplication(6, 6),
        Multiplication(8, 1), Multiplication(8, 8)
    ]
    return puzzle, features


def puzzle_10_17() -> tuple[str, Sequence[Feature]]:
    # noinspection SpellCheckingInspection
    puzzle = "XXXXX3.9.4.1.6.9.4.5.3.8.7.6.5.4X".replace("X", "---").replace("-", "...")
    return puzzle, AdjacentNotConsecutiveFeature.create()


def puzzle_2021_01_21() -> tuple[str, Sequence[Feature]]:
    puzzle = "1..4..6...2..5..9...3..6..87..1..9...9..2..4...4..3..55..8..2...8..7..3...7..9..1"
    features = ()
    return puzzle, features


def puzzle_2021_03_15() -> tuple[str, Sequence[Feature]]:
    puzzle = "...........3.1.4..1..5.9.......2.65.35.....89.79.3.......2.3..8..4.6.2..........."
    features = ()
    return puzzle, features


def puzzle_2021_05_12() -> tuple[str, Sequence[Feature]]:
    puzzle = "4.......2..5.829...2.....3...8.1....56..9..78....6.5...1.....6...615.7..3.......4"
    features = ()
    return puzzle, features


def puzzle_2021_05_24() -> tuple[str, Sequence[Feature]]:
    lines = ["1,1,E", "1,3,SW", "1,6,S", "1,7,E", "2,3,NE", "2,5,N", "2,9,N", "3,1,N", "3,3,NE", "3,4,SE", "3,5,SE",
             "3,6,NE", "3,7,SE", "3,8,SE", "3,9,NW", "4,1,E", "4,3,NW", "4,7,SW", "5,1,E", "5,3,NE", "5,7,SW", "5,9,W",
             "6,3,NE", "6,4,NE", "6,7,SE", "6,9,W", "7,1,SE", "7,2,NW", "7,3,NW", "7,4,SW", "7,6,W", "7,7,SW", "7,9,S",
             "8,1,S", "8,7,SW", "9,3,W", "9,4,N", "9,5,N", "9,7,NE", "9,9,W"]
    thermometers = [ThermometerFeature(line) for line in lines]
    return BLANK_GRID, thermometers


def puzzle_2021_07_06() -> tuple[str, Sequence[Feature]]:
    grid = "-.1.---.8.----.2.----------------".replace('-', '...')
    a, b, c, d, e, f, g, h, i, j = (1, 1), (3, 3), (5, 1), (5, 5), (5, 9), (7, 3), (7, 7), (9, 1), (9, 5), (9, 9)
    ends = [(a, b), (a, c), (b, c), (b, d), (c, d), (c, f), (c, h), (d, f), (d, i), (d, g), (e, g), (e, j),
            (f, h), (f, i), (g, i), (g, j), (h, i), (i, j)]
    features: list[Feature] = [ExtremeEndpointsFeature.between(x, y) for x, y in ends]
    return grid, features


def puzzle_2021_07_10() -> tuple[str, Sequence[Feature]]:
    killers = [
        (7, "2,2,S,S"),
        (5, "2,6,E"),
        (6, "3,7,E"),
        (6, "4,6,E"),
        (23, "6,3,E,S"),
        (15, "6,9,S"),
        (17, "7,3,S"),
        (3, "9,6,E")]
    kill_features: list[PossibilitiesFeature] = [KillerCageFeature(total, squares) for total, squares in killers]
    features = [
        *kill_features,
    ]
    return BLANK_GRID, features


def puzzle_2021_07_11() -> tuple[str, Sequence[Feature]]:
    features = [
        *LimitedValuesFeature.odds_and_evens(evens=[(1, 4), (2, 2), (2, 9), (8, 1), (8, 1), (8, 8), (9, 6)]),
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
    return BLANK_GRID, features


def puzzle_2021_07_31() -> tuple[str, Sequence[Feature]]:
    features = [
        BoxOfNineFeature.major_diagonal(),
        BoxOfNineFeature.minor_diagonal(),
        ArrowSumFeature("27,NW,W,W"),
        ArrowSumFeature("32,SW,S,S"),
        ArrowSumFeature("78,NE,N,N"),
        ArrowSumFeature("83,SE,E"),
        KillerCageFeature(5, "13,S"),
        KillerCageFeature(13, "24,E,SW"),
        KillerCageFeature(9, "38,S,W"),
        KillerCageFeature(22, "42,E,SW"),
        KillerCageFeature(10, "45,E,S"),
        KillerCageFeature(15, "54,S,E"),
        KillerCageFeature(24, "58,S,W"),
        KillerCageFeature(12, "62,E,SW"),
        KillerCageFeature(18, "76,S,W"),
        KillerCageFeature(5, "87,s")
    ]
    grid = 'XXXX-.9.-XX-..X1-'.replace('X', '---').replace('-', '...')
    return grid, features


def puzzle_2021_08_02() -> tuple[str, Sequence[Feature]]:
    features = [
        *ArithmeticFeature.create("14", "6-"),
        *ArithmeticFeature.create("15", "11+"),
        *ArithmeticFeature.create("16", "2"),
        *ArithmeticFeature.create("22", "4-"),
        *ArithmeticFeature.create("37", "2/"),
        *ArithmeticFeature.create("61", "13+"),
        *ArithmeticFeature.create("63", "?/"),
        *ArithmeticFeature.create("66", "?x"),
        *ArithmeticFeature.create("68", "7+"),
        *ArithmeticFeature.create("73", "359"),
        ]
    return BLANK_GRID, features


def puzzle_2021_08_03() -> tuple[str, Sequence[Feature]]:
    features = [
        ThermometerFeature("33,E,E,E,E"),
        ThermometerFeature("77,N,N,N"),
        KillerCageFeature(21, "11,E,S,W"),
        KillerCageFeature(18, "18,E,S,W"),
        KillerCageFeature(30, "88,E,S,W"),
        KillerCageFeature(7, "83,S"),
        ThermometerAsLessThanFeature("91,92"),
        *PalindromeFeature.create("73,N,N,N,N,E,E,E,E"),
        *PalindromeFeature.create("74,E,E,E,N,N,N")
        ]
    return BLANK_GRID, features


def main():
    start = datetime.datetime.now()
    grid, features = puzzle_09_06()
    Sudoku().solve(grid, features=features, initial_only=False, medusa=True, guides=1)
    end = datetime.datetime.now()
    print(end - start)


if __name__ == '__main__':
    main()
