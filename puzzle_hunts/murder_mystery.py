import datetime
import itertools
from typing import Iterable, Optional, Sequence, cast

from cell import Cell, House
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from features.chess_move import KingsMoveFeature, KnightsMoveFeature, QueensMoveFeature, TaxicabFeature
from features.features import AdjacentNotConsecutiveFeature, AlternativeBoxesFeature, \
    BoxOfNineFeature, CloneBoxFeature, \
    KillerCageFeature, MessageFeature, XVFeature, PalindromeFeature
from features.possibilities_feature import PossibilitiesFeature
from features.sandwich_feature import SandwichFeature
from features.thermometer import ThermometerAsLessThanFeature, ThermometerFeature
from human_sudoku import Sudoku


class FakeKillerCageFeature(Feature):
    squares: Sequence[Square]

    def __init__(self, squares: SquaresParseable):
        super().__init__()
        self.squares = Feature.parse_squares(squares)

    def draw(self, context: DrawContext) -> None:
        context.draw_rectangles(self.squares, facecolor='#a89dbc')
        context.draw_outline(self.squares)
        if all((self @ square).is_known for square in self.squares):
            total = sum((self @ square).known_value for square in self.squares)
            row, column = min(self.squares)
            context.draw_text(column + .2, row + .2, str(total),
                              va='top', ha='left', color='blue', fontsize=10, weight='bold')
            if context.done:
                print(f'KillerCageFeature({total}, {self.squares})')


class QKillerCageFeature(PossibilitiesFeature):
    total: int
    squares: Sequence[Square]
    puzzle: int
    criminal: str

    def __init__(self, total: int, squares: SquaresParseable, *,
                 puzzle: int, name: str, dr: int = 0, dc: int = 0):
        self.total = total
        self.criminal = name
        super().__init__([(r + dr, c + dc) for r, c in Feature.parse_squares(squares)], name=f'Suspect #{puzzle}')

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        count = len(self.squares)
        for values in itertools.permutations(range(1, 10), count - 1):
            last_value = self.total - sum(values)
            for x in (last_value - 1, last_value, last_value + 1):
                if 1 <= x <= 9 and x not in values:
                    yield *values, x

    def draw(self, context: DrawContext) -> None:
        context.draw_rectangles(self.squares, facecolor='#d1c6db')
        context.draw_outline(self.squares)
        row, column = min(self.squares)
        context.draw_text(column + .2, row + .2, str(self.total),
                          va='top', ha='left', color='blue', fontsize=10, weight='bold')
        if context.done and all((self @ square).is_known for square in self.squares):
            real_total = sum((self @ square).known_value for square in self.squares)
            symbol = '✓' if self.total == real_total else '❌'
            print(f'For {self}: expected {self.total}; got {real_total} {symbol} {self.criminal}')


class DrawCircleFeature(Feature):
    squares: Sequence[Square]

    def __init__(self, squares: SquaresParseable):
        super().__init__()
        self.squares = self.parse_squares(squares)

    def draw(self, context: DrawContext) -> None:
        for row, column in self.squares:
            context.draw_circle((column + .5, row + .5), radius=.5, fill=False, color='blue')
        if all(self.grid.matrix[square].is_known for square in self.squares):
            value = ''.join(str(self.grid.matrix[square].known_value) for square in self.squares)
            print('Value =', value)


class MyQueenFeature(Feature):
    queens_move_feature: Optional[QueensMoveFeature]

    def __init__(self):
        super().__init__()
        self.queens_move_feature = None

    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        if self.queens_move_feature is not None:
            return self.queens_move_feature.get_neighbors_for_value(cell, value)
        else:
            return ()

    def check(self) -> bool:
        center_cell = self @ (5, 5)
        if not center_cell.is_known or self.queens_move_feature is not None:
            return False
        center_cell_value = center_cell.known_value
        queens_move_feature = QueensMoveFeature(values={center_cell_value})
        queens_move_feature.initialize(self.grid)
        self.queens_move_feature = queens_move_feature

        print(f"Center square assigned value {center_cell_value}")
        affected_cells = {neighbor
                          for cell in self.grid.cells
                          if cell.is_known and cell.known_value == center_cell_value
                          for neighbor in queens_move_feature.get_neighbors_for_value(cell, center_cell_value)
                          if center_cell_value in neighbor.possible_values}
        Cell.remove_value_from_cells(affected_cells, center_cell_value, show=True)
        return True


class DumpResultFeature(Feature):
    def draw(self, context: DrawContext) -> None:
        if context.done:
            result = [str(cell.known_value) if cell.is_known else '.'
                      for square in itertools.product(range(1, 10), repeat=2)
                      for cell in [self @ cast(Square, square)]]
            print(f'old_grid = "{"".join(result)}"')


def from_grid(old_grid: str, squares: Sequence[int], default: str = '.' * 81):
    result = list(default)
    for value in squares:
        assert value >= 10
        high, low = divmod(value, 10)
        index = high * 9 + low - 10
        result[index] = old_grid[index]
    return ''.join(result)


def act_1() -> tuple[str, Sequence[Feature]]:
    # KillerCageFeature(15, [(1, 1), (1, 2)]
    # KillerCageFeature(6, [(1, 5), (2, 5)]
    # KillerCageFeature(11, [(1, 9), (2, 9)]
    # KillerCageFeature(8, [(4, 4), (4, 5)]
    # KillerCageFeature(19, [(5, 3), (5, 4), (6, 4)]
    # KillerCageFeature(13, [(5, 6), (6, 6), (6, 5)]
    # KillerCageFeature(7, [(8, 4), (9, 4)]
    # KillerCageFeature(9, [(8, 6), (9, 6)]
    info = [
        ("1,1,E", None),
        ("1,3,E,S", 15),
        ("1,5,S", None),
        ("1,6,S,S", 21),
        ("1,7,E,SW", 6),
        ("1,9,S", None),
        ("2,1,E,E,SW,W", 21),
        ("2,8,S,W", 23),
        ("3,3,S", 4),
        ("3,4,E", 11),
        ("3,9,S,S", 8),
        ("4,1,S", 12),
        ("4,2,S", 7),
        ("4,4,E", None),
        ("4,6,E", 10),
        ("4,8,S,W", 24),
        ("5,3,E,S", None),
        ("5,6,S,W", None),
        ("6,1,S", 6),
        ("6,2,E", 16),
        ("6,7,E", 8),
        ("6,9,S,W", 16),
        ("7,2,S,W,S", 18),
        ("7,3,E,SW", 12),
        ("7,5,S,S", 22),
        ("7,6,E", 10),
        ("8,4,S", None),
        ("8,6,S", None),
        ("8,7,S,E", 14),
        ("8,8,E,S", 15),
        ("9,2,E", 13)
    ]
    features = [
        *[KillerCageFeature(total, squares) for squares, total in info if total is not None],
        *[FakeKillerCageFeature(squares) for squares, total in info if total is None],
        BoxOfNineFeature.major_diagonal(),
        BoxOfNineFeature.minor_diagonal(),
        # QueensMoveFeature(values={5}),
        MyQueenFeature(),
        DumpResultFeature(),
    ]
    return " " * 81, features


def act_2() -> tuple[str, Sequence[Feature]]:
    # KillerCageFeature(32, [(6, 7), (7, 6), (7, 7), (7, 8), (8, 7)])

    thermos = ("1,2,SW,S,SE",
               "1,4,SE,S,SW", "4,8,N,NE,NW,SW",
               "5,6,SW,W,NW,N", "8,1,NE,S,E", "8,6,NE,NE,S,S,E", "9,3,NE,E")
    thermometers = [ThermometerFeature(thermo) for thermo in thermos]
    # thermometers[4:7] = [CombinedPossibilitiesFeature(thermometers[4:7])]
    # thermometers[0:3] = [CombinedPossibilitiesFeature(thermometers[0:3])]
    features = [
        FakeKillerCageFeature("6,7,SW,E,E,SW"),
        *thermometers,
        DumpResultFeature(),
        # Helper()
    ]

    old_grid = "968145237735629184241837965853716492426953871179482356582374619617298543394561728"
    grid = from_grid(old_grid, [11, 13, 16, 62, 68, 75, 87, 97, 98])
    return grid, features


def act_3() -> tuple[str, Sequence[Feature]]:
    # KillerCageFeature(18, [(5, 3), (6, 3), (7, 2), (7, 3), (7, 4)]
    features = [
        XVFeature(across={5: "15, 22, 24, 32, 52", 10: "14, 41, 54, 56, 62"},
                  down={5: "39", 10: "25, 33, 48"},
                  all_listed=True),
        KingsMoveFeature(),
        FakeKillerCageFeature("5,3,S,SW,E,E"),
        DumpResultFeature(),
    ]

    old_grid = "928145376614723895753689142386957214495812637172436958539278461247361589861594723"
    grid = from_grid(old_grid, [18, 29, 81, 92])
    return grid, features


def act_4_runner():
    boxes = [(1, 9), (3, 4), (5, 7)]
    for box1, box2 in boxes:
        grid, features = act_4(box1, box2)
        try:
            result = Sudoku().solve(grid, features=features)
        except AssertionError:
            result = False

        if result:
            return


def act_4(box1: int = 1, box2: int = 9) -> tuple[str, Sequence[Feature]]:
    # KillerCageFeature(34, [(4, 4), (5, 4), (6, 4), (5, 5), (4, 6), (5, 6), (6, 6)]
    # Success occurs with boxes 1 and 9

    features = [
        CloneBoxFeature(box1, box2),
        FakeKillerCageFeature("4,4,S,S,NE,NE,S,S"),
        DumpResultFeature(),
    ]
    old_grid = "659823174723149685814567293376981542541372869982654317435718926298436751167295438"
    new_grid = ".....4.......8..2..3....9.....5.....4.......7...8.......3.2..1..6..5.........9..2"
    grid = from_grid(old_grid, [11, 36, 53, 58], default=new_grid)
    return grid, features


def act_5() -> tuple[str, Sequence[Feature]]:
    # KillerCageFeature(22, [(7, 4), (8, 4), (8, 5), (7, 6), (8, 6), (9, 5)])_
    boxes = ["11,S,19,W,99,N,91,E,55",
             "14,W,W,SE,W,S,W,S,S",
             "15,E,E,S,E,E,SW,E,S",
             "61,S,E,SW,E,E,S,E,E",
             "59,S,S,W,S,W,SE,W,W",
             "26,W,W,S,W,S,S,E,S",
             "42,S,S,E,S,E,E,N,E",
             "44,E,N,E,E,S,E,S,S",
             "46,S,E,S,S,W,S,W,W"]
    thermometers = ("14,S", "16,S", "25,N", "32,E", "32,S", "33,N", "37,N", "38,W", "38,S",
                    "41,E", "49,W", "51,E", "59,W", "62,W", "62,S", "68,E", "68,S",
                    "73,W", "77,E", "84,S", "86,S", "95,N")
    features = [
        AlternativeBoxesFeature(boxes),
        *[ThermometerAsLessThanFeature(i) for i in thermometers],
        FakeKillerCageFeature("74,S,E,NE,S,SW"),
        DumpResultFeature(),
    ]
    old_grid = "615294378749385126832617954397546281481932567526871493973428615268153749154769832"
    new_grid = "XX...9.4.....5...2......3......2...8......9....XX".replace('X', '.........')
    grid = from_grid(old_grid, [12, 29, 31, 79, 81, 98], default=new_grid)
    return grid, features


def act_6() -> tuple[str, Sequence[Feature]]:
    # KillerCageFeature(20, [(2, 4), (2, 5), (3, 4), (3, 5), (3, 6)])
    features = [
        TaxicabFeature(),
        FakeKillerCageFeature("24,25,34,35,36"),
        DumpResultFeature(),
    ]
    old_grid = "513287964179328546857964321465871293684139752732645819341792685296453178928516437"
    new_grid = "23.--..1..9---.3..6..5.--324--.6..9..4.---1..3..--.89".replace('-', '...')
    grid = from_grid(old_grid, [18, 27, 83, 92], default=new_grid)
    return grid, features


def act_7() -> tuple[str, Sequence[Feature]]:
    # KillerCageFeature(29, [(7, 5), (7, 6), (8, 5), (9, 5), (9, 6)])
    features = [
        *SandwichFeature.create_all(House.Type.ROW, (0, 2, None, None, None, 21, None, 8, 9)),
        *SandwichFeature.create_all(House.Type.COLUMN, (7, None, 9, 11, 11, 4, 11, 6, 35)),
        FakeKillerCageFeature("75,76,85,95,96"),
        DumpResultFeature(),
    ]
    old_grid = "239475168871639542654812937362958471917324856485761293748293615596187324123546789"
    grid = from_grid(old_grid, [16, 33, 35, 39, 41, 74, 99])
    return grid, features


def act_8() -> tuple[str, Sequence[Feature]]:
    # KillerCageFeature(28, [(4, 2), (4, 3), (5, 3), (5, 4)])
    features = [
        KnightsMoveFeature(),
        FakeKillerCageFeature("42,E,S,E"),
        DumpResultFeature(),
    ]
    old_grid = "637425891812937546594618327348576912259143768176892453763259184981364275425781639"
    new_grid = "2....9.65-5..--.3.---------9...82-69..1.-.5.9..852..4".replace('-', "...")
    grid = from_grid(old_grid, [12, 14, 26, 63], default=new_grid)
    return grid, features


# noinspection SpellCheckingInspection
def act_9() -> tuple[str, Sequence[Feature]]:
    # KillerCageFeature(12, [(3, 5), (3, 6), (4, 6), (4, 5)])
    # E=9, G=2, H=7, L=8, N=1, O=4, R=5, S=3, V=6
    # NGSORVHLE

    # KillerCageFeature(12, [(3, 5), (3, 6), (4, 6), (4, 5)])

    features = [
        MessageFeature("HELLSERVELONG", "15,SE,SE,SE,44,SE,SE,SE,SE,63,SE,SE,SE"),
        PalindromeFeature("43,NW,S,S,S,S,NE,E,SE,NE,E,SE,N,N,N,N,SW,W,NW,SW,S"),
        FakeKillerCageFeature("35,E,S,W"),
        DumpResultFeature(),
        # SameValueFeature("62,55"),
        # SameValueFeature("63,56"),
    ]
    old_grid = "231489765894567231675231489758923146149678523326145978582314697413796852967852314"
    grid = from_grid(old_grid, [13, 25, 39, 41, 45, 57, 81, 86, 97])
    return grid, features


def act_10() -> tuple[str, Sequence[Feature]]:
    features = [
        # From Puzzle1
        KillerCageFeature(15, [(1, 1), (1, 2)]),
        KillerCageFeature(6, [(1, 5), (2, 5)]),
        KillerCageFeature(11, [(1, 9), (2, 9)]),
        KillerCageFeature(8, [(4, 4), (4, 5)]),
        KillerCageFeature(19, [(5, 3), (5, 4), (6, 4)]),
        KillerCageFeature(13, [(5, 6), (6, 6), (6, 5)]),
        KillerCageFeature(7, [(8, 4), (9, 4)]),
        KillerCageFeature(9, [(8, 6), (9, 6)]),
        # From Puzzle 2.
        QKillerCageFeature(32, [(6, 7), (7, 6), (7, 7), (7, 8), (8, 7)], puzzle=2, name="Basil"),
        # From Puzzle 3.
        QKillerCageFeature(18, [(5, 3), (6, 3), (7, 2), (7, 3), (7, 4)], dr=-4, puzzle=3, name="Claude"),
        # From Puzzle 4
        QKillerCageFeature(34, [(4, 4), (5, 4), (6, 4), (5, 5), (4, 6), (5, 6), (6, 6)], dr=-3, dc=2,
                           puzzle=4, name="Derek&Eric"),
        # From Puzzle 5
        QKillerCageFeature(22, [(7, 4), (8, 4), (8, 5), (7, 6), (8, 6), (9, 5)], dc=+3, dr=-4, puzzle=5, name="Fiona"),
        # From Puzzle 6
        QKillerCageFeature(20, [(2, 4), (2, 5), (3, 4), (3, 5), (3, 6)], dc=-3, dr=4, puzzle=6, name="Gus"),
        # From Puzzle 7
        QKillerCageFeature(29, [(7, 5), (7, 6), (8, 5), (9, 5), (9, 6)], dc=-4, dr=-5, puzzle=7, name="Horatia"),
        # From Puzzle 8
        QKillerCageFeature(28, [(4, 2), (4, 3), (5, 3), (5, 4)], dr=4, dc=-1, puzzle=8, name="Isaac"),
        # From Puzzle 9
        QKillerCageFeature(12, [(3, 5), (3, 6), (4, 6), (4, 5)], dc=3, dr=5, puzzle=9, name="Jarvis"),
        DumpResultFeature(),
        DrawCircleFeature("51,E,SE,SE,SE,S"),
        DrawCircleFeature("14,S,SE,SE,SE,SE,E")
    ]

    # queens on the board.
    act_1_grid = "968145237735629184241837965853716492426953871179482356582374619617298543394561728"
    grid = from_grid(act_1_grid, [55])
    # thermometers on the board
    act_2_grid = "928145376614723895753689142386957214495812637172436958539278461247361589861594723"
    grid = from_grid(act_2_grid, [43], default=grid)
    # kings on the board
    act_3_grid = "659823174723149685814567293376981542541372869982654317435718926298436751167295438"
    grid = from_grid(act_3_grid, [75], default=grid)
    # clones
    act_4_grid = "615294378749385126832617954397546281481932567526871493973428615268153749154769832"
    grid = from_grid(act_4_grid, [91], default=grid)
    # roses
    act_5_grid = "513287964179328546857964321465871293684139752732645819341792685296453178928516437"
    grid = from_grid(act_5_grid, [17], default=grid)
    # taxicab
    act_6_grid = "239475168871639542654812937362958471917324856485761293748293615596187324123546789"
    grid = from_grid(act_6_grid, [59], default=grid)
    # tuna
    act_7_grid = "637425891812937546594618327348576912259143768176892453763259184981364275425781639"
    grid = from_grid(act_7_grid, [97], default=grid)
    # knight
    act_8_grid = "231489765894567231675231489758923146149678523326145978582314697413796852967852314"
    grid = from_grid(act_8_grid, [79], default=grid)
    # palindrome ???
    act_9_grid = "981273465542869713637154829756321984314698572298745136829437651473516298165982347"
    grid = from_grid(act_9_grid, [83], default=grid)

    return grid, features


def finale() -> tuple[str, Sequence[Feature]]:
    features = [
        KnightsMoveFeature(),
        AdjacentNotConsecutiveFeature(),
        AlternativeBoxesFeature(["11,S,S,S,SE,N,N,N,SE", "12,E,E,E,SE,W,W,W,SE", "16,E,E,E,SW,W,SW,E,E",
                                 "35,SW,E,SW,E,E,SW,E,SW", "29,S,S,S,NW,W,W,SE,S", "43,S,SE,W,W,NW,S,S,S",
                                 "91,E,E,E,NW,W,N,E,E", "76,SE,W,W,W,SE,E,E,E", "58,S,E,S,W,W,SE,E,S"]),
    ]
    # noinspection SpellCheckingInspection
    grid = "X--4..XX--.3.XXXX".replace('X', '---').replace('-', '...')

    # roses
    act_5_grid = "513287964179328546857964321465871293684139752732645819341792685296453178928516437"
    grid = from_grid(act_5_grid, [13], default=grid)
    # knight
    act_8_grid = "231489765894567231675231489758923146149678523326145978582314697413796852967852314"
    grid = from_grid(act_8_grid, [73], default=grid)
    return grid, features


# noinspection SpellCheckingInspection
def main():
    start = datetime.datetime.now()
    puzzles = [
        act_1,
        act_2,
        act_3,
        # act_4, act_5, act_6,
        # act_7,
        # act_8,
        # act_9,
        # act_10,
        # finale,
    ]
    for puzzle in puzzles:
        print('*************', puzzle.__name__, "*****************")
        grid, features = puzzle()
        result = Sudoku().solve(grid, features=features, medusa=False)
        assert result
    end = datetime.datetime.now()
    print(end - start)
    # E=9, G=2, H=7, L=8, N=1, O=4, R=5, S=3, V=6
    # 1=N, 2=G, 3=S, 4=O, 5=R, 6=V, 7=H, 8=L, 9=E
    # SHOVEL
    # REVENGE
    # FIONAISAACREVENGESHOVEL
    #  3 7  9   5 4 3   9          S H E   R O S E


if __name__ == '__main__':
    main()
