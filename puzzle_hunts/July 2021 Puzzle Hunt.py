import itertools
from typing import Sequence, cast

from cell import House
from draw_context import DrawContext
from feature import Feature, Square
from features.chess_move import KingsMoveFeature, KnightsMoveFeature
from features.features import AdjacentNotConsecutiveFeature, ArrowSumFeature, \
    BoxOfNineFeature, ExtremeEndpointsFeature, KillerCageFeature, KropkeDotFeature, LimitedValuesFeature, \
    LittleKillerFeature, LocalMinOrMaxFeature, PalindromeFeature, ValuesAroundIntersectionFeature, XVFeature
from features.same_value_feature import SameValueFeature
from features.sandwich_feature import SandwichFeature
from features.thermometer import ThermometerFeature
from human_sudoku import Sudoku


class DrawCircleFeature(Feature):
    squares: Sequence[tuple[int, int]]

    def __init__(self, squares: Sequence[tuple[int, int]]):
        super().__init__()
        self.squares = squares

    def draw(self, context: DrawContext) -> None:
        for row, column in self.squares:
            context.draw_circle((column + .5, row + .5), radius=.5, fill=False, color='blue')
        if all(self.grid.matrix[square].is_known for square in self.squares):
            value = ''.join(str(self.grid.matrix[square].known_value) for square in self.squares)
            print('Value =', value)
            context.draw_text(5.5, 0, value, fontsize=25, va='center', ha='center')


def puzzle_1() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/somethingforevery1
    puzzle = "...712.....7...3...5...8.471.8.....27.5.....19..1..4.5.2.4...6...3...7.....951..."
    feature = DrawCircleFeature([(1, 1), (1, 9), (3, 4), (4, 7), (5, 5), (6, 3), (7, 6), (9, 1), (9, 9)])
    return puzzle, [feature]


def puzzle_2() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/366662343
    puzzle = "9...2...1.1.....2...2...3.......5...2.......5...3.......6...7...7.....8.5...7...6"
    features = [
        DrawCircleFeature([(1, 3), (2, 4), (3, 9), (4, 8), (5, 5), (6, 2), (7, 1), (8, 6), (9, 7)]),
        *AdjacentNotConsecutiveFeature.create()
    ]
    return puzzle, features


def puzzle_3() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/799745399
    puzzle = "6...1...7.3.....2...1...5.....3.8...1.......2...1.2.....6...2...1.....4.4...2...3"
    features = [
        DrawCircleFeature([(1, 3), (1, 7), (3, 1), (3, 9), (5, 5), (7, 1), (7, 9), (9, 3), (9, 7)]),
        KnightsMoveFeature()
    ]
    return puzzle, features


def puzzle_4() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/438953586
    puzzle = "...........51.23........1...1..5..2...........4..6..3...7........84.36..........."
    circles = cast(Sequence[Square], list(itertools.product((1, 5, 9), (1, 5, 9))))
    features = [
        DrawCircleFeature(circles),
        KingsMoveFeature()
    ]
    return puzzle, features


def puzzle_5() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/192531473
    puzzle = '.6.-.5.1.72..3...3..8.--..8.3.---.4.9..--.1..4...1..25.3.5.-.7.'.replace('-', '...')
    features = [
        DrawCircleFeature([(2, 2), (2, 9), (3, 8), (4, 2), (5, 5), (6, 8), (7, 2), (8, 1), (8, 8)]),
        *BoxOfNineFeature.disjoint_groups()
    ]
    return puzzle, features


def puzzle_6() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/969231896
    puzzle = '1..3..5.9.2.---..4..6.6.2..7..---..8..6.5.8..6..---.4.2.7..5..3'.replace('-', '...')
    features = [
        DrawCircleFeature([(1, 6), (2, 1), (3, 7), (4, 9), (5, 5), (6, 1), (7, 3), (8, 9), (9, 4)]),
        BoxOfNineFeature(((1, 3), (2, 3), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 3), (5, 3))),
        BoxOfNineFeature(((5, 2), (6, 2), (7, 2), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (9, 2))),
        BoxOfNineFeature(((1, 8), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 8), (4, 8), (5, 8))),
        BoxOfNineFeature(((5, 7), (6, 7), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (8, 7), (9, 7))),
    ]
    return puzzle, features


def puzzle_7() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/271479371
    puzzle = '4..-..3.1.2.3.4.------5...6...7------.8.9.1.2.3..-..5'.replace('-', '...')
    palindromes = ["3,1,SE,E", "3,2,E,SE", "3,4,E,SE", "3,6,SE,E", "3,7,E,SE",
                   "5,3,E,NE", "6,5,NE,E",
                   "7,1,NE,E", "7,2,E,NE", "7,4,E,NE", "7,6,NE,E", "7,7,E, NE"]
    features = [
        DrawCircleFeature([(1, 3), (1, 7), (2, 5), (5, 2), (5, 8), (8, 1), (8, 9), (9, 4), (9, 6)]),
        *[f for descriptor in palindromes for f in PalindromeFeature.create(descriptor, color='gray')],
    ]
    return puzzle, features


def puzzle_8() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/867486467
    puzzle = '----,,123,-,,741.-..482.---.481..-.572..-.649..----'.replace('-', '...')
    features = [
        DrawCircleFeature([(1, 4), (1, 8), (2, 1), (4, 9), (5, 5), (6, 1), (8, 9), (9, 2), (9, 6)]),
        *[SameValueFeature(((r, c), (r + 4, c + 4))) for r in (2, 3, 4) for c in (2, 3, 4)]
    ]
    return puzzle, features


def puzzle_9() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/384937216
    grid = '2.O...O.1.e7e.e.o.O.e2o.o9O.o.6..1o...o.O.o...o3..1.o.O5o.o4e.O.o.e.e3e.3.O...O.4'
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.').replace('o', '.').replace('e', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']
    odds = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'o']
    evens = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'e']
    features = [
        DrawCircleFeature(circles),
        *LimitedValuesFeature.odds_and_evens(odds=odds, evens=evens)
    ]
    return puzzle, features


def puzzle_10() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/561876789
    grid = '.O3...1O.1..9.2..8.2.O5O.7...4-5..6..4.5..3O.5...4.O.8..9..1.7..5O1..2.O1...8O.'
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.').replace('o', '.').replace('e', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']
    features = [
        DrawCircleFeature(circles),
        BoxOfNineFeature.major_diagonal(),
        BoxOfNineFeature.minor_diagonal(),
    ]
    return puzzle, features


def puzzle_11() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/521327894
    grid = '-O.O-..9--O..-..O----.O.----O..-..O--2..-O.O-'
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']
    thermometers = ["1,3,W", "1,7,E,E,S,W", "2,6,S,S,E,N,N", "4,2,E,E,N,W,W", "6,8,W,W,S,E,E", "8,4,N,N,W,S,S",
                    "9,3,W,W,N,E", "9,7,E"]
    features = [
        DrawCircleFeature(circles),
        *[ThermometerFeature(thermometer) for thermometer in thermometers]
    ]
    return puzzle, features


def puzzle_12() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/891914194
    grid = '-------.O.--O.O-..O.O.O..-O.O--.O--------'
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']
    features = [
        DrawCircleFeature(circles),
        KillerCageFeature(10, [(1, 1), (1, 2), (2, 1), (2, 2)]),
        KillerCageFeature(12, [(1, 3), (2, 3), (2, 4)]),
        KillerCageFeature(5, [(1, 4), (1, 5), ]),
        KillerCageFeature(18, [(1, 6), (1, 7), (2, 7)]),
        KillerCageFeature(30, [(1, 8), (1, 9), (2, 8), (2, 9)]),
        KillerCageFeature(15, [(2, 5), (2, 6)]),
        KillerCageFeature(16, Feature.parse_squares("3,1,E,S")),
        KillerCageFeature(15, Feature.parse_squares("3,4,W,S")),
        KillerCageFeature(14, Feature.parse_squares("3,6,E,S")),
        KillerCageFeature(14, Feature.parse_squares("3,8,E,S")),
        KillerCageFeature(13, Feature.parse_squares("4,1,S")),
        KillerCageFeature(9, Feature.parse_squares("4,8,S")),
        KillerCageFeature(12, Feature.parse_squares("5,2,S")),
        KillerCageFeature(9, Feature.parse_squares("5,9,S")),
        KillerCageFeature(22, Feature.parse_squares("6,1,S,E")),
        KillerCageFeature(10, Feature.parse_squares("6,3,S,E")),
        KillerCageFeature(17, Feature.parse_squares("6,7,S,W")),
        KillerCageFeature(7, Feature.parse_squares("6,8,S,E")),
        KillerCageFeature(17, Feature.parse_squares("8,1,E,S,W")),
        KillerCageFeature(17, Feature.parse_squares("8,3,S,E")),
        KillerCageFeature(8, Feature.parse_squares("8,4,E")),
        KillerCageFeature(14, Feature.parse_squares("8,6,E,S")),
        KillerCageFeature(21, Feature.parse_squares("8,8,E,S,W")),
        KillerCageFeature(13, Feature.parse_squares("9,5,E"))
    ]
    return puzzle, features


def puzzle_13() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/535762978
    grid = 'O..-..O---..O-O..-.3.--.O.--.6.-..O-O..---O..-..O'
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']
    features = [
        LittleKillerFeature(24, (3, 1), (-1, 1)),
        LittleKillerFeature(31, (5, 1), (-1, 1)),
        LittleKillerFeature(34, (7, 1), (-1, 1)),
        LittleKillerFeature(27, (3, 9), (1, -1)),
        LittleKillerFeature(15, (5, 9), (1, -1)),
        LittleKillerFeature(12, (7, 9), (1, -1)),

        LittleKillerFeature(61, (1, 3), (1, 1)),
        LittleKillerFeature(10, (1, 5), (1, 1)),
        LittleKillerFeature(6,  (1, 7), (1, 1)),

        LittleKillerFeature(8,  (9, 3), (-1, -1)),
        LittleKillerFeature(29, (9, 5), (-1, -1)),
        LittleKillerFeature(29, (9, 7), (-1, -1)),
        DrawCircleFeature(circles),
    ]
    return puzzle, features


def puzzle_14() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/385478394
    grid = '1...O...O.2.--..3-O..-4..-O...5...O...O.6...O..-7..--.8...O.O...9'
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']
    print(circles)
    features = [
        *SandwichFeature.create_all(House.Type.COLUMN, [4, None, 18, None, 20, None, 19, None, 2]),
        *SandwichFeature.create_all(House.Type.ROW, [11, None, 10, None, 8, None, 32, None, 22]),
        DrawCircleFeature(circles),
    ]
    return puzzle, features


def puzzle_15() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/436768313
    grid = '4..O....6----..O..O..O..7..4-.O.-2..1..O..O..O..----5..-O.1'
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']
    all_across = {5:  [(2, 5), (3, 2), (3, 7), (7, 2), (8, 5)],
                  10: [(2, 4), (7, 7), (8, 4)]}
    all_down = {5: [(6, 6), (7, 7)],
                10: [(2, 3), (2, 7), (3, 4), (4, 1), (5, 9), (7, 3)]}

    features = [
        DrawCircleFeature(circles),
        *XVFeature.create(across=all_across, down=all_down, all_listed=False)
    ]
    return puzzle, features


def puzzle_16() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/395396854
    grid = 'O..-O..-----..O-..1--.O...O-3..-O..-----..O.O...O'
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']
    features = [
        DrawCircleFeature(circles),
        *KropkeDotFeature.create("3,1,N,E,N,E", color='white'),
        *KropkeDotFeature.create("5,1,E,N,E,N,E,N,E,N", color='black'),
        *KropkeDotFeature.create("6,3,N,E,N,E,N,E", color='white'),
        *KropkeDotFeature.create("8,6,E,N,E,N", color='black'),
        *KropkeDotFeature.create("9,8,N,E", color="white"),
        *KropkeDotFeature.create("9,1,E", color="white"),
        *KropkeDotFeature.create("1,8,E", color='black'),
    ]
    return puzzle, features


def puzzle_17() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/272936289
    grid = '1....O..O--.O.---O..-..O..6--O..---.1.-.O.---O.O..6'
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']

    features = [
        DrawCircleFeature(circles),
        ArrowSumFeature("1,2,E,E"),
        ArrowSumFeature("3,7,SW,SW,SW"),
        ArrowSumFeature("4,4,NW,NW"),
        ArrowSumFeature("4,5,NW,NE,SE"),
        ArrowSumFeature("5,4,NW,SW,SE"),
        ArrowSumFeature("5,6,SE,NE,NW"),
        ArrowSumFeature("6,5,SE,SW,NW"),
        ArrowSumFeature("6,6,SE,SE"),
        ArrowSumFeature("8,9,N,N"),
    ]
    return puzzle, features


def puzzle_18() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/725978489
    grid = "2....O3.O.3.-1O...4-.97O..1....O-.O.--..9-.O....6..3.O....7.67...O..8"
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']

    def create_between_lines(row: int, column: int) -> Feature:
        squares = []
        while row <= 9 and column <= 9:
            squares.append((row, column))
            row, column = row + 1, column + 1
        return ExtremeEndpointsFeature(squares)

    features = [
        DrawCircleFeature(circles),
        create_between_lines(5, 1),
        create_between_lines(2, 1),
        create_between_lines(1, 2),
        create_between_lines(1, 3),
        create_between_lines(1, 4),

    ]
    return puzzle, features


def puzzle_19() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/548565193
    grid = ".R.1O9.G.G.O.R.6.R.9.G.G.O.9.G...G.1OR..G..RO6.G...G.2.O.G.G.7.R.8.R.O.G.G.8O4.R."
    grid = grid.replace('-', '...')
    puzzle = grid.replace('O', '.')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']
    reds = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'R']
    greens = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'G']
    circles.append((5, 5))
    circles.sort()

    features = [
        *LocalMinOrMaxFeature.create(reds=reds, greens=greens),
        DrawCircleFeature(circles),
    ]
    return puzzle, features


def puzzle_20() -> tuple[str, Sequence[Feature]]:
    # http://tinyurl.com/421265171
    grid = "..O-O..---O..-..O----.O.----O..-..O---..O..O-"
    grid = grid.replace('-', '...')
    circles = [(r, c) for (r, c), letter in zip(itertools.product(range(1, 10), repeat=2), grid) if letter == 'O']

    features = [
        DrawCircleFeature(circles),
        ValuesAroundIntersectionFeature(top_left=(1, 1), values=(5, 7, 9)),
        ValuesAroundIntersectionFeature(top_left=(1, 8), values=(1, 2, 3, 4)),
        ValuesAroundIntersectionFeature(top_left=(2, 2), values=(1, 3, 4)),
        ValuesAroundIntersectionFeature(top_left=(2, 4), values=(4, 5)),
        ValuesAroundIntersectionFeature(top_left=(3, 3), values=(7, 8, 9)),
        ValuesAroundIntersectionFeature(top_left=(3, 6), values=(1, 2, 3)),
        ValuesAroundIntersectionFeature(top_left=(4, 5), values=(4, 5, 6)),
        ValuesAroundIntersectionFeature(top_left=(4, 8), values=(1, 4)),
        ValuesAroundIntersectionFeature(top_left=(5, 1), values=(4, 5)),
        ValuesAroundIntersectionFeature(top_left=(5, 4), values=(7, 8, 9)),
        ValuesAroundIntersectionFeature(top_left=(6, 3), values=(1, 2, 3)),
        ValuesAroundIntersectionFeature(top_left=(6, 6), values=(4, 5, 7)),
        ValuesAroundIntersectionFeature(top_left=(7, 5), values=(3, 7)),
        ValuesAroundIntersectionFeature(top_left=(7, 7), values=(1, 8, 9)),
        ValuesAroundIntersectionFeature(top_left=(8, 1), values=(4, 7, 8, 9)),
        ValuesAroundIntersectionFeature(top_left=(8, 8), values=(2, 5, 6)),
    ]
    return ' ' * 81, features


def run() -> None:
    puzzles = [
        puzzle_1, puzzle_2, puzzle_3, puzzle_4, puzzle_5, puzzle_6, puzzle_7, puzzle_8,
        puzzle_9,
        puzzle_10,
        puzzle_11, puzzle_12,
        puzzle_13,
        puzzle_14, puzzle_15,
        puzzle_16,
        puzzle_17, puzzle_18,
        puzzle_19,
        puzzle_20,
        ]
    for puzzle in puzzles:
        grid, features = puzzle()
        print()
        print(f'---------- { puzzle.__name__} ----------')
        result = Sudoku().solve(grid, features=features)
        assert result, f'Puzzle {puzzle} failed'


if __name__ == '__main__':
    run()
