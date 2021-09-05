import datetime
import itertools
from collections.abc import Iterator, Sequence, Mapping

from cell import House
from draw_context import DrawContext
from feature import Feature, Square
from features.chess_move import KnightsMoveFeature
from features.features import ArrowSumFeature, DrawOnlyFeature, KillerCageFeature, LimitedValuesFeature, \
    LocalMinOrMaxFeature, \
    PalindromeFeature
from features.possibilities_feature import PossibilitiesFeature, Possibility
from features.sandwich_feature import SandwichFeature
from features.thermometer import ThermometerFeature
from human_sudoku import Sudoku

BLANK_GRID = ' ' * 81

def puzzle_1() -> tuple[str, Sequence[Feature]]:
    features = [
        KillerCageFeature(10, "11,E"),
        KillerCageFeature(5, "22, S"),
        KillerCageFeature(18, "27, S, W"),
        KillerCageFeature(6, "28, S, E"),
        KillerCageFeature(17, "35, S"),
        KillerCageFeature(10, "41, S, S"),
        KillerCageFeature(12, "59,S"),
        KillerCageFeature(12, "62, E, E"),
        KillerCageFeature(10, "65, S"),
        KillerCageFeature(11, "66, E, E"),
        KillerCageFeature(14, "91, E"),
        PalindromeFeature("22,SE,NE,SE,NE,SE,E,S,SW,SW,SW,W,SE,SE,NE,SE,E")
    ]
    return BLANK_GRID, features


def puzzle_2() -> tuple[str, Sequence[Feature]]:
    features = [
        ArrowSumFeature("14,SW,SW,S,S,SE"),
        ArrowSumFeature("35,N,N"),
        ArrowSumFeature("35,S,S"),
        ArrowSumFeature("59,N,N"),
        ArrowSumFeature("67,NE,N,N,NW,NW"),
        ArrowSumFeature("75,W,N"),
        ArrowSumFeature("75,E,N"),
        ArrowSumFeature("77,SW,W,W,NW"),
        KillerCageFeature(30, "51,S,S,S,62"),
        KillerCageFeature(30, "59,S,S,S,68"),
        LocalMinOrMaxFeature(reds="44,46", diagonals=True)
    ]
    return BLANK_GRID, features


def puzzle_3() -> tuple[str, Sequence[Feature]]:
    def bunny() -> Iterator[Possibility]:
        for x1, x2, a1, a2, a3, a4 in itertools.product(range(1, 10), repeat=6):
            if x1 + x2 == 6 + a1 + a2 + a3 + a4:
                yield x1, x2, a1, a2, a3, a4

    def drawer(context: DrawContext):
        context.draw_rectangles(Feature.parse_squares("14,S,SW,E,E,S,NE,N,N,37,S"), facecolor="#D2D2D2")
        context.draw_rectangles(Feature.parse_squares("56,E,E,S,W"), facecolor="#D2E39E")
        context.draw_rectangles(Feature.parse_squares("66,SW,E,E,SW,W,W,SW,E,E,E"), facecolor="#F2BC97")

    features = [
        LimitedValuesFeature("74", {9}),
        LimitedValuesFeature("56", {9}),
        LimitedValuesFeature("57,58,67,68", {6, 7, 8, 9}),
        LimitedValuesFeature("93,66,77", {1}),
        ThermometerFeature("95,N", color='yellow'),
        ThermometerFeature("37,N,SE,N,NW", color='yellow'),
        KnightsMoveFeature(),
        PossibilitiesFeature("15,25,33,37,45,47", neighbors=True, possibility_function=bunny),
        DrawOnlyFeature(drawer)
    ]
    return BLANK_GRID, features


def puzzle_4() -> None:
    class CopyBoxFeature(Feature):
        def __init__(self, grid: Mapping[Square, int], box: int) -> None:
            super().__init__()
            self.puzzle = grid
            self.box = box

        def start(self):
            for square in self.get_house_squares(House.Type.BOX, self.box):
                (self @ square).set_value_to(self.puzzle[square])

    def puzzle_to_grid(puzzle: str) -> Mapping[Square, int]:
        return {(row, column): int(letter)
                for (row, column), letter in zip(itertools.product(range(1, 10), repeat=2), puzzle)}
    puzzle_1 = '285913746743256918916784532378691425129845673654372189462537891831469257597128364'
    puzzle_2 = '357921684482356197916478523725819436834567219691243875273685941549132768168794352'
    puzzle_3 = '534682791187593462962714358793846215218359674456271983845927136329168547671435829'
    grid1 = puzzle_to_grid(puzzle_1)
    grid2 = puzzle_to_grid(puzzle_2)
    grid3 = puzzle_to_grid(puzzle_3)

    for a, b, c in [(4, 8, 6)]: # itertools.permutations(range(1, 10), 3):
        features = [
            CopyBoxFeature(grid1, a),
            CopyBoxFeature(grid2, b),
            CopyBoxFeature(grid3, c),
            *SandwichFeature.create_all(House.Type.ROW, [None, 5, 15, None, None, None, 28, 12, None]),
            *SandwichFeature.create_all(House.Type.COLUMN, [None, 6, 14, None, None, None, 20, 19, None]),
        ]
        sudoku = Sudoku()
        try:
            print('******', a, b, c)
            result = sudoku.solve(BLANK_GRID, features=features, initial_only=False)
            if result:
                break
        except:
            continue


def run() -> None:
    start = datetime.datetime.now()

    puzzles = [
        puzzle_1,
        puzzle_2,
        puzzle_3,
    ]
    for puzzle in puzzles:
        grid, features = puzzle()
        print()
        print(f'---------- {puzzle.__name__} ----------')
        sudoku = Sudoku()
        result = sudoku.solve(grid, features=features, initial_only=False)
        if result:
            name = puzzle.__name__
            print(f'    {name} = \'{sudoku.get_grid()}\'')
        assert result

    end = datetime.datetime.now()
    print(end - start)
    puzzle_4()


if __name__ == '__main__':
    run()
