import datetime
import itertools
from collections.abc import Iterator, Sequence, Mapping

from cell import House
from draw_context import DrawContext
from feature import Feature, Square
from features.chess_move import KnightsMoveFeature
from features.features import ArrowSumFeature, DrawOnlyFeature, KillerCageFeature, KropkeDotFeature, \
    LimitedValuesFeature, \
    LocalMinOrMaxFeature, \
    LockoutLineFeature, PalindromeFeature, XVFeature
from features.possibilities_feature import PossibilitiesFeature, Possibility
from features.sandwich_feature import SandwichFeature
from features.thermometer import ThermometerFeature
from human_sudoku import Sudoku

BLANK_GRID = ' ' * 81

def puzzle_1() -> tuple[str, Sequence[Feature]]:
    features = [
        LockoutLineFeature.between((1, 1), (9, 9)),
        LockoutLineFeature("14,S,E,S,E,S,e,S,E,S,E"),
        LockoutLineFeature("19,W,W,SE"),
        LockoutLineFeature("29,S,S"),
        LockoutLineFeature("31,E,SE,SE,SE,SE"),
        LockoutLineFeature("41,E,S,E,S,E,S,E,S,E,S"),
        LockoutLineFeature("73,NW,SW,S,SE,NE"),
        XVFeature(across={5: "57,68,78", 10: "26,73,95"},
                  down={5:"84", 10: "42"},
                  all_listed=False)
    ]
    grid = "--2..XXXXXX--.6.X".replace("X", "---").replace('-', '...')
    return grid, features


def puzzle_2():
    features = [
        LockoutLineFeature("15,S,S"),
        LockoutLineFeature("18,S,E"),
        LockoutLineFeature("23,NW,SW,SE"),
        LockoutLineFeature("36,S,E"),
        LockoutLineFeature("45,SW,W,W"),
        LockoutLineFeature("58,W,W,SW"),
        LockoutLineFeature("75,S,S"),
        LockoutLineFeature("81,E,S"),
        KillerCageFeature(14, "15,S,S"),
        KillerCageFeature(18, "18,E,S"),
        KillerCageFeature(19, "23,S,W"),
        KillerCageFeature(15, "63,E,S"),
        KillerCageFeature(14, "75,S,S"),
        KillerCageFeature(19, "81,S,E")
    ]
    grid = "-6..-.7.----9..XXX..8--X-..7-".replace("X", "---").replace('-', '...')
    return grid, features


def puzzle_3():
    lines = ["15,W,W,SW", "15,S,S,S", "22,SW,S,S,S", "28, S,S,S", "45,E,S,S", "55,W,S", "61,E,E,E", "64,SE,SE,NE,NW",
             "66, E, E", "78,E,S", "82,E,E", "85,S,E,E,N,E", "91,E,E"]
    features = [
        *[LockoutLineFeature(x) for x in lines],
        KropkeDotFeature("45,S", color="black"),
        KropkeDotFeature("64,E,E", color="black"),
        KropkeDotFeature("82,S", color="white"),
        KropkeDotFeature("83,S", color="white"),
        KropkeDotFeature("23,S", color="white"),
        KropkeDotFeature("24,S", color="white"),
    ]
    return BLANK_GRID, features



def foo():
    # Phase 1
    puzzle_1 = '153867249782914653946532178214653987597481326638729514469175832375298461821346795'
    # Jailbreak
    puzzle_2 = '825691437379425618146738952964172385531849726287563194758314269412986573693257841'


def run() -> None:
    start = datetime.datetime.now()

    puzzles = [
        puzzle_3,
    ]
    result = "528914673173286549964375812847569231256137984319842765435698127782451396691723458"
    for puzzle in puzzles:
        grid, features = puzzle()
        grid = grid.replace("X", "---").replace('-', '...')
        print()
        print(f'---------- {puzzle.__name__} ----------')
        sudoku = Sudoku()
        result = sudoku.solve(grid, features=features, initial_only=False, verify=result)
        if result:
            name = puzzle.__name__
            print(f'    {name} = \'{sudoku.get_grid()}\'')
        assert result

    end = datetime.datetime.now()
    print(end - start)



if __name__ == '__main__':
    run()
