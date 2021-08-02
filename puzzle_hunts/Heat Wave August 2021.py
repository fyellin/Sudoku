import datetime
import itertools
import re
from typing import Sequence, Union

from draw_context import DrawContext
from feature import Feature, Square
from features.thermometer import SlowThermometerFeature, TestSlowThermometerFeature, ThermometerFeature
from human_sudoku import Sudoku


class DrawLetterFeature(Feature):
    square: Square

    def __init__(self, square: Union[str, Square], *, name: str):
        super().__init__(name=name)
        self.square = self.parse_square(square)

    def draw(self, context: DrawContext) -> None:
        cell = self @ self.square
        if not cell.is_known:
            y, x = self.square
            context.draw_text(x + .5, y + .5, self.name,
                              fontsize=20, va='center', ha='center', color='blue')
        elif context.done:
            print(f'{self.name} =', cell.known_value)


def puzzle_1() -> tuple[str, Sequence[Feature]]:
    grid = '...679...XX3x64x29x1XX...841...'.replace('x', '.......').replace('X', '.........')
    thermometers = [
        "11,E", "11,S", "11,SE", "13,W", "13,SW", "13,SE",
        "23,W", "23,S", "23,SE", "25,E", "25,W", "25,S", "25,SE", "25,SW",
        "31,N", "31,NE", "31,SE",
        "32,N", "32,E", "32,SE",
        "44,SW", "44,W", "44,NW", "44,N", "44,NE", "55,N"
    ]
    features = [
        *[ThermometerFeature(thermometer + suffix) for thermometer in thermometers
          for suffix in ('', ',R', ',R,R', ',R,R,R')],
        DrawLetterFeature("99", name="A")
    ]
    return grid, features


def puzzle_2() -> tuple[str, Sequence[Feature]]:
    grid = 'XXX.1.....3.....4.....2.....9.XXX'.replace("X", '---').replace('-', '...')
    thermometers = [
        '33,W,SW', '33,W,W', '33,NW,W', '33,NW,N', '33,N,N', '33,N,NE',
    ]
    thermometers2 = [
        '52,NE,E', '52,NE,SE', '45,NE,NW', '46,N,Nw'
    ]
    features = [
        *[ThermometerFeature(thermometer + suffix) for thermometer in thermometers
          for suffix in ('', ',R', ',R,R', ',R,R,R')],
        *[ThermometerFeature(thermometer + suffix) for thermometer in thermometers2
          for suffix in ('', ',R,R')],
        DrawLetterFeature("16", name="B")
    ]
    return grid, features


def puzzle_3() -> tuple[str, Sequence[Feature]]:
    grid = "9..4.3-1..-------..66..-..57..--X--..1-6.9..4"
    grid = grid.replace("X", '---').replace('-', '...')
    thermometers = [
        '19,S,S,SW', '25,S,SW,SW', '25,S,SE,SW', '42,N,N,N', "52,NE,N,N", "52,SE,S,S",
    ]
    features = [
        *[ThermometerFeature(thermometer + suffix) for thermometer in thermometers
          for suffix in ('', ',R,R')],
        DrawLetterFeature("34", name="C")
    ]
    return grid, features


def puzzle_4() -> tuple[str, Sequence[Feature]]:
    grid = "6..--XX8..-..2X5..-..3XX--..6"
    grid = grid.replace("X", '---').replace('-', '...')
    thermometers = [
        '22,E,E,E,E', '22,S,S,S,S', '22,SE,E,E,NE', '22,SE,S,S,SW', '19,SW,W,S,SW'
    ]
    features = [
        *[ThermometerFeature(thermometer + suffix) for thermometer in thermometers
          for suffix in ('', ',R,R')],
        DrawLetterFeature("55", name="D")
    ]
    return grid, features


def puzzle_5() -> tuple[str, Sequence[Feature]]:
    grid = "X--1.."
    grid = grid.replace("X", '---').replace('-', '...')
    thermometers = [
        "11,S,S,SE,E,E", "46,E,E,NE,N,N"
    ]
    features = [
        *[ThermometerFeature(thermometer + suffix) for thermometer in thermometers
          for suffix in ('', ',R,R')],
        ThermometerFeature("64,NE,E,NW,NW,E"),
        ThermometerFeature("59,N,NW,W,NW,NW"),
        ThermometerFeature("95,NW,NW,W,NW,N"),
        DrawLetterFeature("92", name="E")
    ]
    return grid, features


def puzzle_6() -> tuple[str, Sequence[Feature]]:
    grid = "5..1....4XXX--..8X--..6XX"
    grid = grid.replace("X", '---').replace('-', '...')
    features = [
        ThermometerFeature("43,SE,E,NE,N,N,N"),
        ThermometerFeature("71,E,N,E,N,W,N"),
        ThermometerFeature("71,E,E,SE,E,E,SE"),
        ThermometerFeature("76,NE,N,N,N,N,NE"),
        DrawLetterFeature("34", name="F")
    ]
    return grid, features


def puzzle_7() -> tuple[str, Sequence[Feature]]:
    grid = "--..2.9..3.-XX.6.--XXX6..--"
    grid = grid.replace("X", '---').replace('-', '...')
    features = [
        SlowThermometerFeature("51,N,NE,E,SE,S,SW,W"),
        SlowThermometerFeature("46,W,NW,N,NE,E,SE,S"),
        SlowThermometerFeature(temp := "77,E,NE,N,NW,W,SW,S"),
        SlowThermometerFeature(temp + ",R"),
        DrawLetterFeature("68", name="G")
    ]

    return grid, features


def puzzle_8() -> tuple[str, Sequence[Feature]]:
    grid = "X5..--X4..--XX7..--X-74...2"
    grid = grid.replace("X", '---').replace('-', '...')
    features = [
        *TestSlowThermometerFeature.create("91,NE,NE,NE,NE,NE,NE,NE,NE"),
        *TestSlowThermometerFeature.create("24,NE,E,SE,SE,SE,S,SW,SW"),
        *TestSlowThermometerFeature.create("35,S,SW,SW,SW,SE,NE,NE,NE"),
        *TestSlowThermometerFeature.create("76,NE,NE,NW,NW,NW,SW,SW,SW"),
        DrawLetterFeature("14", name="H")
    ]

    return grid, features


def puzzle_9() -> tuple[str, Sequence[Feature]]:
    grid = "XXX-..1-X-9..-XXX"
    grid = grid.replace("X", '---').replace('-', '...')
    features = [
        SlowThermometerFeature("82,SW"),
        SlowThermometerFeature("13,SE,SE"),
        SlowThermometerFeature("61,SE,SE,SE"),
        SlowThermometerFeature("85,E,SE,E,NE"),
        SlowThermometerFeature("78,NE,N,N,NW,NW"),
        SlowThermometerFeature("42,E,SE,E,E,SE,E"),
        SlowThermometerFeature("96,NE,N,NW,W,SW,W,NW,N"),
        SlowThermometerFeature("14,SW,S,SE,E,NE,E,SE"),
        DrawLetterFeature("32", name="I")
    ]
    return grid, features


def puzzle_10() -> tuple[str, Sequence[Feature]]:
    info = dict(A=5, B=4, C=9, D=9, E=2, F=3, G=2, H=5, I=6)
    # puzzle_1 = '581679243763214598249538617357192486418756932926483751135967824874325169692841375'
    # puzzle_2 = '296534781874291356531678249417962835359847612628315497942753168763189524185426973'
    # puzzle_3 = '985473162176852943243961587532718496618394725794526318827145639469237851351689274'
    # puzzle_4 = '648732951715689324932451687863547192274193865591826743187265439356974218429318576'
    # puzzle_5 = '265371948389456127417892536178963452694725381532148769743689215851237694926514873'
    # puzzle_6 = '563178294712946835498325761381754629976231458245689317134892576829567143657413982'
    # puzzle_7 = '516978432794632851823514976249351768168427395375869124951786243432195687687243519'
    # puzzle_8 = '867534129513279486942816753438927615691358247275461398754682931326195874189743562'
    # puzzle_9 = '583129476971654823462378951734281695298567134156943782825496317617835249349712568'

    class SpecialFeature(Feature):
        letter: str
        square: Square

        def __init__(self, letter: str, square: Square):
            super().__init__(name=f'Letter {letter}')
            self.letter = letter
            self.square = square

        def start(self) -> None:
            (self @ self.square).set_value_to(info[self.letter])

        def draw(self, context: DrawContext) -> None:
            y, x = self.square
            context.draw_text(x + .5, y + .5, self.letter,
                              fontsize=20, va='center', ha='center', color='blue')

    grid = "....B.....A.....C...6...............D...E...F...............8...G.....I.....H...."
    locations = {letter: location for letter, location in zip(grid, itertools.product(range(1, 10), repeat=2))
                 if 'A' <= letter <= 'I'}
    grid = re.sub(r'[A-Z]', r'.', grid)

    features = [
        ThermometerFeature("21,SE,NE,NW"), ThermometerFeature("14,S,E,E"), ThermometerFeature("38,NW,NE,SE"),
        ThermometerFeature("61,E,N,N"), ThermometerFeature("45,SW,SE,NE"), ThermometerFeature("49,W,S,S"),
        ThermometerFeature("72,SE,SW,NW"), ThermometerFeature("96,N,W,W"), ThermometerFeature("87,SE,NE,NW"),
        *[SpecialFeature(letter, location) for letter, location in locations.items()]
    ]
    return grid, features


def run() -> None:
    start = datetime.datetime.now()

    puzzles = [puzzle_1,
               puzzle_2,
               puzzle_3, puzzle_4, puzzle_5, puzzle_6, puzzle_7, puzzle_8, puzzle_9, puzzle_10
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


if __name__ == '__main__':
    run()
