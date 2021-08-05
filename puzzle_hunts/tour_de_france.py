import datetime
import itertools
from typing import Iterable, Sequence, Set, Tuple, cast

from cell import House
from draw_context import DrawContext
from feature import Feature, Square
from features.possibilities_feature import GroupedPossibilitiesFeature, HousePossibilitiesFeature
from human_sudoku import Sudoku


class LocateOneFeature(HousePossibilitiesFeature):
    """Both edges gives the location of the 1 in the row or column"""
    def __init__(self, htype: House.Type, index: int):
        super().__init__(htype, index, prefix="Location1")

    def match(self, permutation: tuple[int, ...]) -> bool:
        return permutation[0] + permutation[8] == 10 and permutation[permutation[0] - 1] == 1

    def draw(self, context: DrawContext) -> None:
        context.draw_rectangles([self.squares[0], self.squares[8]])


class PainInTheButtFeature(GroupedPossibilitiesFeature):
    """Tour de France Puzzle 2.
    For the values 1, 2, and 3 in row 8.
    In rows 7 and 6, the 1 must be 1 up and 1 right.  The 2 must be 1 up and 2 right.  The 3, etc.
    """
    value: int

    def __init__(self, value: int):
        self.value = value
        squares = [(row, column) for row in range(6, 9) for column in range(1, 10)]
        super().__init__(squares, name=f"Pain{value}", neighbors=True)

    def get_possibilities(self) -> Iterable[Tuple[Set[int], ...]]:
        others = set(range(1, 10)) - {self.value}
        prototype = [others] * 27
        for column in range(1, 10):
            delta = -self.value
            if 1 <= column + 2 * delta <= 9:
                possibility = prototype[:]
                possibility[-1 + column] = {self.value}
                possibility[-1 + 9 + column + delta] = {self.value}
                possibility[-1 + 18 + column + 2 * delta] = {self.value}
                yield tuple(possibility)

    def check_special(self) -> bool:
        if self.value == 1 and not self.grid.matrix[8, 6].is_known:
            self.grid.matrix[8, 6].set_value_to(1, show=True)
            return True
        return False


class PainInTheButtFeatureX(GroupedPossibilitiesFeature):
    """Like PainInTheButt.
    There is a diagonal of three ones going up and to the right somewhere in rows 2,3,4,5
    """
    def __init__(self) -> None:
        squares = [(row, column) for row in range(2, 6) for column in range(1, 10)]
        super().__init__(squares, name=f"PainX", neighbors=True)

    def get_possibilities(self) -> Iterable[Tuple[Set[int], ...]]:
        others = set(range(1, 10)) - {1}
        prototype = [others] * 27
        normal = [set(range(1, 10))] * 9
        for column in range(1, 10):
            if 1 <= column - 2 <= 9:
                possibility = prototype[:]
                possibility[column - 1] = {1}
                possibility[column + 7] = {1}
                possibility[column + 15] = {1}
                yield tuple(possibility + normal)
                yield tuple(normal + possibility)

    def check_special(self) -> bool:
        square = 3, 2
        if not self.grid.matrix[square].is_known:
            print("SPECIAL")
            # // self.grid.matrix[square].set_value_to(1, show=True)
            # // return True
        return False


class DrawCircleFeature(Feature):
    squares: Sequence[Square]

    def __init__(self, squares: Sequence[Square]):
        super().__init__()
        self.squares = squares

    def draw(self, context: DrawContext) -> None:
        for row, column in self.squares:
            context.draw_circle((column + .5, row + .5), radius=.5, fill=False, color='blue')
        if self.grid.is_solved():
            puzzle = ''.join(str(self.grid.matrix[square].known_value) if square in self.squares else '.'
                             # Type system seems not to like itertools.product
                             for square in cast(Iterable[Square], itertools.product(range(1, 10), repeat=2)))
            print(f'previous = "{puzzle}"')


def merge(p1: str, p2: str) -> str:
    assert len(p1) == len(p2) == 81
    assert(p1[i] == '.' or p2[i] == '.' or p1[i] == p2[i] for i in range(81))
    result = ((y if x == '.' else x) for x, y in zip(p1, p2))
    return ''.join(result)


def tour_puzzle_one() -> tuple[str, Sequence[Feature]]:
    features = [
        *(LocateOneFeature(House.Type.ROW, i) for i in range(2, 9)),
        *(LocateOneFeature(House.Type.COLUMN, i) for i in range(2, 9)),
        DrawCircleFeature(((1, 2), (1, 4), (2, 9), (4, 8), (5, 2), (5, 3), (5, 7), (5, 8),
                           (6, 2), (8, 1), (9, 6,), (9, 8)))
    ]
    puzzle = '.......................9.......5.3.....4.37....3.8.......36...........5..........'
    return puzzle, features


def tour_puzzle_two() -> tuple[str, Sequence[Feature]]:
    previous = ".4.8.............7................9..85...76..7................4.............4.7."
    puzzle = "X,,6--XXXXX--6...5.--".replace('X', '---').replace('-', '...')
    puzzle = merge(previous, puzzle)
    features: list[Feature] = [PainInTheButtFeature(i) for i in range(1, 4)]
    features.append(PainInTheButtFeatureX())
    return puzzle, features


if __name__ == '__main__':
    start = datetime.datetime.now()
    grid, features = tour_puzzle_two()
    Sudoku().solve(grid, features=features)
    end = datetime.datetime.now()
    print(end - start)
