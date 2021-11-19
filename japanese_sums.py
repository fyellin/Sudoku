from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence, Hashable
from ctypes import Union
from functools import cache
from itertools import combinations, islice, permutations, product
from typing import Optional, cast

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from tools.dancing_links import DancingLinks

WHITE = 0
GREEN = 1
PURPLE = 2
BLUE = 3
YELLOW = 4
RED = 5

class JapaneseSums:
    ENCODING_A: tuple[tuple[str]] = tuple(islice(combinations('abcdef', 3), 10))
    ENCODING_B: tuple[tuple[str]] = list(tuple(x for x in 'abcdef' if x not in encoding) for encoding in ENCODING_A)

    grid: list[list[str]]
    size: int
    is_complicated: Optional[tuple[int, int]]
    maximum: int

    def __init__(self, size: int  = 5, is_complicated: Union[bool, tuple[int, int]] = False):
        self.size = size
        if not is_complicated:
            self.is_complicated = None
        elif isinstance(is_complicated, bool):
            self.is_complicated = (3, 3)
        else:
            self.is_complicated = cast(tuple[int, int], is_complicated)
        self.maximum = (size * (size + 1)) // 2
        self.totals = self.__get_totals()

    def solve(self, puzzle: tuple[tuple[Union[tuple[int,...], int],...],...],
              colors: Optional[tuple[tuple[Optional[int]], ...]] = None, *,
              dump: bool = False,
              draw_grid: bool = False,
              diagonal: Optional[tuple[int, int, int]] = None) -> None:
        constraints: dict[Hashable, list[str]] = {}
        if colors is None:
            temp = [GREEN] * self.size
            colors = [temp, temp]
        self.all_colors = list((set(colors[0]) | set(colors[1])) - {None})
        self.diagonal = diagonal
        self.puzzle = puzzle
        self.colors = colors

        assert len(puzzle[0]) == len(puzzle[1]) == self.size
        value1 = sum(x if isinstance(x, int) else sum(x) for x in puzzle[0])
        value2 = sum(x if isinstance(x, int) else sum(x) for x in puzzle[1])
        assert value1 == value2

        if draw_grid:
            self.draw_grid()

        def add_all_constraints_for_line(is_row: bool, line_number: int,
                                         possible_rows: Sequence[tuple[tuple[int, int], ...]]):
            for possible_row in possible_rows:
                self.__add_constraint_for_line_possibility(constraints, line_number, is_row, possible_row)
                if self.is_complicated:
                    possible_row = tuple((value, int(not color)) for value, color in possible_row)
                    self.__add_constraint_for_line_possibility(constraints, line_number, is_row, possible_row)

        for direction, color, is_row in zip(puzzle, colors, (True, False)):
            for line_number, (this_info, this_color) in enumerate(zip(direction, color), start=1):
                if isinstance(this_info, int):
                    this_info = (this_info,)
                possible_rows = self.get_possible_rows(this_info, this_color)
                add_all_constraints_for_line(is_row, line_number, possible_rows)

        for row, col, value in product(range(1, self.size + 1), repeat=3):
            for encoding, name in ((self.ENCODING_A, 'XRow'), (self.ENCODING_B, 'XCol')):
                constraint = []
                self.__add_value_to_constraint(constraint, row, col, value, encoding)
                constraints[name, (row, col), value] = constraint

        if dump:
            with open("/tmp/junk", "w") as file:
                for name, constraint in constraints.items():
                    print(f"{name} {constraint}", file=file)

        links = DancingLinks(constraints, row_printer=self.handle_solution)
        links.solve(recursive=False, debug=3)

    @cache
    def get_possible_rows(self, row: tuple[int], color: Optional[int]) ->  Sequence[tuple[tuple[int, int], ...]]:
        length = len(row)
        if color is not None:
            color_list = [[color for _ in range(length)]]
        else:
            color_list = [colors for colors in product(self.all_colors, repeat=length) if len(set(colors)) > 1]

        if length == 1 and (delta := (self.maximum - row[0])) < 3:
            assert color is not None
            if delta == 0:
                return [((0, color),) * self.size]
            else:
                temp = ((0, color),) * (self.size - 1)
                return [((delta, WHITE), *temp), (*temp, (delta, WHITE))]

        result = []
        groups_list = self.__get_group_lists(row)
        for groups in groups_list:
            product_args = [list(permutations(group)) for group in groups]
            unused_item_count = self.size - sum(len(i) for i in groups)
            for colors in color_list:
                bitmap = sum((1 << i) for i in range(1, length) if colors[i - 1] != colors[i]) + 1 + (1 << length)
                for hole_sizes in self.__get_hole_sizes_list(length + 1, unused_item_count, bitmap):
                    for groups_all_permutations in product(*product_args):
                        temp = [(0, WHITE)] * hole_sizes[0]
                        for group, color, hole in zip(groups_all_permutations, colors, hole_sizes[1:]):
                            temp.extend((v, color) for v in group)
                            temp.extend([(0, WHITE)] * hole)
                        result.append(temp)
        return result

    @cache
    def __get_hole_sizes_list(self, holes: int, fill_count: int, zero_mask: int) -> Sequence[tuple[int, ...]]:
        if holes == 0:
            return [()] if fill_count == 0 else []
        if holes == 1:
            return [(fill_count,)] if (fill_count > 0 or zero_mask & 1) else []
        result = []
        for first_hole in range((zero_mask & 1) ^ 1, fill_count + 1):
            result.extend((first_hole, *remainder)
                          for remainder in self.__get_hole_sizes_list(holes - 1, fill_count - first_hole, zero_mask >> 1))
        result.sort()
        return result

    @cache
    def __get_group_lists(self, row: tuple[int]) -> Sequence[tuple[tuple[int]]]:
        def internal(row: tuple[int], available: set[int]) -> Iterable[tuple[tuple[int]]]:
            if not row:
                yield ()
                return
            first, *rest = row
            for int_list in self.totals[first]:
                if all(x in available for x in int_list):
                    yield from ((int_list, *other_int_lists)
                                for other_int_lists in internal(rest, available.difference(int_list)))

        return list(internal(row, set(range(1, self.size + 1))))

    def __add_constraint_for_line_possibility(
            self, constraints: dict[Hashable, list[str]],
            line_number: int, is_row: bool, possible_row: tuple[tuple[int, int]]):
        name, encoding = ('Row', self.ENCODING_A) if is_row else ('Col', self.ENCODING_B)
        constraint = [f"{name}{line_number}_set"]
        for (value, color), (row, col) in zip(possible_row, self.__get_coordinates(line_number, is_row)):
            if self.is_complicated and (row, col) == self.is_complicated and color == WHITE:
                return
            if value != 0:
                self.__add_value_to_constraint(constraint, row, col, value, encoding)
            if len(self.all_colors) == 1:
                constraint.append(f"R{row}C{col}_color_{'X' if (color == 0) == is_row else 'Y'}")
            else:
                constraint.extend(f"R{row}C{col}_color_{code}" for code in encoding[color])
        constraints[name, line_number, tuple(possible_row)] = constraint

    def __add_value_to_constraint(self, constraint: list[str],
                                  row: int, column: int, value: int, encoding) -> None:
        constraint.extend(f"R{row}C{column}_{code}" for code in encoding[value])
        constraint.extend(f"R{row}={value}_{code}" for code in encoding[value])
        constraint.extend(f"C{column}={value}_{code}" for code in encoding[value])
        if self.size == 6:
            box = ((row - 1) // 2) * 3 + (column - 1) // 3 + 1
            constraint.extend(f"B{box}={value}_{code}" for code in encoding[value])
        if self.size == 8:
            box = ((row - 1) // 2) * 3 + (column - 1) // 4 + 1
            constraint.extend(f"B{box}={value}_{code}" for code in encoding[value])

    def __get_totals(self) -> dict[int, list[tuple[int]]]:
        result = defaultdict(list)
        items = list(range(1, self.size + 1))
        for size in range(1, self.size + 1):
            for subset in combinations(items, size):
                result[sum(subset)].append(subset)
        return result

    def __get_coordinates(self, rc, is_row):
        if is_row:
            return [(rc, i) for i in range(1, self.size + 1)]
        else:
            return [(i, rc) for i in range(1, self.size + 1)]

    COLOR_MAP = {WHITE: 'white', GREEN: 'green', PURPLE: 'purple', BLUE: 'blue',
                 YELLOW: 'goldenrod', RED: 'red'}

    def handle_solution(self, results: Sequence[Hashable]) -> None:
        shading = {}
        grid = {}

        for item in cast(Sequence[tuple], results):
            if item[0] == 'Row':
                _, line_number, values = item
                for (value, color), (row, col) in zip(values, self.__get_coordinates(line_number, True)):
                    if value != 0:
                        grid[row, col] = value
                    shading[row, col] = color
            elif item[0] == 'RowX':
                _, (row, col), value = item
                grid[row, col] = value

        if self.diagonal:
            rc_sum, expected_total, magic = self.diagonal
            values = [grid[(row, column)]
                      for row in range(1, self.size + 1) for column in [rc_sum - row] if column >= 1]
            if sum(values) != expected_total:
                return
            length = len(values)
            if sum(1 for i in range(length) for j in range(i + 1, length + 1) if sum(values[i:j]) == magic) != 1:
                return
        self.draw_grid(grid, shading)

    def draw_grid(self, grid: Optional[dict[tuple[int, int], int]] = None,
                  shading: Optional[dict[tuple[int,int], int]] = None) -> None:
        figure, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=100)

        # Set (1,1) as the top-left corner, and (max_column, max_row) as the bottom right.
        axes.axis([1, self.size + 1, self.size + 1, 1])
        axes.axis('equal')
        axes.axis('off')
        figure.tight_layout()

        # Draw the bold outline
        for x in range(1, self.size + 2):
            hbox = self.size if self.size != 6 else 3
            vbox = self.size if self.size != 6 else 2
            width = 3 if (x % hbox) == 1 else 1
            axes.plot([x, x], [1, self.size + 1], linewidth=width, color='black')
            width = 3 if (x % vbox) == 1 else 1
            axes.plot([1, self.size + 1], [x, x], linewidth=width, color='black')

        for direction, color_list, is_row in zip(self.puzzle, self.colors, (True, False)):
            for line_number, (line, color) in enumerate(zip(direction, color_list), start=1):
                label = ' '.join(map(str, line)) if isinstance(line, tuple) else str(line)
                color_name = 'black' if color is None or len(self.all_colors) == 1 else self.COLOR_MAP[color]
                args = dict(color=color_name, rotation=-30)
                if is_row:
                    axes.text(.8, line_number + .5, label, va='center', ha='right', **args)
                else:
                    axes.text(line_number + .5, .8, label, va='bottom', ha='center', **args)

        if grid and shading:
            args = dict(fontsize=15, color='black', weight='heavy')
            for row, col in product(range(1, self.size + 1), repeat=2):
                color = self.COLOR_MAP.get(shading[row, col], 'white')
                plt.gca().add_patch(Rectangle((col, row), width=1, height=1, fc=color))
                axes.text(col + .5, row + .5, str(grid[row, col]), va='center', ha='center', **args)

        plt.show()


def puzzle_book_1():
    temp = JapaneseSums(5)
    puzzle1 = (
        ((5, 4), (3, 4, 1), (4, 5), (1, 3), (3,)),
        ((7,), (5, 1), (4, 3), (4, 3), (6,))
    )
    puzzle2 = (
        ((8, 3), (9,), (2, 1, 3), (15,), (2,)),
        ((5, 3), (4, 2), (11,), (7, 3), (1, 7)),
    )
    puzzle3 = (
        (2, 8, (1, 2), 11, 12),
        (7, 9, 15, 3, 2),
    )
    puzzle4 = ((11, (2, 7), 12, (8, 2), 1), (10, (2, 7), 12, (9, 2), 1))
    temp.solve(puzzle1)
    temp.solve(puzzle2)
    temp.solve(puzzle3)
    temp.solve(puzzle4)

    temp = JapaneseSums(5, is_complicated=True)
    puzzle5 = ((7, 1), (2, 4), 14, (3, 5), (4, 7)), (15, 13, (1, 9), (3, 3), 14)
    puzzle6 = (7, (6, 6), (3, 4, 1), 12, 9), (15, (1, 12), (5, 6), (4, 10), 15)
    puzzle7 = (12, (3, 2, 5), 4, (7, 3), (7, 7)), (13, (4, 10), 10, (5, 6), 14)
    temp.solve(puzzle5)
    temp.solve(puzzle6)
    temp.solve(puzzle7)
    ## Password is 1115244

def puzzle_book_2():
    temp = JapaneseSums(5)
    puzzle2 = (5, (7, 3), (4, 8), (10, 4), 3), (4, (10, 3), (2, 8), (3, 2), 12)
    colors2 =  (PURPLE, PURPLE, PURPLE, None, GREEN), (PURPLE, None, None, None, PURPLE)
    temp.solve(puzzle2, colors2)

    temp = JapaneseSums(5)
    puzzle3 = ((5, 1, 2), (7, 2, 6), (2, 4, 5), (5, 3, 4), (3, 5, 4)), ((9, 3), 10, 15, 10, (7, 4),)
    colors3 = (None, None, None, None, None), (PURPLE, PURPLE, BLUE, PURPLE, PURPLE)
    temp.solve(puzzle3, colors3)

    temp = JapaneseSums(5)
    puzzle4 = (3, (2,4, 5), (7, 8), (3, 12), 12), (6, (2, 4, 5), (6, 9), (1, 14), 9)
    colors4 = (GREEN, None, None, None, RED), (YELLOW, None, None, None, RED)
    temp.solve(puzzle4, colors4)

    temp = JapaneseSums(5)
    puzzle5 = (7, (5, 4, 6), 14, 10, 3), (5, (2, 4, 8), (12, 3), 12, 3)
    colors5 = (RED, None, RED, RED, YELLOW), (YELLOW, None, None, RED, RED, GREEN)
    temp.solve(puzzle5, colors5)

    temp = JapaneseSums(5)
    puzzle6 = ((2, 1), (8, 5), 5, (6, 5), (4, 5)), ((5, 1), (5, 9), 5, (5, 7), (1, 3))
    colors6 = None
    temp.solve(puzzle6, colors6)
    # 414145

def puzzle_book_3():
    puzzle1 = ((), (2, 1), (8, 12), 19, (2, 1), (3, 2)), (3, 11, 11, 3, 8, 14)
    JapaneseSums(6).solve(puzzle1)

    # puzzle2 = (21, (2, 6), (5, 10, 3), (1, 3, 5), (6, 14), 4), (21, 6, (4, 14), (1, 4, 2), (5, 3), 20)
    # JapaneseSums(6).solve(puzzle2)
    #
    # puzzle3 = (20, (2, 4), (1, 3), (4, 11), (9, 9), 9), (10, 21, 4, 3, (5, 9), 20)
    # JapaneseSums(6).solve(puzzle3)
    #
    # puzzle4 = (21, (4, 2), (6, 5), 21, 8, 18), (18, (2, 1, 3), (1, 15), (4, 13), (3, 2, 6), 17)
    # JapaneseSums(6).solve(puzzle4)
    #
    # puzzle5 = (10, 11, 21, (10, 10), (2, 5, 5), (10, 10)), ((7, 13), (1, 2, 4), (12, 1), 19, (2, 15), (11, 8))
    # JapaneseSums(6, is_complicated=(3, 3)).solve(puzzle5)
    #
    # puzzle6 = ((17, 3), 8, 2, (6, 11), (2, 8), (3, 17)), (21, 15, 7, 11, 18, 21)
    # JapaneseSums(6, is_complicated=(3, 4)).solve(puzzle6)
    # # 166245

def puzzle_book_4():
    # puzzle1 = (3, 6, (6, 1), (1, 5), 5, 10), (7, 6, (2, 4), (1, 6), 5, 6)
    # JapaneseSums(6).solve(puzzle1, diagonal=(7, 21, 10))
    # value = 5

    # puzzle2 = (1, 10, (3, 3, 5), (9, 2), 6, 8), (5, 7, (3, 9), (4, 5), 13, 1)
    # colors2 = (GREEN, GREEN, None, None, RED, RED), (RED, RED, None, None, GREEN, GREEN)
    # JapaneseSums(6).solve(puzzle2, colors2)
    # # value = 3

    # puzzle3 = (1, (5, 6), (9, 4), (16, 3), (15, 5), 16), ((3, 1), (10, 5), (7, 6), (17, 4), (9, 5), 13)
    # colors3 = (RED, None, None, None, None, GREEN), (None, None, None, None, None, GREEN)
    # JapaneseSums(6).solve(puzzle3, colors3)
    # # value = 1

    # puzzle4 = (3, (3, 10, 2), (11, 2, 8), 18, 17, 7), (4, (9, 9), (4, 1, 14), (10, 11), (1, 3, 8), 7)
    # colors4 = (YELLOW, None, None, BLUE, BLUE, BLUE), (RED, None, None, None, None, RED)
    # JapaneseSums(6).solve(puzzle4, colors4)
    # # value = 1

    puzzle5 = ((3, 1), 21, (9, 6, 1), 21, (11, 3), 20), (10, 21, (1, 14), (14, 4), (7, 4, 5), 16)
    JapaneseSums(6).solve(puzzle5)
    # value = 2
    #password for book 5 is 5 3 1 1 2

def puzzle_book_5():
    # puzzle1 = (23, (3, 4), (5, 1, 7), (8, 7), (3, 24), (4, 18, 11), 33, (16)), \
    #           (20, (3, 8), (7, 10), (8, 1, 22), (2, 20), (6, 2, 13), (4, 21), 20)
    # JapaneseSums(8).solve(puzzle1)
    # # Answer is 3

    # puzzle2 = (15, (4, 6), (15, 18), (16, 15), (35), (14, 16), (2, 12), 12), \
    #           ((11, 10), 17, (1, 12, 3), 36, 36, (4, 15), 25, 10)
    # JapaneseSums(8).solve(puzzle2)
    # # Answer is 6

    # puzzle3 = (2, 34, 3, 32, 2, 35, 14, 1), (12, (18, 9), (4, 2, 15), (6, 5, 2), (1, 6, 3), (5, 7, 4), (3, 11), 10)
    # JapaneseSums(8).solve(puzzle3)
    # # Answer is 5

    # puzzle4 = (14, (12, 3), (9, 18, 2), (2, 7, 4, 9), (3, 9, 21), (8, 7, 1), (2, 18), (25, 5)), \
    #           ((13, 17), (8, 15), (4, 14, 8), (8, 5, 4, 4), (5, 11, 17), (10, 8, 10), (7, 3, 6), (27, 5))
    # JapaneseSums(8, is_complicated=(4, 4)).solve(puzzle4)
    # # Answer is 6

    # puzzle5.  Paint by numbers
    # #  Answer is 7.  I did it by hand

    puzzle6 = ((2, 1), 15, (5, 6, 7, 3), 28, (9, 5, 6), (5, 4, 15, 2), (8, 19), (1, 3)), \
              (13, (4, 3, 5), (2, 16, 17, 1), (6, 4, 1, 7, 3), (1, 35), (3, 6, 7), 8, 2)
    colors6 = (PURPLE, GREEN, None, GREEN, None, None, None, GREEN), \
              (GREEN, GREEN, None, None, None, None, GREEN, GREEN)
    JapaneseSums(8).solve(puzzle6, colors6)
    #
    # puzzle7 = (14, (1, 18), (6, 3, 4, 15), (15, 4, 17), (18, 17), (7, 17), (11, 7), (9, 8)), \
    #           (11, (7, 1 , 3, 2, 5), (8, 12, 13, 1), (21, 15), (22, 11), 21, (16, 5), 17)
    # colors7 = (RED, None, None, None, None, None, YELLOW, YELLOW), \
    #           (YELLOW, None, None, None, None, YELLOW, YELLOW, YELLOW)


if __name__ == '__main__':
    puzzle_book_5()
