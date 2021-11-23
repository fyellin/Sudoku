from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence, Hashable
from functools import cache
from itertools import combinations, count, islice, permutations, product
from typing import Optional, cast, Union

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from tools.dancing_links import DancingLinks

WHITE = 0
GREEN = 1
PURPLE = 2
BLUE = 3
YELLOW = 4
RED = 5
GRAY = 6

Possibility = tuple[tuple[int, int], ...]
LineClue = Union[tuple[int, ...], int]
IntTuple = tuple[int, ...]
IntTupleTuple = tuple[IntTuple, ...]

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

    def solve(self, puzzle: tuple[tuple[LineClue, ...], tuple[LineClue, ...]],
              colors: Optional[tuple[tuple[Optional[int]], ...]] = None, *,
              dump: bool = False,
              draw_grid: bool = False,
              diagonal: tuple[int, int, int] = (),
              diagonals: Sequence = ()) -> None:
        constraints: dict[Hashable, list[str]] = {}
        if colors is None:
            temp = [GREEN] * self.size
            colors = [temp, temp]
        temp = colors[0] + colors[1]
        self.all_colors = list(set(y for x in filter(None, temp) for y in ([x] if isinstance(x, int) else x)) - {None})
        self.diagonals = list(diagonals)
        if diagonal:
            rc, total, magic = diagonal
            self.diagonals.append((lambda r, c: r + c == rc, total, magic))
        self.puzzle = puzzle
        self.colors = colors

        assert len(puzzle[0]) == len(puzzle[1]) == self.size
        if not self.is_complicated:
            list1 = [y for x in puzzle[0] for y in ([x] if isinstance(x, int) else x)]
            list2 = [y for x in puzzle[1] for y in ([x] if isinstance(x, int) else x)]
            if -1 not in list1 and -1 not in list2:
                assert sum(list1) == sum(list2), f'{sum(list1)} ≠ {sum(list2)}'

        if draw_grid:
            self.draw_grid()

        possibilities_by_line = {}
        for all_numbers, all_colors, is_row in zip(puzzle, colors, (True, False)):
            for line_number, (these_numbers, this_color) in enumerate(zip(all_numbers, all_colors), start=1):
                if isinstance(these_numbers, int):
                    these_numbers = (these_numbers,)
                temp = self.get_possible_rows(these_numbers, this_color)
                possible_rows = set(temp)
                if self.is_complicated:
                    possible_rows |= {tuple((value, int(not color)) for value, color in possible_row)
                                      for possible_row in possible_rows}
                possibilities_by_line[is_row, line_number] = possible_rows
                print(f'{"Row" if is_row else "Col"} {line_number} has {len(possible_rows)} possibilities')

        self.big_cleanup_attempt(possibilities_by_line)
        for (is_row, line_number), possible_rows in possibilities_by_line.items():
            for possible_row in possible_rows:
                self.__add_constraint_for_line_possibility(constraints, line_number, is_row, possible_row)

        for row, column, value in product(range(1, self.size + 1), repeat=3):
            for encoding, name in ((self.ENCODING_A, 'XRow'), (self.ENCODING_B, 'XCol')):
                constraint = []
                self.__add_value_to_constraint(constraint, row, column, value, encoding)
                constraints[name, (row, column), value] = constraint

        if dump:
            with open("/tmp/junk", "w") as file:
                for name, constraint in constraints.items():
                    print(f"{name} {constraint}", file=file)

        links = DancingLinks(constraints, row_printer=self.handle_solution)
        links.solve(recursive=False, debug=5)

    @cache
    def get_possible_rows(self, row: tuple[int], color: Optional[int]) ->  Sequence[Possibility]:
        length = len(row)

        if length == 0 or length == 1 and row[0] == 0:
            return [((0, WHITE),) * self.size]

        if length == 1 and row[0] > self.maximum // 2:
            assert isinstance(color, int)
            results = []
            for group in self.totals[self.maximum - row[0]]:
                expanded_group = [(x, WHITE) for x in group]
                center = [(0, color)] * (self.size - len(group))
                for outside in permutations(expanded_group):
                    results.extend((*outside[0:i], *center, *outside[i:]) for i in range(0, len(group) + 1))
            return results

        if color is None:
            # Produce all color lists with more than one color in it
            colors_list = [colors for colors in product(self.all_colors, repeat=length) if len(set(colors)) > 1]
        elif isinstance(color, Sequence):
            assert len(color) == length
            colors_list = [list(color)]
        else:
            colors_list = [[color for _ in range(length)]]

        extended_colors_list = [(colors, { i for i in range(1, length) if colors[i - 1] == colors[i]})
                                for colors in colors_list]
        result = []
        groups_list = self.__get_group_lists(row)
        for groups in groups_list:
            for group in groups:
                assert sum(1 for x in group if x == 0) in {0, len(group)}
            product_args = [list(permutations(group)) if group and group[0] > 0 else [group]
                            for group in groups]
            unused_boxes_count = self.size - sum(len(i) for i in groups)
            for colors, required_holes in extended_colors_list:
                if len(required_holes) <= unused_boxes_count:
                    for hole_sizes in self.__get_hole_sizes_list(length + 1, unused_boxes_count - len(required_holes)):
                        for groups_all_permutations in product(*product_args):
                            temp = []
                            for group, color, hole, index in zip(groups_all_permutations, colors, hole_sizes, count()):
                                temp.extend([(0, WHITE)] * (hole + (index in required_holes)))
                                temp.extend((v, color) for v in group)
                            temp.extend([(0, WHITE)] * hole_sizes[-1])
                            assert len(temp) == self.size
                            result.append(tuple(temp))
        return result

    @cache
    def __get_hole_sizes_list(self, holes: int, fill_count: int) -> Sequence[tuple[int, ...]]:
        assert holes > 0
        if holes == 1:
            return [(fill_count,)]
        return [(first_hole, *remainder)
                for first_hole in range(0, fill_count + 1)
                for remainder in self.__get_hole_sizes_list(holes - 1, fill_count - first_hole)]

    @cache
    def __get_group_lists(self, row: tuple[int]) -> Sequence[IntTupleTuple]:
        def internal(row: tuple[int], slots: int, available: set[int]) -> Iterable[IntTupleTuple]:
            if not row:
                yield ()
                return
            first, *rest = row
            if first == -1:
                for count in range(1, 1 + slots - len(rest)):
                    int_list = (0,) * count
                    yield from ((int_list, *other_int_lists)
                                for other_int_lists in internal(rest, slots - count, available))
            else:
                for int_list in self.totals[first]:
                    if all(x in available for x in int_list):
                        yield from ((int_list, *other_int_lists)
                                    for other_int_lists in internal(rest, slots - len(int_list), available.difference(int_list)))

        return list(internal(row, 9, set(range(1, self.size + 1))))

    def __add_constraint_for_line_possibility(
            self, constraints: dict[Hashable, list[str]],
            line_number: int, is_row: bool, possible_row: tuple[tuple[int, int]]):
        name, encoding = ('Row', self.ENCODING_A) if is_row else ('Col', self.ENCODING_B)
        constraint = [f"{name}{line_number}_set"]
        for (value, color), (row, col) in zip(possible_row, self.__get_coordinates(is_row, line_number)):
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
        constraint.extend(f"R{row}={value}_{code}" for code in encoding[column])
        constraint.extend(f"C{column}={value}_{code}" for code in encoding[row])
        if self.size >= 6:
            box, box_item = self.__get_box(row, column)
            constraint.extend(f"B{box}={value}_{code}" for code in encoding[box_item])

    def __get_totals(self) -> dict[int, list[tuple[int]]]:
        result = defaultdict(list)
        items = list(range(1, self.size + 1))
        for size in range(0, self.size + 1):
            for subset in combinations(items, size):
                result[sum(subset)].append(subset)
        return result

    @cache
    def __get_box(self, row, column):
        box_width = 3 if self.size == 9 else self.size >> 1
        box_height = self.size // box_width
        box = ((row - 1) // box_height) * box_width + (column - 1) // box_width + 1
        box_item = ((row - 1) % box_height) * box_width + (column - 1) % box_width + 1
        return box, box_item

    @cache
    def __get_coordinates(self, is_row, line_number):
        if is_row:
            return [(line_number, i) for i in range(1, self.size + 1)]
        else:
            return [(i, line_number) for i in range(1, self.size + 1)]

    COLOR_MAP = {WHITE: 'white', GREEN: 'green', PURPLE: 'purple', BLUE: 'blue',
                 YELLOW: 'goldenrod', RED: 'red', GRAY: 'gray'}

    def handle_solution(self, results: Sequence[Hashable]) -> None:
        shading = {}
        grid = {}

        for item in cast(Sequence[tuple], results):
            if item[0] == 'Row':
                _, line_number, values = item
                for (value, color), (row, col) in zip(values, self.__get_coordinates(True, line_number)):
                    if value != 0:
                        grid[row, col] = value
                    shading[row, col] = color
            elif item[0] == 'XRow':
                _, (row, col), value = item
                grid[row, col] = value

        for (func, expected_total, *extra) in self.diagonals:
            values = [grid[(row, column)]
                      for row in range(1, self.size + 1) for column in range(1, self.size + 1) if func(row, column)]
            if sum(values) != expected_total:
                return
            length = len(values)
            magic = 10 if not extra else extra[0]
            if sum(1 for i in range(length) for j in range(i + 1, length + 1) if sum(values[i:j]) == magic) != 1:
                return

        self.draw_grid(grid, shading)

    def big_cleanup_attempt(self, possibilities_by_line: dict[tuple[bool, int], set[Possibility]]):
        all_colors = set(self.all_colors) | {WHITE}
        grid = {(row, column): { (value, color) for value in range(1, self.size + 1) for color in all_colors}
                for row in range(1, self.size + 1) for column in range(1, self.size + 1)}
        if self.is_complicated:
            grid[self.is_complicated] -= {(value, WHITE) for value in range(1, self.size + 1)}
        possibility_legal_values = {}
        seen_unique = set()
        for possibilities in possibilities_by_line.values():
            for possibility in possibilities:
                if possibility not in possibility_legal_values:
                    used = {value for value, _ in possibility if value != 0}
                    not_used = frozenset(range(1, self.size + 1)) - used
                    temp = [{(value, color)} if value > 0 else {(x, color) for x in not_used}
                            for (value, color) in possibility]
                    possibility_legal_values[possibility] = temp

        total_possibilities = sum(len(possibilities) for possibilities in possibilities_by_line.values())
        print(f"There are now {total_possibilities} ({len(possibility_legal_values)})")

        queue: deque[tuple[bool, int], set[Possibility]] = deque(possibilities_by_line.items())
        possibilities_by_line.clear()
        while queue:
            deleted_possibilities = set()
            possibilities: set[Possibility]
            (is_row, line_number), possibilities = queue.popleft()
            name = "Row" if is_row else "Col"
            coordinates = self.__get_coordinates(is_row, line_number)
            buckets = [set() for _ in range(self.size)]
            for possibility in possibilities:
                my_legal_values = possibility_legal_values[possibility]
                bad_index = next((index
                                  for (row, column), legal_values, index in zip(coordinates, my_legal_values, count())
                                  if legal_values.isdisjoint(grid[row, column])), -1)
                if bad_index >= 0:
                    deleted_possibilities.add(possibility)
                else:
                    for bucket, legal_values in zip(buckets, my_legal_values):
                        bucket |= legal_values

            if deleted_possibilities:
                old_length = len(possibilities)
                possibilities -= deleted_possibilities
                print(f"Possibilities for {name} {line_number}: {old_length} -> {len(possibilities)}")
                assert possibilities, f"We have eleminated all possibilities for {name} {line_number}"

            for (row, column), bucket in zip(coordinates, buckets):
                old_length = len(grid[row, column])
                grid[row, column] &= bucket
                length = len(grid[row, column])
                if length < old_length:
                    for key in ((True, row), (False, column)):
                        if t := possibilities_by_line.pop(key, None):
                            queue.append((key, t))

            possibilities_by_line[is_row, line_number] = possibilities

            if not queue:
                total_possibilities = sum(len(possibilities) for possibilities in possibilities_by_line.values())
                print(f"There are now {total_possibilities} possibilities")
                modified = False
                for row, column in product(range(1, self.size + 1), repeat=2):
                    if (row, column) in seen_unique:
                        continue
                    values = {value for value, _ in grid[row, column]}
                    if len(values) == 1:
                        value = values.pop()
                        seen_unique.add((row, column))
                        box, _ = self.__get_box(row, column)
                        for row2, column2 in product(range(1, self.size + 1), repeat=2):
                            if (row, column) != (row2, column2):
                                if row2 == row or column2 == column  or self.__get_box(row2, column2)[0] == box:
                                    values2 = {value for value, _ in grid[row2, column2]}
                                    if value in values2:
                                        grid[row2, column2] = {(v, color) for v, color in grid[row2, column2] if v != value}
                                        print(f'Deleting r{row}c{column}={value} from r{row2}c{column2}')
                                        modified = True
                if modified:
                    queue.extend(possibilities_by_line.items())
                    possibilities_by_line.clear()

        return possibilities_by_line

    def test(self, i):
        from math import factorial as fact
        sum1 = sum(fact(len(x)) * (1 + self.size - len(x)) for x in self.totals[i])
        sum2 = sum(fact(len(x)) * (1 + len(x)) for x in self.totals[self.maximum - i])
        print(i, sum1, sum2)

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
            hbox = {6: 3, 8: 2, 9: 3}.get(self.size, self.size)
            vbox = {6: 2, 8: 2, 9: 3}.get(self.size, self.size)
            width = 3 if (x % hbox) == 1 else 1
            axes.plot([x, x], [1, self.size + 1], linewidth=width, color='black')
            width = 3 if (x % vbox) == 1 else 1
            axes.plot([1, self.size + 1], [x, x], linewidth=width, color='black')

        for direction, color_list, is_row in zip(self.puzzle, self.colors, (True, False)):
            for line_number, (line, color) in enumerate(zip(direction, color_list), start=1):
                label = ' '.join(map(str, line)) if isinstance(line, tuple) else str(line)
                if isinstance(color, Sequence):
                    # We just can't deal with multi color single lines yet
                    color = None
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
    pass
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

    # puzzle5 = ((3, 1), 21, (9, 6, 1), 21, (11, 3), 20), (10, 21, (1, 14), (14, 4), (7, 4, 5), 16)
    # JapaneseSums(6).solve(puzzle5)
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

    # puzzle6 = ((2, 1), 15, (5, 6, 7, 3), 28, (9, 5, 6), (5, 4, 15, 2), (8, 19), (1, 3)), \
    #           (13, (4, 3, 5), (2, 16, 17, 1), (6, 4, 1, 7, 3), (1, 35), (3, 6, 7), 8, 2)
    # colors6 = (PURPLE, GREEN, None, GREEN, None, None, None, GREEN), \
    #           (GREEN, GREEN, None, None, None, None, GREEN, GREEN)
    # JapaneseSums(8).solve(puzzle6, colors6)
    # # Answer is 4
    #
    # puzzle7 = (14, (1, 18), (6, 3, 4, 15), (15, 4, 17), (18, 17), (7, 17), (11, 7), (9, 8)), \
    #           (11, (7, 1 , 3, 2, 5), (8, 12, 13, 1), (21, 15), (22, 11), 21, (16, 5), 17)
    # colors7 = (RED, None, None, None, None, None, YELLOW, YELLOW), \
    #           (YELLOW, None, None, None, None, YELLOW, YELLOW, YELLOW)
    # JapaneseSums(8).solve(puzzle7, colors7)
    # # Answer is 1

    # puzzle8 = ((19, 7), (1, 20, 5), (6, 2, 4), (5, 21, 3), (10, 23, 1, 2), (9, 18), 25, (2, 4)), \
    #           (8, (9, 16, 6, 2), (3, 18, 14), (5, 4, 2, 10), (1, 16, 16), (2, 3, 6, 20, 4), 1, (19, 2))
    # colors8 = (BLUE, None, None, None, None, None, BLUE, YELLOW), (RED, None, None, None, None, None, YELLOW, None)
    # JapaneseSums(8).solve(puzzle8, colors8)
    # # Answer is 2

    puzzle9 = (17, 32, 35, (6, 3, 4), (3, 6, 8), (1, 6), (5, 1), 30), (5, 36, (10, 6), (14, 8), (18, 4), 34, (8, 2), 12)
    JapaneseSums(8).solve(puzzle9)
    # Answer is 1

    # 365674121

def r_plus_c(rc, total):
    return lambda r, c: r + c == rc, total
def r_minus_c(rc, total):
    return lambda r, c: r - c == rc, total
def c_minus_r(rc, total):
    return lambda r, c: c - r == rc, total


def puzzle_book_6():
    puzzle1 = (28, 40, 45, (5, 22, 1), (2, 4, 5), (19, 24), 41, 18, (3, 5, 4)), \
              (17, (5, 17), (21, 10), (23, 18), (24, 19), (24, 18), (18, 16), (14, 5), 17)
    JapaneseSums(9).solve(puzzle1)
    # Center is 4

    puzzle2 = (28, (3, 9, 1), (4, 6, 8), (8, 1, 18, 7), (7, 8, 13), (9, 6), (6, 22, 3), (8, 6), 32), \
              (34, (3, 8), (8, 1, 9), (7, 9, 6), (14, 6, 8, 2), (2, 23, 5, 7), (6, 3, 8), (1, 4, 6), 33)
    JapaneseSums(9).solve(puzzle2)
    # Center is 1

    puzzle3 = (42, (18, 26), (20, 7), (7, 24), (9, 5, 6), 41, 39, (11, 28), 29), \
              ((6, 18), (5, 7, 14, 9), (4, 9, 16), (15, 24), (3, 40), (7, 3, 15, 2), (8, 15), (7, 25), (11, 31))
    JapaneseSums(9, is_complicated=(5, 5)).solve(puzzle3)
    # Center is 5

    puzzle4 = (0, 5, 26, (21, 2), (1, 4), 33, 42, 17, 5), \
              (0, 6, (19, 12), (5, 7), (5, 9), (21, 9), (1, 18), (2, 21), 21)
    colors4 = (None, YELLOW, YELLOW, YELLOW, YELLOW, GRAY, GRAY, GRAY, GRAY), \
              (None, YELLOW, *([(YELLOW, GRAY)] * 6), GRAY)
    JapaneseSums(9).solve(puzzle4, colors4, diagonal=(9, 44, 10))
    # Center is 7

    puzzle5 = ((2, 1), 21, (4, 3, 5), 25, 42, (11, 19, 12), 19, (3, 4), 21), \
              (6, 13, (13, 3), (41, 4), (8, 3, 16, 2), (32, 13), (13, 6), 10, 9)
    colors5 = (GRAY, GRAY, (GRAY, YELLOW, GRAY), GRAY, GRAY, GRAY, GRAY, YELLOW, YELLOW), \
              (GRAY, GRAY, (GRAY, YELLOW), (GRAY, YELLOW), (GRAY, YELLOW, GRAY, YELLOW), (GRAY, YELLOW), (GRAY, YELLOW), GRAY, GRAY)
    diagonals = [r_plus_c(8,40), r_plus_c(16, 10), r_minus_c(4, 19), r_minus_c(2, 40), c_minus_r(6, 17), c_minus_r(7, 10)]
    JapaneseSums(9).solve(puzzle5, colors5, diagonals=diagonals)
    # Center is 6

    puzzle6 = (26, 39, (-1, 15, -1, -1), (24, 15), (11, -1, 28), (-1), (31, -1, 8), (8, -1, 8), (10, -1, 16)), \
              ((), (5, -1), (13, 32), (16, 10, 3, 16), (-1, -1, 5), (-1, 42), (20, 11, -1, 12), (6, 7, 22), 15)
    g, y, t = GREEN, YELLOW, GRAY
    colors6 = (g, g, (g, y, g, t), (y, t), (y, t, y), y, (y, t, y), (y, t, y), (y, t, y)), \
              (None, (g, y), (g, y), (g, y, t, y), (g, y, t), (g, y), (g, y, t, y), (g, t, y), t)
    diagonals6 = [c_minus_r(5, 21)]
    JapaneseSums(9).solve(puzzle6, colors6, diagonals=diagonals6)
    # Center is 7

    # y, t, b = YELLOW, GRAY, BLUE
    #
    # puzzle7 = (9, 5, 6, (4, 7, 16, 6), (7, 3, 4, 1, 8), (8, 15, 8, 4), (9, 36), 45, ()), \
    #           (31, (14, -1), (3, 14, -1), (4, 6, -1), (25, 3, -1), (2, 8, 8, -1), (6, 9, -1), (3, 4, -1), (18, 1, -1))
    # colors7 = (t, t, t, (t, t, y, t), (t, y, t, y, t), (t, b, b, t), (t, b), t, None), \
    #           (t, (b, t), (y, b, t), (t, b, t), (t, b, t), (t, y, b, t), (y, b, t), (y, b, t), (t, b, t))
    # diagonals7 = [r_plus_c(5, 18), c_minus_r(5, 18)]
    # JapaneseSums(9).solve(puzzle7, colors7, diagonals=diagonals7)
    # # Center is 4, even though we don't know which possibility it is!
    #
    # 4157674


def test():
    x = JapaneseSums(8)
    for i in range(0, 1 + size.maximum + 1):
        x.test(i)

if __name__ == '__main__':
    puzzle_book_6()
    # test()

