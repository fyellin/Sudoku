from __future__ import annotations

import datetime
import itertools
from collections import defaultdict
from typing import Tuple, Sequence, Dict, Any, List, NamedTuple, Set, Optional, Mapping

import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.pyplot import arrow


class Square(NamedTuple):
    row: int
    column: int

    def __str__(self) -> str:
        return f"r{self.row}c{self.column}"

    def __repr__(self) -> str:
        return str(self)


class Hidato:
    grid: Mapping[Square, int]

    def __init__(self, puzzle: str):
        self.grid = self.get_initial_grid(puzzle)

    def solve(self) -> None:
        reverse_grid = {value: location for location, value in self.grid.items() if value > 0}
        values = reverse_grid.keys()
        sorted_values = sorted(values)
        segments = []
        for this, next_one in zip(sorted_values, sorted_values[1:]):
            if next_one != this + 1:
                segment = Segment(self.grid, this, next_one, reverse_grid[this], reverse_grid[next_one])
                segments.append(segment)

        segment_square_already_handled = set()
        done_segments = set()

        while segments:
            print(f'\n******************* {len(segments)}')
            table = {segment: segment.get_statistics() for segment in segments}
            already_handled_length = len(segment_square_already_handled)

            square_to_segment = defaultdict(list)
            for segment in segments:
                for square in table[segment].possible:
                    square_to_segment[square].append(segment)
            singletons = [(square, segments[0]) for square, segments in square_to_segment.items()
                          if len(segments) == 1]
            for square, segment in singletons:
                if not (segment, square) in segment_square_already_handled:
                    print(f"{segment} is the only segment containing {square}")
                    segment.require_square(square)
                    segment_square_already_handled.add((segment, square))

            for segment in segments:
                delta = [square for square in table[segment].required
                         if (segment, square) not in segment_square_already_handled]
                if delta:
                    print(f'{segment} always includes {", ".join(map(str, delta))}')

                for square in delta:
                    segment_square_already_handled.add((segment, square))
                    for other_segment in segments:
                        if other_segment != segment and square in table[other_segment].possible:
                            other_segment.prohibit_square(square)

                delta = [squares for squares in table[segment].required_pairs
                         if (segment, squares) not in segment_square_already_handled]
                for squares in delta:
                    segment_square_already_handled.add((segment, squares))
                    print(f'{segment} always includes one of {squares}')
                    for other_segment in segments:
                        if other_segment != segment and \
                                all(square in table[other_segment].possible for square in squares):
                            other_segment.prohibit_squares(squares)

            done_segments.update(segment for segment in segments if table[segment].length == 1)
            segments = [segment for segment in segments if segment not in done_segments]

            if already_handled_length == len(segment_square_already_handled):
                break

        square_to_values = defaultdict(set)
        for segment in segments + list(done_segments):
            segment.add_results(square_to_values)
        for square, value in self.grid.items():
            if value != 0:
                square_to_values[square].add(value)
        self.draw_grid(square_to_values)

    def solve2(self) -> None:
        def neighbors(location: Square) -> Sequence[Square]:
            r, c = location
            return [Square(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

        reverse_grid = {value: location for location, value in self.grid.items() if value > 0}
        values = reverse_grid.keys()
        paths = [[reverse_grid[1]]]
        for value in range(2, max(values) + 1):
            if value not in reverse_grid:
                paths = [path + [neighbor] for path in paths
                         for neighbor in neighbors(path[-1])
                         if neighbor not in path and self.grid.get(neighbor) == 0]
            else:
                square = reverse_grid[value]
                good_neighbors = set(neighbors(square))
                paths = [path + [square] for path in paths if path[-1] in good_neighbors]
            print(value, len(paths))

        square_to_values = defaultdict(set)
        for path in paths:
            for value, square in enumerate(path, start=1):
                square_to_values[square].add(value)
        self.draw_grid(square_to_values)

    @staticmethod
    def get_initial_grid(puzzle: str) -> Dict[Square, int]:
        grid = {}
        lines = [line for line in puzzle.splitlines() if line]
        for row, line in enumerate(lines, start=1):
            for col, item in enumerate(line.rstrip().split(' '), start=1):
                if item == '..':
                    grid[Square(row, col)] = 0
                elif item.lower() == 'xx':
                    pass
                else:
                    grid[Square(row, col)] = int(item)
        max_value = max(grid.values())
        # assert max_value == len(grid)
        return grid

    def draw_grid(self, square_to_values: Mapping[Square, Set[int]]) -> None:
        figure, axes = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
        reverse_grid = {value: square for square, values in square_to_values.items()
                        if len(values) == 1 for value in [next(iter(values))]}

        max_row = max(row for row, _ in self.grid.keys())
        max_col = max(col for _, col in self.grid.keys())
        max_value = max(self.grid.values())

        # Set (1,1) as the top-left corner, and (max_column, max_row) as the bottom right.
        axes.axis([1, max_col + 1, max_row + 1, 1])
        axes.axis('equal')
        axes.axis('off')
        figure.tight_layout()

        given = dict(fontsize=13, color='black', weight='heavy')
        found = dict(fontsize=12, color='blue', weight='normal')
        small = dict(fontsize=8, color='red', weight='normal')

        self.draw_outline(list(self.grid.keys()), inset=-2/72, linestyle=None, linewidth=4)
        for (row, col), values in square_to_values.items():
            value = next(iter(values)) if len(values) == 1 else 0
            if value not in (1, max_value):
                axes.add_patch(patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor='black', facecolor='none'))
            else:
                axes.add_patch(patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor='black', facecolor='black'))
                axes.add_patch(patches.Circle((col + .5, row + .5), .4, linewidth=1, edgecolor='white', facecolor='w'))
            if value != 0:
                args = given if value in self.grid.values() else found
                axes.text(col + .5, row + .5, str(value),
                          verticalalignment='center', horizontalalignment='center', **args)
                if value + 1 in reverse_grid:
                    row2, col2 = reverse_grid[value + 1]
                    mid_row, mid_col = (row + row2 + 1) / 2, (col + col2 + 1) / 2
                    d_row, d_col = row2 - row, col2 - col
                    axes.add_patch(arrow(mid_col - .2 * d_col, mid_row - .2 * d_row, .4 * d_col, .4 * d_row,
                                         head_length=.15, overhang=.3,
                                         head_width=.15, length_includes_head=True))
            elif len(values) <= 3:
                string = '\n'.join(str(value) for value in sorted(values))
                axes.text(col + .5, row + .5, string,
                          verticalalignment='center', horizontalalignment='center', **small)
        plt.show()

    @staticmethod
    def draw_outline(squares: Sequence[Square], *, inset: float = .1, **args: Any) -> None:
        args = {'color': 'black', 'linewidth': 2, 'linestyle': "dotted", **args}
        squares_set = set(squares)

        # A wall is identified by the square it is in, and the direction you'd be facing from the center of that
        # square to see the wall.  A wall separates a square inside of "squares" from a square out of it.
        walls = {(row, column, dr, dc)
                 for row, column in squares for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
                 if (row + dr, column + dc) not in squares_set}

        while walls:
            start_wall = current_wall = next(iter(walls))  # pick some wall
            points: List[np.ndarray] = []

            while True:
                # Find the connecting point between the current wall and the wall to the right and add it to our
                # set of points

                row, column, ahead_dr, ahead_dc = current_wall  # square, and direction of wall from center
                right_dr, right_dc = ahead_dc, -ahead_dr  # The direction if we turned right

                # Three possible next walls, in order of preference.
                #  1) The wall makes a right turn, staying with the current square
                #  2) The wall continues in its direction, going into the square to our right
                #  3) The wall makes a left turn, continuing in the square diagonally ahead to the right.
                next1 = (row, column, right_dr, right_dc)   # right
                next2 = (row + right_dr, column + right_dc, ahead_dr, ahead_dc)  # straight
                next3 = (row + right_dr + ahead_dr, column + right_dc + ahead_dc, -right_dr, -right_dc)  # left

                # It is possible for next1 and next3 to both be in walls if we have two squares touching diagonally.
                # In that case, we prefer to stay within the same cell, so we prefer next1 to next3.
                next_wall = next(x for x in (next1, next2, next3) if x in walls)
                walls.remove(next_wall)

                if next_wall == next2:
                    # We don't need to add a point if the wall is continuing in the direction it was going.
                    pass
                else:
                    np_center = np.array((row, column)) + .5
                    np_ahead = np.array((ahead_dr, ahead_dc))
                    np_right = np.array((right_dr, right_dc))
                    right_inset = inset if next_wall == next1 else -inset
                    points.append(np_center + (.5 - inset) * np_ahead + (.5 - right_inset) * np_right)

                if next_wall == start_wall:
                    break
                current_wall = next_wall

            points.append(points[0])
            pts = np.vstack(points)
            plt.plot(pts[:, 1], pts[:, 0], **args)


class Statistic(NamedTuple):
    length: int
    possible: Set[Square]
    required: Set[Square]
    required_pairs: Set[Tuple[Square, Square]]


class Segment:
    start: int
    end: int
    start_location: Square
    end_location: Square
    paths: List[Tuple[Square, ...]]
    statistics: Optional[Statistic]

    def __init__(self, grid: Mapping[Square, int], start: int, end: int,
                 start_location: Square, end_location: Square) -> None:
        self.start = start
        self.end = end
        self.start_location = start_location
        self.end_location = end_location
        self.paths = self.get_initial_paths(grid)
        self.statistics = None
        print(f'There are {len(self.paths)} paths from {self.start} to {self.end}')

    def __len__(self) -> int:
        return len(self.paths)

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Segment) -> bool:
        return self is other

    def __str__(self) -> str:
        return f"Segment {self.start}-{self.end}"

    def __repr__(self) -> str:
        return str(self)

    def get_initial_paths(self, grid) -> List[Tuple[Square, ...]]:
        def neighbors(location: Square) -> Sequence[Square]:
            r, c = location
            return [Square(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

        paths = [[self.start_location]]
        for _ in range(self.end - self.start - 1):
            paths = [path + [neighbor] for path in paths
                     for neighbor in neighbors(path[-1])
                     if neighbor not in path and grid.get(neighbor) == 0]
        paths = [path + [neighbor] for path in paths
                 for neighbor in neighbors(path[-1])
                 if neighbor == self.end_location]
        self.check_debug()
        return [tuple(path[1:-1]) for path in paths]

    def require_square(self, square: Square):
        self.check_debug()
        old_length = len(self)
        self.paths = [path for path in self.paths if square in path]
        if old_length != len(self):
            print(f'  {self} reduced {old_length} -> {len(self)} because path must include {square}')

    def prohibit_square(self, square: Square):
        self.check_debug()
        old_length = len(self)
        self.paths = [path for path in self.paths if square not in path]
        if old_length != len(self):
            print(f'  {self} reduced {old_length} -> {len(self)} because path cannot include {square}')

    def prohibit_squares(self, squares: Sequence[Square]):
        self.check_debug()
        old_length = len(self)
        self.paths = [path for path in self.paths if not all(square in path for square in squares)]
        if old_length != len(self):
            print(f'  {self} reduced {old_length} -> {len(self)} because path cannot include all {squares}')

    def get_statistics(self) -> Statistic:
        if self.statistics and self.statistics.length == len(self):
            return self.statistics
        paths_as_sets = [set(path) for path in self.paths]
        possible = set.union(*paths_as_sets)
        required = set.intersection(*paths_as_sets)
        pairs = {
            (x, y) for x, y in itertools.combinations(possible - required, 2)
            if all(x in path or y in path for path in self.paths)}
        pairs = {tuple(sorted(item)) for item in pairs}
        self.statistics = Statistic(length=len(self), possible=possible, required=required, required_pairs=pairs)
        return self.statistics

    def add_results(self, result: Dict[Square, Set[int]]) -> None:
        result[self.start_location].add(self.start)
        result[self.end_location].add(self.end)
        for path in self.paths:
            for value, square in enumerate(path, start=self.start + 1):
                result[square].add(value)

    def check_debug(self) -> None:
        pass


PUZZLE = """
.. .. xx xx xx xx xx xx 67 68
04 .. .. 09 xx xx .. .. .. ..
xx 02 .. .. .. .. .. 64 .. xx
xx 01 .. 24 .. .. 18 .. 62 xx
xx xx 26 .. 12 16 .. .. xx xx
xx xx 27 30 .. .. 15 .. xx xx
xx 35 .. 32 .. .. .. .. 58 xx
xx .. 34 .. 43 45 49 48 .. xx
.. 39 .. .. xx xx .. .. 54 55
.. .. xx xx xx xx xx xx .. ..
"""

PUZZLE2 = """
35 33 .. .. 26 .. .. .. .. ..
36 .. 32 31 .. 27 23 20 19 ..
.. 39 xx xx xx xx xx xx .. 16
38 .. xx .. .. .. 74 xx .. 15
.. 43 xx .. .. 75 .. xx .. 10
42 .. xx .. 71 69 .. xx 09 ..
45 .. xx .. .. 82 .. xx .. 01
.. 47 xx xx .. .. xx xx .. 02
.. 51 54 .. 64 .. 62 61 06 ..
.. 52 .. 56 .. 63 .. .. 05 ..
"""

PUZZLE3 = """
.. .. 72 16 18 28 .. 26 25 24
69 .. .. .. .. .. 20 .. .. ..
.. 68 .. ..
65 .. .. ..
33 .. 10 .. 07 .. 04 01
34 .. .. .. 08 05 .. 02
35 .. .. 41
61 .. .. ..
.. 37 .. .. .. 44 45 46 .. ..
.. .. .. 55 .. .. .. .. 47 49
"""

PUZZLE4 = """
65 .. .. .. .. .. 18 .. 16 ..
.. 66 68 .. .. .. .. 17 15 ..
XX 67 .. .. XX XX .. 11 ..
XX .. .. .. XX XX .. .. ..
58 .. XX XX .. 26 XX XX 01 ..
76 57 XX XX .. 28 XX XX .. ..
XX .. 55 .. XX XX 04 .. ..
XX .. .. 34 XX XX .. .. 42
.. .. 32 .. .. .. 45 38 .. ..
.. 51 .. .. .. 46 .. .. .. ..
"""

PUZZLE_09_24 = """
XX XX .. 01 .. .. .. ..
XX 04 06 .. .. 70 69 .. 62
.. .. 03 XX .. .. XX 66 65 ..
13 .. .. 76 26 .. .. .. .. 59
.. 12 11 .. .. 27 .. 56 .. ..
.. 17 .. .. 78 .. 29 53 .. 51
16 .. .. XX .. .. XX 44 52 ..
19 .. .. .. XX XX .. 45 .. ..
XX XX 34 .. .. 39 42 ..
XX XX XX 36 37 41 ..
"""


PUZZLE_09_26 = """
XX XX 40 04 ..
XX .. .. 03 17 ..
.. 36 02 .. .. 18 ..
.. 32 .. 01 07 13 ..
.. .. .. .. .. .. 12
28 .. .. .. .. .. 11
29 XX 25 XX .. XX 10
"""

PUZZLE_09_27 = """
XX XX XX 44 .. XX XX XX
XX 16 .. .. 36 38 .. XX
XX 18 .. .. 42 .. 01
21 .. 13 14 .. .. 03 02
.. 20 .. .. .. .. 06 ..
XX .. 30 .. .. .. 08
XX .. .. .. 28 .. ..
XX XX XX .. ..
"""

PUZZLE_09_28 = """
.. .. 02 19 .. 22 .. 31
.. 01 16 .. 20 .. .. 29
.. .. 14 .. .. 26 28 ..
.. .. .. xx xx 25 .. ..
12 .. .. xx xx 37 35 60
.. .. .. .. 52 38 .. ..
.. .. 43 41 .. 39 55 ..
.. .. .. .. .. .. .. 57
"""

PUZZLE_09_30 = """
.. 72 73 .. .. .. .. 14 12
68 .. .. 18 75 76 .. .. 11
.. 69 .. 25 .. .. 04 .. ..
.. .. .. .. XX 29 01 .. ..
.. 23 .. XX XX XX .. 06 ..
.. 54 .. 52 XX .. 36 .. ..
.. .. 53 .. .. 33 34 37 ..
.. 59 56 .. .. .. 43 40 39
.. 60 .. .. 46 45 .. .. ..
"""

PUZZLE_10_07 = """
XX XX 20 .. .. .. .. ..
XX 21 .. 19 .. 15 .. .. ..
26 .. 23 XX .. .. XX .. 61 ..
.. 24 .. .. 08 .. 02 .. 04 55
.. 31 30 78 .. 11 01 .. .. ..
.. 35 .. 71 70 10 .. .. .. ..
.. 36 .. XX 72 69 XX .. 50 ..
.. 38 .. .. XX XX .. .. 49 ..
XX XX .. .. 43 .. .. ..
XX XX XX .. .. 42 46 
"""

PUZZLE_11_11 = """
39 .. 01 .. .. .. .. 06
.. 37 .. 35 03 31 .. ..
41 42 62 29 .. .. .. ..
45 .. .. .. .. .. .. XX
XX 46 47 27 .. .. 14 ..
.. .. .. 59 .. .. .. 13
52 55 .. 58 .. 22 20 17
54 .. .. 57 .. 23 .. 18
"""

PUZZLE_11_14 = """
.. 13 15 .. .. .. 22 .. .. ..
.. 12 .. 16 17 34 .. 31 30 ..
09 .. 05 .. .. .. .. .. .. ..
08 06 .. 36 02 XX XX XX XX XX
40 39 38 .. 01 .. .. .. .. ..
.. .. .. .. .. 48 49 53 .. ..
XX XX XX XX XX .. .. 57 85 83
.. 66 .. .. .. 75 .. .. .. 81
.. 65 72 61 .. .. .. 77 .. ..
.. .. .. 73 .. 90 .. .. .. ..
"""

PUZZLE_2021_01_28 = """
03 04 .. .. .. .. XX XX XX ..
XX .. 01 07 .. 10 XX XX .. 23
XX XX .. 15 .. .. XX .. 24 ..
XX XX XX 13 16 18 .. 33 31 ..
64 .. .. 60 XX XX .. .. 30 ..
.. 63 59 .. XX XX .. .. 29 ..
.. 68 58 .. 53 .. ..
.. .. 56 XX .. .. 47 39
71 .. XX XX .. 48 .. 44 ..
72 XX XX XX .. 49 .. .. .. ..
"""

PUZZLE_2021_02_04 = """
XX .. .. 01 .. .. ..
.. .. .. .. .. 05 17 ..
34 .. 30 .. 06 08 21 ..
35 .. .. 25 .. .. 09 ..
.. 38 .. .. 24 .. 10 14
.. 40 57 .. 48 .. 12 ..
60 43 .. .. .. 49 .. ..
XX .. .. .. 55 .. ..
"""

PUZZLE_2021_02_13 = """
.. .. .. .. .. .. .. .. 01 ..
.. 14 .. .. .. .. 05 82 .. ..
.. 13 .. 22 10 84 .. .. 76 ..
.. .. 24 XX XX XX XX .. 77 ..
30 28 26 XX XX XX XX 79 .. ..
32 .. .. XX XX XX XX .. 65 ..
33 .. 43 XX XX XX XX 66 .. ..
35 .. .. 44 .. 51 .. 68 .. ..
.. 36 .. .. .. 52 53 56 58 ..
.. 37 .. 47 48 .. .. .. 60 ..
"""

PUZZLE_2021_02_24 = """
XX XX XX XX ..
XX XX .. .. 27 .. ..
18 .. 22 21 26 .. .. 44 ..
.. .. .. 63 .. .. .. .. 43
.. .. .. .. 33 61 40 49 ..
15 .. .. .. .. .. .. 41 ..
.. 09 35 03 .. .. 58 52 ..
.. 08 .. .. XX 01 .. 53 ..
.. .. XX XX XX XX XX .. 55
"""
PUZZLE_2021_03_05 = """
XX XX 18 .. 21 .. .. .. 34 ..
XX XX XX 17 .. .. 23 36 .. ..
.. XX XX XX .. 27 .. .. .. ..
.. 69 XX XX 14 .. 41 38 30 ..
.. .. 67 12 .. .. .. 42 .. ..
.. 66 .. .. 55 .. .. .. .. ..
.. 62 .. .. 10 54 XX XX 44 ..
61 .. .. .. 08 .. XX XX XX ..
.. .. .. .. .. .. ..
XX .. 59 79 01 02 03 ..
"""

PUZZLE_2021_06_28 = """
51 .. .. 46 .. .. .. 75 76
.. .. 48 45 .. 56 .. .. 74
.. 01 .. .. .. 39 .. 41 ..
.. 02 .. 36 XX .. .. .. ..
.. 06 .. XX XX XX 70 .. ..
08 .. 33 .. XX .. .. 69 ..
.. 11 .. .. .. 66 68 .. ..
10 .. 16 .. .. .. .. .. 25
.. 15 .. .. 19 21 27 23 ..
"""


def hidato_run() -> None:
    hidato = Hidato(PUZZLE_2021_06_28)
    start = datetime.datetime.now()
    hidato.solve2()
    end = datetime.datetime.now()
    print(end - start)


if __name__ == '__main__':
    hidato_run()

