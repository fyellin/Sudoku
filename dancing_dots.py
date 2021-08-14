from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Sequence
from enum import Enum

from matplotlib import pyplot as plt

from color import Color


class RowOrColumn(Enum):
    ROW = 1
    COLUMN = 2

    def __str__(self) -> str:
        return self.name.title()[:3]

    def __repr__(self) -> str:
        return self.__str__()


class DancingDots:
    grid: list[list[str]]

    def solve(self, puzzle: str) -> None:
        from tools.dancing_links import DancingLinks

        self.grid = self.get_initial_grid(puzzle)
        constraints = {}
        optional_constraints = set()

        all_possible_solutions = self.get_possible_solutions()

        for solution in all_possible_solutions:
            optional_constraints.update((f'row-{solution}', f'column-{solution}'))

        for row_or_column in RowOrColumn:
            for index in range(10):
                for solution in all_possible_solutions:
                    if self.is_possible_solution(row_or_column, index, solution):
                        constraints[(row_or_column, index, solution)] = [
                            f'{"row" if row_or_column == RowOrColumn.ROW else "column"}-{solution}',
                            *self.get_row_values(row_or_column, index, solution),
                        ]

        links = DancingLinks(constraints, optional_constraints=optional_constraints, row_printer=self.draw_grid)
        links.solve(recursive=False)

    def solve2(self, puzzle: str) -> None:
        self.grid = self.get_initial_grid(puzzle)
        initial_grid = {(row, column): value for row in range(10) for column in range(10)
                        if (value := self.grid[row][column]) != '.'}
        all_possible_solutions = self.get_possible_solutions()
        info = {(row_or_column, index): all_possible_solutions
                for row_or_column in RowOrColumn for index in range(10)}
        queue = deque(info.keys())
        while True:
            while queue:
                row_or_column, index = queue.popleft()
                slots: Sequence[tuple[int, int]] = [(index, column) for column in range(10)] \
                    if row_or_column == RowOrColumn.ROW else [(row, index) for row in range(10)]
                solutions = info[row_or_column, index]
                old_length = len(solutions)
                new_solutions = [solution for solution in solutions
                                 if self.is_possible_solution(row_or_column, index, solution)]
                new_length = len(new_solutions)
                info[row_or_column, index] = new_solutions
                assert new_solutions, f'Problem on {row_or_column} {index}'
                old_line = ''.join(self.grid[row][column] for row, column in slots)
                for j, (row, column) in enumerate(slots):
                    expected = self.grid[row][column]
                    values = {solution[j] for solution in new_solutions}
                    value = values.pop() if len(values) == 1 else '.'
                    if expected != '.':
                        assert value == expected
                    elif value != '.':
                        # print(f'{row_or_column.name}#{index}  {row},{column}={value}')
                        self.grid[row][column] = value
                        other = (RowOrColumn.COLUMN, column) if row_or_column == RowOrColumn.ROW else (
                            RowOrColumn.ROW, row)
                        if other not in queue:
                            queue.append(other)
                new_line = ''.join(self.grid[row][column] for row, column in slots)
                if old_line != new_line:
                    temp = ''.join(y if x == y else Color.red + Color.bold + y + Color.reset
                                   for x, y in zip(old_line, new_line))
                    print(f'{row_or_column:3} {index}: {temp}  {old_length} -> {new_length}')

            for row_info, solutions in info.items():
                if len(solutions) == 1:
                    row_or_column1, index1 = row_info
                    for row_info2, solutions2 in info.items():
                        row_or_column2, index2 = row_info2
                        if row_or_column1 == row_or_column2 and index1 != index2 and solutions[0] in solutions2:
                            print(f'{row_info2} cannot be a duplicate of {row_info}')
                            solutions2.remove(solutions[0])
                            if row_info2 not in queue:
                                queue.append(row_info2)

            if not queue:
                break

        self.draw_grid2(initial_grid)

    @staticmethod
    def get_initial_grid(puzzle: str) -> list[list[str]]:
        lines = [list(line) for line in puzzle.splitlines() if line]
        assert len(lines) == 10
        assert all(len(line) == 10 for line in lines)
        return lines

    def is_possible_solution(self, row_or_column: RowOrColumn, index: int, solution: Sequence[str]):
        def fetch_row(col: int) -> str:
            return self.grid[index][col]

        def fetch_col(row: int) -> str:
            return self.grid[row][index]

        fetcher = fetch_row if row_or_column == RowOrColumn.ROW else fetch_col
        return all(fetcher(i) == '.' or fetcher(i) == solution[i] for i in range(10))

    @staticmethod
    def get_row_values(row_or_column: RowOrColumn, index: int, solution: Sequence[str]) -> Sequence[str]:
        if row_or_column == RowOrColumn.ROW:
            return [f'r{index}c{col}={"A" if solution[col] == "X" else "B"}' for col in range(10)]
        else:
            return [f'r{row}c{index}={"B" if solution[row] == "X" else "A"}' for row in range(10)]

    @staticmethod
    def get_possible_solutions() -> Sequence[str]:
        result = []
        for values in itertools.combinations(range(10), 5):
            if any(i in values and i + 1 in values and i + 2 in values for i in range(8)):
                continue
            if any(i not in values and i + 1 not in values and i + 2 not in values for i in range(8)):
                continue
            temp = ''.join(('X' if i in values else 'O') for i in range(10))
            result.append(temp)
        return result

    def draw_grid(self, results: Sequence[tuple[RowOrColumn, int, str]]) -> None:
        figure, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=100)

        # Set (1,1) as the top-left corner, and (max_column, max_row) as the bottom right.
        axes.axis([0, 10, 10, 0])
        axes.axis('equal')
        axes.axis('off')
        figure.tight_layout()

        # Draw the bold outline
        for x in range(0, 11):
            axes.plot([x, x], [0, 10], linewidth=1, color='black')
            axes.plot([0, 10], [x, x], linewidth=1, color='black')

        given = dict(fontsize=13, color='black', weight='heavy')
        found = dict(fontsize=12, color='blue', weight='normal')

        for row_or_column, index, solution in results:
            if row_or_column != RowOrColumn.ROW:
                continue
            for column in range(10):
                args = given if self.grid[index][column] != '.' else found
                axes.text(column + .5, index + .5, solution[column], va='center', ha='center', **args)
        plt.show()

    def draw_grid2(self, initial_grid) -> None:
        figure, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=100)

        # Set (1,1) as the top-left corner, and (max_column, max_row) as the bottom right.
        axes.axis([0, 10, 10, 0])
        axes.axis('equal')
        axes.axis('off')
        figure.tight_layout()

        # Draw the bold outline
        for x in range(0, 11):
            axes.plot([x, x], [0, 10], linewidth=1, color='black')
            axes.plot([0, 10], [x, x], linewidth=1, color='black')

        given = dict(fontsize=13, color='black', weight='heavy')
        found = dict(fontsize=12, color='blue', weight='normal')

        for row, column in itertools.product(range(10), repeat=2):
            value = self.grid[row][column]
            if value != '.':
                args = given if (row, column) in initial_grid else found
                axes.text(column + .5, row + .5, value, va='center', ha='center', **args)
        plt.show()


PUZZLE = \
    """
...O.XOOX.
O..XOXOXXO
..O.XOXOOX
..X.OXXOOX
XOXOOXOXXO
..OXXOOXXO
OXXOXOXOOX
XOOXOXXOOX
.X.OXOOXXO
O..X.OXXO.
"""

if __name__ == '__main__':
    DancingDots().solve2(PUZZLE)
