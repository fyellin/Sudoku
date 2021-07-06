from __future__ import annotations

from itertools import product, combinations, permutations, groupby
from collections import deque, defaultdict
from collections.abc import Sequence, Mapping, Iterable
from operator import attrgetter

from matplotlib import pyplot as plt

from cell import House, Cell, CellValue
from chain import Chains
from draw_context import DrawContext
from feature import Feature
from grid import Grid
from hard_medusa import HardMedusa


class Sudoku:
    grid: Grid
    features: Sequence[Feature]
    initial_grid: Mapping[tuple[int, int], int]

    def solve(self, puzzle: str, *, features: Sequence[Feature] = (), show: bool = False) -> bool:
        self.features = features
        self.grid = grid = Grid(features)
        grid.reset()
        self.initial_grid = {(row, column): int(letter)
                             for (row, column), letter in zip(product(range(1, 10), repeat=2), puzzle)
                             if '1' <= letter <= '9'}

        for square, value in self.initial_grid.items():
            grid.matrix[square].set_value_to(value)
        try:
            return self.run_solver(show)
        except Exception:
            self.draw_grid()
            raise

    def run_solver(self, show: bool) -> bool:
        self.grid.print()
        self.draw_grid()

        if show:
            return True

        while True:
            if self.is_solved():
                self.draw_grid()
                return True
            if self.check_naked_singles() or self.check_hidden_singles():
                continue
            if any(feature.check() for feature in self.features):
                continue
            if self.check_intersection_removal():
                continue
            if self.check_tuples():
                continue
            # if self.check_intersection_removal_double():
            #     continue
            self.grid.print()
            self.draw_grid()

            if any(feature.check_special() for feature in self.features):
                continue

            if self.check_fish() or self.check_xy_sword() or self.check_xyz_sword() or self.check_tower():
                continue
            if self.check_xy_chain(81):
                continue
            chains = Chains.create(self.grid.cells, True)
            if self.check_chain_colors(chains):
                continue
            if HardMedusa.run(chains, self.features):
                continue

            self.draw_grid()
            return False

    def is_solved(self) -> bool:
        """Returns true if every square has a known value"""
        return self.grid.is_solved()

    def check_naked_singles(self) -> bool:
        """
        Finds those squares which are forced because they only have one possible remaining value.
        Returns true if any changes are made to the grid
        """
        found_naked_single = False
        while True:
            # Cells that only have one possible value
            naked_singles = {cell for cell in self.grid.cells
                             if not cell.is_known and len(cell.possible_values) == 1}
            if not naked_singles:
                break
            found_naked_single = True
            # Officially set the cell to its one possible value
            output = [cell.set_value_to(list(cell.possible_values)[0])
                      for cell in naked_singles]
            print("Naked Single: " + '; '.join(output))
        return found_naked_single

    def check_hidden_singles(self) -> bool:
        """
        Finds a house for which there is only one place that one or more digits can go.
        Returns true if it finds such a house.
        """
        return any(self.__check_hidden_singles(house) for house in self.grid.houses)

    @staticmethod
    def __check_hidden_singles(house: House) -> bool:
        # Make a sorted list of all cell/value combinations not yet known
        all_unknown_cell_values = [CellValue(cell, value)
                                   for cell in house.unknown_cells
                                   for value in cell.possible_values]
        all_unknown_cell_values.sort(key=attrgetter("value"))
        result = False
        for value, iterator in groupby(all_unknown_cell_values, attrgetter("value")):
            cell_values = tuple(iterator)
            if len(cell_values) == 1:
                cell = cell_values[0].cell
                cell.set_value_to(value)
                print(f'Hidden Single: {house} = {value} must be {cell}')
                result = True
        return result

    def check_intersection_removal(self) -> bool:
        """
        Original explanation:
        If the only possible places to place a digit in a particular house are all also within another house, then
        all other occurrences of that digit in the latter house can be deleted.

        New explanation:
        If the only possible places to place a digit in a particular house are all neighbors of a cell outside the
        house, then that outside cell cannot contain the digit (or it would eliminate all the possibilities.)  This
        is more general than the original explanation, and allows this to work with knight- and king-sudoku, too.

        Returns true if we make a change.
        """
        return any(self.__check_intersection_removal(house, value)
                   for house in self.grid.houses
                   for value in house.unknown_values)

    def __check_intersection_removal(self, house: House, value: int) -> bool:
        """Checks for intersection removing of the specific value in the specific house"""
        candidates = [cell for cell in house.unknown_cells if value in cell.possible_values]
        assert len(candidates) > 1
        cell0, *other_candidates = candidates
        # Find all cells that both have the specified value, and are neighbors of all the candidates.

        def get_all_neighbors(cell: Cell) -> set[Cell]:
            immediate_neighbors = cell.neighbors
            other_neighbors = {x for feature in self.features
                               for x in feature.get_neighbors_for_value(cell, value)}
            if other_neighbors:
                return immediate_neighbors.union(other_neighbors)
            else:
                return immediate_neighbors

        fixers = {cell for cell in get_all_neighbors(cell0) if value in cell.possible_values}
        fixers.intersection_update(*(get_all_neighbors(cell) for cell in other_candidates))
        if fixers:
            print(f'Intersection Removal: {house} = {value} must be one of {sorted(candidates)}')
            Cell.remove_value_from_cells(fixers, value)
            return True
        return False

    def check_tuples(self) -> bool:
        """
        If there are a group of n cells, all of whose possible values are a subset of a specific n digits, then
        that digit can only occur in one of those n cells.
        Returns true if it makes any change.
        """
        return any(self.__check_tuples(house, set(values))
                   # Specifically find all tuples of 2 before trying all tuples of 3, . . . .
                   for count in range(2, 9)
                   # Look at each house
                   for house in self.grid.houses if len(house.unknown_values) > count
                   # Look at each subset of size "count" of the unknown values of that house
                   for values in combinations(house.unknown_values, count))

    @staticmethod
    def __check_tuples(house: House, values: set[int]) -> bool:
        """
        Looks to see if "values" is a tuple in this house.  Returns true if it makes any changes.
        """
        # Find those cells in this house whose possible values are a subset of the tuple
        tuple_cells = [cell for cell in house.unknown_cells if cell.possible_values <= values]
        if len(tuple_cells) != len(values):
            return False
        # We have precisely the right number.  Delete these values if they occur in any other cells
        fixers = [cell for cell in house.unknown_cells
                  if cell not in tuple_cells and cell.possible_values & values]
        if not fixers:
            return False

        # Let n = len(values) and k = len(house.unknown_values) - n
        # We've discovered that n cells only contain a subset of n values.  But that means that the remaining
        # k values only occur in the remaining k cells.  Both same the same thing.   We can look at what we're about
        # to do as either
        #     (1) The n values can only occur in those n cells, and must be deleted from all other cells or
        #     (2) The remaining k values must occur in those k cells, and all other digits can be deleted.
        # Both say the same thing.  How we word it depends on which is smaller, n or k.
        if len(values) * 2 <= len(house.unknown_values):
            print(f'{house} has tuple {sorted(values)} in squares {sorted(tuple_cells)}:')
        else:
            hidden_tuple = house.unknown_values - values
            hidden_squares = house.unknown_cells.difference(tuple_cells)
            print(f'{house} has hidden tuple {sorted(hidden_tuple)} in squares {sorted(hidden_squares)}:')
        Cell.remove_values_from_cells(fixers, values)
        return True

    def check_intersection_removal_double(self) -> bool:
        boxes = [house for house in self.grid.houses if house.house_type == House.Type.BOX]
        if any(self.__check_intersection_removal_double(boxes, htype, value)
               for htype in (House.Type.COLUMN, House.Type.ROW)
               for value in range(1, 10)):
            return True
        return False

    @staticmethod
    def __check_intersection_removal_double(all_boxes: Sequence[House], htype: House.Type, value: int) -> bool:
        info: dict[frozenset[House], set[House]] = defaultdict(set)
        for box in all_boxes:
            if value in box.unknown_values:
                rows = frozenset(cell.house_of_type(htype)
                                 for cell in box.unknown_cells if value in cell.possible_values)
                info[rows].add(box)

        for rows, boxes in info.items():
            if len(rows) == 2 and len(boxes) == 2:
                impossible_cells: set[Cell] = set()
                for rows2, boxes2 in info.items():
                    if rows2 != rows and rows2.intersection(rows):
                        impossible_cells.update(cell for box in boxes2 for cell in box.unknown_cells
                                                if value in cell.possible_values and cell.house_of_type(htype) in rows)
                if impossible_cells:
                    print(f'{boxes} has {value} in {rows}')
                    Cell.remove_value_from_cells(impossible_cells, value)
                    return True
        return False

    def check_fish(self) -> bool:
        """Looks for a fish of any size.  Returns true if a change is made to the grid."""
        for value in range(1, 10):
            # Find all houses for which the value is missing
            empty_houses = [house for house in self.grid.houses if value in house.unknown_values]
            if not empty_houses:
                continue
            # For convenience, make a map from each "empty" house to the cells in that house that can contain the value
            empty_house_to_cell = {house: [cell for cell in house.unknown_cells if value in cell.possible_values]
                                   for house in empty_houses}
            # Look for a fish between any two House types on the specified value
            # noinspection PyTypeChecker
            house_types = (House.Type.COLUMN, House.Type.ROW, House.Type.BOX)
            for this_house_type, that_house_type in permutations(house_types, 2):
                if self.__check_fish(value, empty_houses, empty_house_to_cell, this_house_type, that_house_type):
                    return True
        return False

    @staticmethod
    def __check_fish(value: int,
                     empty_houses: Sequence[House],
                     empty_house_to_cell: Mapping[House, Sequence[Cell]],
                     this_house_type: House.Type,
                     that_house_type: House.Type) -> bool:
        these_unknown_houses = [house for house in empty_houses if house.house_type == this_house_type]
        those_unknown_houses = [house for house in empty_houses if house.house_type == that_house_type]
        assert len(these_unknown_houses) == len(those_unknown_houses) >= 2
        unknown_size = len(these_unknown_houses)
        # We arbitrarily pretend that this_house_type is ROW and that_house_type is COLUMN in the naming of our
        # variables below.  But that's just to simplify the algorithm.  Either House can be any type.
        max_rows_to_choose = unknown_size - 1
        # if this_house_type == House.Type.BOX or that_house_type == House.Type.BOX:
        #     max_rows_to_choose = min(2, max_rows_to_choose)
        # Look at all subsets of the rows, but do small subsets before doing large subsets
        for number_rows_to_choose in range(2, max_rows_to_choose + 1):
            for rows in combinations(these_unknown_houses, number_rows_to_choose):
                # Find all the possible cells in those rows
                row_cells = {cell for house in rows for cell in empty_house_to_cell[house]}
                # Find the columns that those cells belong to
                columns = {cell.house_of_type(that_house_type) for cell in row_cells}
                assert len(columns) >= number_rows_to_choose
                if len(columns) > number_rows_to_choose:
                    continue
                # If len(columns) == number_rows_to_choose, we have a fish.  Let's see if there is something to delete.
                # Find all the cells in those columns
                column_cells = {cell for house in columns for cell in empty_house_to_cell[house]}
                assert row_cells <= column_cells
                if len(row_cells) < len(column_cells):
                    # There are some column cells that aren't in our rows.  The value can be deleted.
                    fixer_cells = column_cells - row_cells
                    print(f'Fish. {tuple(sorted(rows))}  have their {value} in {tuple(sorted(columns))}.  ')
                    print(f'Other occurrences of {value} in {tuple(sorted(columns))} can be deleted.')
                    Cell.remove_value_from_cells(fixer_cells, value)
                    return True
        return False

    def check_xy_sword(self) -> bool:
        return self.check_xy_chain(3)

    def check_xy_chain(self, max_length: int = 81) -> bool:
        """
        Look at every cell and see if we can create an xy-chain. up to the specified length.
        Returns true if a change is made to the grid.

        An XY chain is a chain of cells, each of which is a cell with two possible values, and in which each cell
        is  neighbor of the previous one and has a digit in common with the previous one.  Given a chain
        AB - BC - CA (a sword), we know that either the first element or the last element must be an A.  Hence any
        cell visible to both of them can't contain A.
        """
        return any(self.__check_xy_chain(cell, value, max_length)
                   for cell in self.grid.cells
                   if len(cell.possible_values) == 2
                   for value in cell.possible_values)

    @staticmethod
    def __check_xy_chain(init_cell: Cell, init_value: int, max_length: int) -> bool:
        todo = deque([(init_cell, init_value, 1)])
        links = {(init_cell, init_value): ((init_cell, init_value), 0)}

        def run_queue() -> bool:
            while todo:
                cell, value, depth = todo.popleft()
                next_value = (cell.possible_values - {value}).pop()
                for next_cell in cell.neighbors:
                    if len(next_cell.possible_values) == 2 and next_value in next_cell.possible_values:
                        if not (next_cell, next_value) in links:
                            new_depth = depth + 1
                            if new_depth < max_length:
                                todo.append((next_cell, next_value, depth + 1))
                            links[(next_cell, next_value)] = ((cell, value), depth + 1)
                            if look_for_cell_to_update(next_cell, next_value, depth + 1):
                                return True
            return False

        def look_for_cell_to_update(next_cell: Cell, next_value: int, depth: int) -> bool:
            if depth >= 3 and init_cell != next_cell and init_cell not in next_cell.neighbors:
                if init_value != next_value and init_value in next_cell.possible_values:
                    # We can remove init_value from any cell that sees both init_cell and next_cell
                    fixers = {cell for cell in init_cell.joint_neighbors(next_cell)
                              if init_value in cell.possible_values}
                    if fixers:
                        print(f'Found an XY chain {chain_to_string(next_cell, next_value)}')
                        Cell.remove_value_from_cells(fixers, init_value)
                        return True
            return False

        def chain_to_string(next_cell: Cell, next_value: int) -> str:
            result = [str(init_value)]
            cell, value = next_cell, next_value
            while True:
                result.append(f'{cell}={cell.possible_value_string()}')
                (cell, value), depth = links[cell, value]
                if depth == 0:
                    return ' '.join(result)

        return run_queue()

    def check_xyz_sword(self) -> bool:
        for triple in self.grid.cells:
            if len(triple.possible_values) == 3:
                possibilities = [cell for cell in triple.neighbors
                                 if len(cell.possible_values) == 2 and cell.possible_values <= triple.possible_values]
                for pair1, pair2 in combinations(possibilities, 2):
                    if pair1.possible_values != pair2.possible_values:
                        common = pair1.possible_values.intersection(pair2.possible_values).pop()
                        fixers = [cell for cell in pair1.joint_neighbors(pair2)
                                  if cell.is_neighbor(triple) and common in cell.possible_values]
                        if fixers:
                            print(
                                f'Found XYZ sword {pair1}={pair1.possible_value_string()}, '
                                f'{pair2}={pair2.possible_value_string()}, '
                                f'{triple}={triple.possible_value_string()}')
                            Cell.remove_value_from_cells(fixers, common)
                            return True
        return False

    def check_chain_colors(self, chains: Chains) -> bool:
        """
        Create strong chains for all the unsolved cells.  See if looking at any two items on the same chain
        yields an insight or contradiction.
        """
        return any(chain.check_colors(self.features) for chain in chains.chains)

    def check_tower(self) -> bool:
        def strong_pair_iterator(cell: Cell, house: House, val: int) -> Iterable[Cell]:
            paired_cell = cell.strong_pair(house, val)
            if paired_cell:
                yield paired_cell

        for cell1 in self.grid.cells:
            if cell1.is_known:
                continue
            for value in cell1.possible_values:
                for house1 in cell1.all_houses():
                    for cell2 in strong_pair_iterator(cell1, house1, value):
                        for house2 in cell2.all_houses_but(house1):
                            for cell3 in cell2.weak_pair(house2, value):
                                for house3 in cell3.all_houses_but(house2):
                                    for cell4 in strong_pair_iterator(cell3, house3, value):
                                        if cell4 in (cell1, cell2, cell3):
                                            continue
                                        fixers = {cell for cell in cell1.joint_neighbors(cell4)
                                                  if value in cell.possible_values}
                                        if fixers:
                                            print(f'Tower on /{value}/ {cell1}={cell2}-{cell3}={cell4}')
                                            Cell.remove_value_from_cells(fixers, value)
                                            return True
        return False

    def draw_grid(self) -> None:
        figure, axes = plt.subplots(1, 1, figsize=(6, 6), dpi=100)

        # set (1,1) as the top-left corner, and (max_column, max_row) as the bottom right.
        axes.axis([1, 10, 10, 1])
        axes.axis('equal')
        axes.axis('off')
        figure.tight_layout()
        _plt = plt

        context = DrawContext(axes)
        for feature in self.features:
            feature.draw(context)

        # Draw the bold outline
        for x in range(1, 11):
            width = 3 if x in (1, 4, 7, 10) else 1
            axes.plot([x, x], [1, 10], linewidth=width, color='black')
            axes.plot([1, 10], [x, x], linewidth=width, color='black')

        given = dict(fontsize=25, color='black', weight='heavy')
        found = dict(fontsize=25, color='blue', weight='bold')
        digit_width = (7/8) / 3
        for cell in self.grid.cells:
            row, column = cell.index
            if cell.known_value:
                args = given if cell.index in self.initial_grid else found
                axes.text(column + .5, row + .5, cell.known_value,
                          verticalalignment='center', horizontalalignment='center', **args)
            else:
                for value in cell.possible_values:
                    y, x = divmod(value - 1, 3)
                    axes.text(column + .5 + (x - 1) * digit_width, row + .5 + (y - 1) * digit_width, str(value),
                              verticalalignment='center', horizontalalignment='center',
                              fontsize=8, color='blue', weight='light')
        plt.show()
