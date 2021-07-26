from __future__ import annotations

from collections import deque, defaultdict
from collections.abc import Sequence, Mapping
from itertools import product, combinations, permutations

from matplotlib import pyplot as plt

from cell import House, Cell, SmallIntSet
from chain import Chains
from draw_context import DrawContext
from feature import Feature
from grid import Grid
from hard_medusa import HardMedusa


class Sudoku:
    grid: Grid
    features: Sequence[Feature]
    initial_grid: Mapping[tuple[int, int], int]
    draw_verbose: bool
    __check_hidden_singles_cache: dict[House, list[int]]

    def solve(self, puzzle: str, *, features: Sequence[Feature] = (),
              initial_only: bool = False,
              medusa: bool = False,
              draw_verbose: bool = False) -> bool:
        self.features = features
        self.grid = grid = Grid(features)
        self.draw_verbose = draw_verbose
        grid.start()
        self.initial_grid = {(row, column): int(letter)
                             for (row, column), letter in zip(product(range(1, 10), repeat=2), puzzle)
                             if '1' <= letter <= '9'}

        for square, value in self.initial_grid.items():
            grid.matrix[square].set_value_to(value)

        self.grid.print()
        self.draw_grid()
        if initial_only:
            return True
        try:
            return self.run_solver(medusa=medusa)
        except Exception:
            self.draw_grid()
            raise

    def run_solver(self, *, medusa: bool) -> bool:
        self.__check_hidden_singles_cache = defaultdict(list)
        checking_features = [f for f in self.features if f.has_check_method()]

        while True:
            if self.is_solved():
                self.draw_grid(done=True, result=True)
                return True
            if self.check_naked_singles() or self.check_hidden_singles():
                continue
            if any(feature.check() for feature in checking_features):
                continue
            if self.check_intersection_removal():
                continue
            if self.check_tuples():
                continue
            if any(feature.check_special() for feature in checking_features):
                continue

            print()
            self.grid.print()
            self.draw_grid()
            print()

            if self.check_fish() or self.check_xy_sword() or self.check_xyz_sword() or self.check_tower():
                continue
            if self.check_xy_chain(81):
                continue
            if self.check_tower_extended():
                continue
            if medusa:
                chains = Chains.create(self.grid.cells, True)
                if self.check_chain_colors(chains):
                    continue
                if HardMedusa.run(chains, self.features):
                    continue

            self.draw_grid(done=True, result=False)
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
            output = [cell.set_value_to(cell.possible_values.unique()) for cell in sorted(naked_singles)]
            print("Naked Single: " + '; '.join(output))
        return found_naked_single

    def check_hidden_singles(self) -> bool:
        """
        Finds a house for which there is only one place that one or more digits can go.
        Returns true if it finds such a house.
        """
        def check_house(house: House) -> bool:
            changed = False
            mapper = defaultdict(list)
            for cell in house.unknown_cells:
                for value in cell.possible_values:
                    mapper[value].append(cell)
            for value, cells in mapper.items():
                if len(cells) == 1:
                    print(f'Hidden Single: {house} = {value} must be {cells[0]}')
                    cells[0].set_value_to(value)
                    changed = True
            return changed

        return any(check_house(house) for house in self.grid.houses)

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

        fixers = {cell for cell in cell0.get_all_neighbors_for_value(value) if value in cell.possible_values}
        fixers.intersection_update(*(cell.get_all_neighbors_for_value(value) for cell in other_candidates))
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
        return any(self.__check_tuples(house, SmallIntSet(values))
                   # Specifically find all tuples of 2 before trying all tuples of 3, . . . .
                   for count in range(2, 9)
                   # Look at each house
                   for house in self.grid.houses
                   if len(house.unknown_values) > count
                   # Look at each subset of size "count" of the unknown values of that house
                   for values in combinations(house.unknown_values, count))

    @staticmethod
    def __check_tuples(house: House, values: SmallIntSet) -> bool:
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
        cell_to_house = {(cell, house.house_type): house
                         for house in self.grid.houses for cell in house.unknown_cells}

        for value in range(1, 10):
            # Find all houses for which the value is missing
            houses = [house for house in self.grid.houses if value in house.unknown_values]
            if not houses:
                continue
            # For convenience, make a map from each "empty" house to the cells in that house that can contain the value
            house_to_cells = {house: [cell for cell in house.unknown_cells if value in cell.possible_values]
                              for house in houses}
            # Look for a fish between any two House types on the specified value
            # noinspection PyTypeChecker
            house_types = (House.Type.COLUMN, House.Type.ROW, House.Type.BOX)
            for this_house_type, that_house_type in permutations(house_types, 2):
                if self.__check_fish(value, houses, house_to_cells, cell_to_house,
                                     this_house_type, that_house_type):
                    return True
        return False

    @staticmethod
    def __check_fish(value: int,
                     houses: Sequence[House],
                     house_to_cells: Mapping[House, Sequence[Cell]],
                     cell_to_house: Mapping[tuple[Cell, House.Type], House],
                     this_house_type: House.Type,
                     that_house_type: House.Type) -> bool:
        these_unknown_houses = [house for house in houses if house.house_type == this_house_type]
        those_unknown_houses = [house for house in houses if house.house_type == that_house_type]
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
                row_cells = {cell for house in rows for cell in house_to_cells[house]}
                # Find the columns that those cells belong to
                columns = {cell_to_house[cell, that_house_type] for cell in row_cells}
                assert len(columns) >= number_rows_to_choose
                if len(columns) > number_rows_to_choose:
                    continue
                # If len(columns) == number_rows_to_choose, we have a fish.  Let's see if there is something to delete.
                # Find all the cells in those columns
                column_cells = {cell for house in columns for cell in house_to_cells[house]}
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
                next_value = (cell.possible_values - {value}).unique()
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
                        # pair1 and pair2 have precisely one common value, by definition.
                        common = (pair1.possible_values & pair2.possible_values).unique()
                        # We can remove this common digit from anything that sees all three cells
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
        return any(chain.check_colors() for chain in chains.chains)

    def check_tower(self) -> bool:
        for cell1 in self.grid.cells:
            if cell1.is_known:
                continue
            for value in cell1.possible_values:
                for cell2, house2 in cell1.strong_pair(value):
                    for cell3, house3 in cell2.weak_pair(value):
                        if house3 == house2 or cell3 in (cell1, cell2):
                            continue
                        for cell4, house4 in cell3.strong_pair(value):
                            if house4 == house3 or cell4 in (cell1, cell2, cell3):
                                continue
                            fixers = {cell for cell in cell1.joint_neighbors(cell4)
                                      if value in cell.possible_values}
                            if fixers:
                                print(f'Tower: {cell1}≠{value} → {cell2}={value} → {cell3}≠{value} → {cell4}={value}. '
                                      f'So {cell1}={value} or {cell4}={value}.')
                                print(f'Tower on /{value}/ {cell1}={cell2}-{cell3}={cell4}')
                                Cell.remove_value_from_cells(fixers, value)
                                return True
        return False

    def check_tower_extended(self) -> bool:
        for cell1 in self.grid.cells:
            if cell1.is_known:
                continue
            for value1 in cell1.possible_values:
                for cell2, value2, house2 in cell1.extended_strong_pair(value1):
                    for cell3, value3, house3 in cell2.extended_weak_pair(value2):
                        if house3 is house2 or (cell3, value3) == (cell1, value1):
                            continue
                        for cell4, value4, house4 in cell3.extended_strong_pair(value3):
                            if house4 is house3 or (cell4, value4) in ((cell2, value2), (cell1, value1)):
                                continue
                            fixers = []

                            def print_tower():
                                print(f'Tower: {cell1}≠{value1} → {cell2}={value2} → {cell3}≠{value3} → {cell4}={value4}. '
                                      f'So {cell1}={value1} or {cell4}={value4}.')
                            # At least one of cell1 == value1 or cell4 == value4 is True
                            if cell1 == cell4:  # The cell must have one of the other two values
                                temp = SmallIntSet([value1, value4])
                                assert temp <= cell1.possible_values
                                if temp != cell1.possible_values:
                                    print_tower()
                                    Cell.keep_values_for_cell([cell1], {value1, value4})
                                    return True
                            elif value1 == value4:
                                # TODO:  Joint neighbors for value??
                                fixers = {(cell, value1) for cell in cell1.joint_neighbors(cell4)
                                          if value1 in cell.possible_values}
                            else:
                                if value1 in cell4.possible_values and cell1.is_neighbor_for_value(cell4, value1):
                                    fixers.append([cell4, value1])
                                if value4 in cell1.possible_values and cell4.is_neighbor_for_value(cell1, value4):
                                    fixers.append([cell1, value4])
                            if fixers:
                                print_tower()
                                for cell, value in fixers:
                                    Cell.remove_value_from_cells([cell], value, show=True)
                                return True
        return False

    def draw_grid(self, *, done: bool = False, result: bool = False):
        figure, axes = plt.subplots(1, 1, figsize=(6, 6), dpi=100)

        # set (1,1) as the top-left corner, and (max_column, max_row) as the bottom right.
        axes.axis([1, 10, 10, 1])
        axes.axis('equal')
        axes.axis('off')
        figure.tight_layout()

        context = DrawContext(axes, done=done, result=result)
        for feature in self.features:
            feature.draw(context)

        # Draw the bold outline
        for x in range(1, 11):
            width = 3 if x in (1, 4, 7, 10) and context.draw_normal_boxes else 1
            axes.plot([x, x], [1, 10], linewidth=width, color='black')
            axes.plot([1, 10], [x, x], linewidth=width, color='black')

        if self.draw_verbose:
            self.__fill_in_grid_verbose(axes)
        else:
            self.__fill_in_grid_simple(axes)

        plt.show()

    def __fill_in_grid_verbose(self, axes) -> None:
        given = dict(fontsize=25, color='black', weight='heavy')
        found = dict(fontsize=25, color='blue', weight='bold')
        digit_width = (7/8) / 3
        for row, column in product(range(1, 10), repeat=2):
            cell = self.grid.matrix[row, column]
            if cell.known_value:
                args = given if (row, column) in self.initial_grid else found
                axes.text(column + .5, row + .5, cell.known_value,
                          verticalalignment='center', horizontalalignment='center', **args)
            else:
                for value in cell.possible_values:
                    y, x = divmod(value - 1, 3)
                    axes.text(column + .5 + (x - 1) * digit_width, row + .5 + (y - 1) * digit_width, str(value),
                              verticalalignment='center', horizontalalignment='center',
                              fontsize=8, color='blue', weight='light')

    def __fill_in_grid_simple(self, axes) -> None:
        given = dict(fontsize=25, color='black', weight='heavy')
        found = dict(fontsize=25, color='blue', weight='bold')
        corner_args = dict(fontsize=8, color='blue', weight='light')

        for row, column in product(range(1, 10), repeat=2):
            cell = self.grid.matrix[row, column]
            if cell.known_value:
                args = given if (row, column) in self.initial_grid else found
                axes.text(column + .5, row + .5, cell.known_value,
                          verticalalignment='center', horizontalalignment='center', **args)
            else:
                if len(cell.possible_values) <= 8:
                    axes.text(column + .5, row + .5, ''.join(str(x) for x in sorted(cell.possible_values)),
                              verticalalignment='center', horizontalalignment='center', **corner_args)

        for house in self.grid.houses:
            if house.house_type != House.Type.BOX:
                continue
            cells_to_corners = defaultdict(list)
            # Look at the all unknown values that can only fit into three or fewer cells.  We put
            # these into the corner, unless all the cells have been handled above
            for value in house.unknown_values:
                cells = {cell for cell in house.unknown_cells if value in cell.possible_values}
                if len(cells) <= 3 and any(len(cell.possible_values) > len(cells) for cell in cells):
                    for cell in cells:
                        cells_to_corners[cell].append(value)
            for cell, values in cells_to_corners.items():
                row, column = cell.index
                for i, value in enumerate(sorted(values)):
                    if i == 0:
                        axes.text(column + .1, row + .1, str(value),
                                  verticalalignment='top', horizontalalignment='left', **corner_args)
                    elif i == 1:
                        axes.text(column + .9, row + .1, str(value),
                                  verticalalignment='top', horizontalalignment='right', **corner_args)
                    elif i == 2:
                        axes.text(column + .1, row + .9, str(value),
                                  verticalalignment='bottom', horizontalalignment='left', **corner_args)
                    else:
                        axes.text(column + .9, row + .9, str(value),
                                  verticalalignment='bottom', horizontalalignment='right', **corner_args)
