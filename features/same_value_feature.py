from __future__ import annotations

import colorsys
import functools
import operator
from collections import defaultdict
from typing import Optional, ClassVar

from cell import Cell
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from grid import Grid


class SameValueFeature(Feature):
    VERIFY: ClassVar[bool] = False

    squares: list[Square]
    cells: list[Cell]
    features: set[SameValueFeature]

    shared_data: _SameValueSharedData
    __check_cache: list[int]

    @classmethod
    def create(cls, grid: Grid, squares: SquaresParseable, *, name: Optional[str] = None) -> Optional[SameValueFeature]:
        shared_data = _SameValueSharedData.get_unique(grid)
        squares = cls.parse_squares(squares)
        cells = [grid.matrix[square] for square in squares]
        features = [shared_data.cell_to_feature[cell] for cell in cells]
        if not any(features):
            result = SameValueFeature(squares, name=name)
            result.initialize(grid)
            result.start()
            return result
        else:
            main_feature = max(features, key=lambda f: (len(f.cells), f.name))
            for cell in cells:
                if shared_data.cell_to_feature.get(cell) != main_feature:
                    main_feature.__merge_square_into_me(cell)
            return None

    def __init__(self, squares: SquaresParseable, name: Optional[str] = None) -> None:
        self.squares = list(self.parse_squares(squares))
        assert len(self.squares) > 1
        name = name or '='.join(f'r{r}c{c}' for r, c in self.squares)
        self.__check_cache = []
        super().__init__(name=name)

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        self.cells = [self @ square for square in self.squares]
        self.shared_data = _SameValueSharedData.get_unique(grid)
        self.shared_data.owner = self.shared_data.owner or self

    def start(self) -> None:
        shared_data = self.shared_data
        self.shared_data.features.add(self)

        neighbors = frozenset.union(*(cell.neighbors for cell in self.cells))
        self.__set_all_neighbors(neighbors)

        mergers = defaultdict(list)
        for cell in self.cells:
            if cell in shared_data.cell_to_feature:
                mergers[shared_data.cell_to_feature[cell]].append(cell)
        for merged_feature, shared_cells in mergers.items():
            print(f'Merging {merged_feature} into {self} because it shares',
                  ", ".join(str(cell) for cell in shared_cells))
            self.__merge_feature_into_me(merged_feature)

        self.shared_data.cell_to_feature.update((cell, self) for cell in self.cells)
        self.__verify_neighbors()

    def check(self) -> bool:
        print(self.name, self.cells, ','.join(str(cell.possible_values) for cell in self.cells))
        if self not in self.shared_data.features:
            return False
        if not self.cells_changed_since_last_invocation(self.__check_cache, self.cells):
            return False
        return self._check()

    def _check(self):
        result = functools.reduce(operator.__and__, (cell.possible_values for cell in self.cells))
        if len(result) == 1:
            value = result.unique()
            print(f'{self} is known to have the value {value}')
            cells_to_update = [cell for cell in self.cells if not cell.is_known]
            if cells_to_update:
                for cell in cells_to_update:
                    cell.set_value_to(value)
                print(f'  {", ".join(str(cell) for cell in sorted(cells_to_update))} := {value}')
            self.shared_data.features.remove(self)
            [self.shared_data.cell_to_feature.pop(cell) for cell in self.cells]
            return bool(cells_to_update)

        cells_to_update = [cell for cell in self.cells if cell.possible_values != result]
        if cells_to_update:
            Cell.keep_values_for_cell(cells_to_update, result)
            return True

        return False

    def check_special(self) -> bool:
        if self not in self.shared_data.features:
            return False
        return self.__check_all_values_legal_in_all_houses() | self.__check_feature_can_expand()

    def __check_all_values_legal_in_all_houses(self):
        neighbors = self.cells[0].neighbors  # All cells have the same neighbors and the same value
        for value in self.cells[0].possible_values:
            for house in self.grid.houses:
                if value not in house.unknown_values:
                    continue
                value_in_house = {cell for cell in house.unknown_cells if value in cell.possible_values}
                if value_in_house <= neighbors:
                    # If we set ourselves to that value, then every occurrence of that value in the given house
                    # would be eliminated as they are all our neighbors
                    print(f'{self} ≠ {value} because it would eliminate all {value}s from {house}')
                    Cell.remove_value_from_cells(self.cells, value, show=False)
                    return True
        return False

    def __check_feature_can_expand(self) -> bool:
        changed = False
        while self in self.shared_data.features:  # The call to self._check() may make us drop out
            my_neighbors = self.cells[0].neighbors
            my_values = self.cells[0].possible_values
            my_houses = {house for cell in self.cells for house in cell.houses}
            for house in self.grid.houses:
                if house in my_houses:
                    continue
                viable_candidates = [cell for cell in set(house.cells)
                                     if cell not in my_neighbors
                                     if not cell.possible_values.isdisjoint(my_values)]
                assert len(viable_candidates) > 0
                if len(viable_candidates) > 1:
                    continue
                new_cell = viable_candidates.pop()
                print(f'In {house}, {self} must include {new_cell}')
                self.__merge_square_into_me(new_cell)
                self._check()
                self.__verify_neighbors()
                changed = True
                break
            else:
                return changed
        return changed

    def __set_all_neighbors(self, neighbors: frozenset[Cell]):
        assert neighbors.isdisjoint(self.cells)
        for cell in self.cells:
            cell.neighbors = neighbors
        cells_as_list = set(self.cells)
        for cell in neighbors:
            cell.neighbors |= cells_as_list

    def __merge_feature_into_me(self, other: SameValueFeature) -> None:
        self.shared_data.features.remove(other)
        my_squares = set(self.squares)
        added_stuff = [(other_square, other_cell)
                       for other_square, other_cell in zip(other.squares, other.cells)
                       if other_square not in my_squares]
        if added_stuff:
            neighbors = self.cells[0].neighbors | other.cells[0].neighbors
            added_squares, added_cells = zip(*added_stuff)
            self.squares += added_squares
            self.cells += added_cells
            self.__set_all_neighbors(neighbors)
            self.shared_data.cell_to_feature.update((cell, self) for cell in added_cells)

    def __merge_square_into_me(self, cell: Cell):
        cell_feature = self.shared_data.cell_to_feature.get(cell)
        if cell_feature == self:
            pass
        elif cell_feature is not None:
            print(f"Cell {cell} is part of {cell_feature}, so merging the two")
            self.__merge_feature_into_me(cell_feature)
        else:
            neighbors = self.cells[0].neighbors | cell.neighbors
            self.cells += [cell]
            self.squares += [cell.index]
            self.__set_all_neighbors(neighbors)
            self.shared_data.cell_to_feature[cell] = self

    def __verify_neighbors(self) -> None:
        if not self.VERIFY:
            return
        for cell in self.grid.cells:
            for neighbor in cell.neighbors:
                assert cell in neighbor.neighbors
        copy = self.shared_data.cell_to_feature.copy()
        for feature in self.shared_data.features:
            for cell in feature.cells:
                cell_feature = copy.pop(cell)
                assert cell_feature == feature
        assert len(copy) == 0

    def draw(self, context: DrawContext) -> None:
        if not self.shared_data.owner == self:
            return
        for feature in self.shared_data.features:
            if any(not cell.is_known for cell in feature.cells):
                hue = (hash(feature.name) % 1000) / 1000
                color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                for y, x in feature.squares:
                    context.draw_circle((x + .5, y + .2), radius=.1, fill=True, color=color)


class _SameValueSharedData:
    owner: Optional[SameValueFeature]
    features: set[SameValueFeature]
    cell_to_feature: dict[Cell, SameValueFeature]
    grid: Grid

    def __init__(self,  grid: Grid):
        self.features = set()
        self.cell_to_feature = {}
        self.grid = grid
        self.owner = None

    @staticmethod
    def get_unique(grid: Grid) -> _SameValueSharedData:
        key = _SameValueSharedData
        shared_data = grid.get(key)
        if not shared_data:
            shared_data = grid[key] = _SameValueSharedData(grid)
        return shared_data



