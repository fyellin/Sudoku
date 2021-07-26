from __future__ import annotations

import colorsys
import functools
import operator
from collections import Sequence, defaultdict
from typing import Optional

from cell import Cell
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from grid import Grid


class SameValueFeature(Feature):
    squares: list[Square]
    cells: list[Cell]
    features: set[SameValueFeature]

    master: Optional[_SaveValueSupervisor]
    __check_cache: list[int]

    def __init__(self, squares: SquaresParseable, name: Optional[str] = None) -> None:
        self.squares = list(self.parse_squares(squares))
        assert len(self.squares) > 1
        name = name or '='.join(f'r{r}c{c}' for r, c in self.squares)
        self.master = None
        self.__check_cache = []
        super().__init__(name=name)

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        self.cells = [self @ square for square in self.squares]
        features = self.grid.get((self.__class__, "feature"), None)
        if not features:
            self.grid[(self.__class__, "feature")] = features = set()
            self.master = _SaveValueSupervisor(features, grid)
        features.add(self)
        self.features = features

    def start(self) -> None:
        if self.master:
            self.master.start()

    def check(self) -> bool:
        if self not in self.features:
            return False
        if not self.cells_changed_since_last_invocation(self.__check_cache, self.cells):
            return False
        result = functools.reduce(operator.__and__, (cell.possible_values for cell in self.cells))
        cells_to_update = [cell for cell in self.cells if cell.possible_values != result]
        if cells_to_update:
            Cell.keep_values_for_cell(cells_to_update, result)
            return True
        return False

    def check_special(self) -> bool:
        if self not in self.features:
            return False
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

    def draw(self, context: DrawContext) -> None:
        if not self.master:
            return
        for feature in self.features:
            hue = (hash(feature.name) % 1000) / 1000
            color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            for y, x in feature.squares:
                context.draw_circle((x + .5, y + .2), radius=.1, fill=True, color=color)


class _SaveValueSupervisor:
    features: set[SameValueFeature]
    grid: Grid

    def __init__(self, features: set[SameValueFeature], grid: Grid):
        self.features = features
        self.grid = grid

    def start(self):
        self.verify_neighbors()
        self.merge_overlapping_features_and_set_neighbors()
        self.expand_features()
        self.verify_neighbors()

    def merge_overlapping_features_and_set_neighbors(self):
        cell_to_feature = {}
        features = self.features
        for feature in sorted(features, key=lambda f: len(f.cells)):
            neighbors = frozenset.union(*(cell.neighbors for cell in feature.cells))
            self.set_all_neighbors(feature, neighbors)
            merged_feature_and_cells = defaultdict(list)
            for cell in feature.cells:
                if cell in cell_to_feature:
                    merged_feature_and_cells[cell_to_feature[cell]].append(cell)
            for merged_feature, shared_cells in merged_feature_and_cells.items():
                print(f'Merging {merged_feature} into {feature} because it shares',
                      ", ".join(str(cell) for cell in shared_cells))
                self.merge_into_feature(feature, merged_feature)
            cell_to_feature.update((cell, feature) for cell in feature.cells)

    def expand_features(self):
        changed = True
        while changed:
            changed = False
            for feature in sorted(self.features, key=lambda f: len(f.cells), reverse=True):
                if feature in self.features:
                    changed |= self.expand_one_feature(feature)

    def expand_one_feature(self, feature):
        changed = False
        while True:
            neighbors = feature.cells[0].neighbors
            my_houses = {house for cell in feature.cells for house in cell.houses}
            for house in feature.grid.houses:
                if house in my_houses:
                    continue
                temp = set(house.cells) - neighbors
                assert len(temp) > 0
                if len(temp) > 1:
                    continue
                new_cell = temp.pop()
                other_feature = next((feature for feature in self.features if new_cell in feature.cells), None)
                if other_feature:
                    print(f'In {house}, {feature} must include {new_cell}, part of {other_feature}')
                    self.merge_into_feature(feature, other_feature)
                else:
                    print(f'In {house}, {feature} must include {new_cell}')
                    self.add_square_to_feature(feature, new_cell)
                changed = True
                break
            else:
                return changed

    def add_square_to_feature(self, feature, cell: Cell) -> None:
        assert cell not in feature.cells
        neighbors = feature.cells[0].neighbors | cell.neighbors
        feature.cells += [cell]
        feature.squares += [cell.index]
        self.set_all_neighbors(feature, neighbors)

    def merge_into_feature(self, feature: SameValueFeature, other_feature: SameValueFeature) -> None:
        self.features.remove(other_feature)
        my_squares = set(feature.squares)
        added_stuff = [(other_square, other_cell)
                       for other_square, other_cell in zip(other_feature.squares, other_feature.cells)
                       if other_square not in my_squares]
        if added_stuff:
            neighbors = feature.cells[0].neighbors | other_feature.cells[0].neighbors
            added_squares, added_cells = zip(*added_stuff)
            feature.squares += added_squares
            feature.cells += added_cells
            self.set_all_neighbors(feature, neighbors)

    @staticmethod
    def set_all_neighbors(feature, neighbors: frozenset[Cell]):
        assert neighbors.isdisjoint(feature.cells)
        for cell in feature.cells:
            cell.neighbors = neighbors
        cells_as_list = set(feature.cells)
        for cell in neighbors:
            cell.neighbors |= cells_as_list

    def verify_neighbors(self) -> None:
        for cell in self.grid.cells:
            for neighbor in cell.neighbors:
                assert cell in neighbor.neighbors
