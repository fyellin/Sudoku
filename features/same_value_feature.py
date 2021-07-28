from __future__ import annotations

import functools
import itertools
import operator
from collections import deque
from typing import Optional, ClassVar, Sequence

from cell import Cell
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from grid import Grid
from tools.union_find import UnionFind


class SameValueFeature(Feature):
    VERIFY: ClassVar[bool] = False

    squares: list[Square]
    cells: list[Cell]
    color: Optional[str]
    is_assigned: bool
    has_real_name: bool

    shared_data: _SameValueSharedData
    __check_cache: list[int]

    @classmethod
    def already_paired(cls, grid: Grid, cell1, cell2: Cell):
        shared_data = _SameValueSharedData.get(grid)
        feature1 = shared_data.cell_to_feature(cell1)
        return feature1 is not None and feature1 == shared_data.cell_to_feature(cell2)

    @classmethod
    def create(cls, grid: Grid, cells: Sequence[Cell], *, name: Optional[str] = None, prefix: Optional[str] = None) -> \
            tuple[Optional[SameValueFeature], bool]:

        shared_data = _SameValueSharedData.get(grid)
        features = [shared_data.cell_to_feature(cell) for cell in cells]
        if features[0] is not None and all(features[0] == feature for feature in features):
            return None, False

        result = SameValueFeature('', name=name, prefix=prefix, cells=cells)
        result.initialize(grid)
        result.start()
        if result in shared_data.features:
            return result, True
        else:
            return None, True

    def __init__(self, squares: SquaresParseable, name: Optional[str] = None, *, prefix: Optional[str] = None,
                 cells: Sequence[Cell] = ()) -> None:
        if cells:
            assert not squares
            self.squares = []
            self.cells = list(cells)
        else:
            self.squares = list(self.parse_squares(squares))
            self.cells = []
        self.color = None
        self.is_assigned = False
        self.__check_cache = []
        self.has_real_name = name is not None or prefix is not None
        super().__init__(name=name, prefix=prefix)

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        if not self.cells:
            self.cells = [self @ square for square in self.squares]
        self.shared_data = _SameValueSharedData.get(grid)
        self.shared_data.owner = self.shared_data.owner or self

    def start(self) -> None:
        neighbors = frozenset.union(*(cell.neighbors for cell in self.cells))
        self.set_all_neighbors(neighbors)
        self.shared_data.add_feature(self)

    def check(self) -> bool:
        if not self.is_assigned and self in self.shared_data.features:
            if self.cells_changed_since_last_invocation(self.__check_cache, self.cells):
                if self.__check():
                    return True
        return False

    def __check(self):
        result = functools.reduce(operator.__and__, (cell.possible_values for cell in self.cells))
        value = result.unique() if len(result) == 1 else None
        if value is not None:
            cells_to_update = [cell for cell in self.cells if not cell.is_known]
            self.is_assigned = True
        else:
            cells_to_update = [cell for cell in self.cells if cell.possible_values != result]
        if cells_to_update:
            if value:
                print(f"All cells of {self} have the value {value}")
                [cell.set_value_to(value) for cell in cells_to_update]
                print(f'  {", ".join(str(cell) for cell in sorted(cells_to_update))} := {value}')
            else:
                print("Intersection of possible values for {self}")
                Cell.keep_values_for_cell(cells_to_update, result)
            return True
        return False

    def check_special(self) -> bool:
        if not self.is_assigned and self in self.shared_data.features:
            if self.__check_all_values_legal_in_all_houses() | self.__check_try_to_expand_feature():
                return True

        return False

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

    def __check_try_to_expand_feature(self) -> bool:
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
                candidate = viable_candidates.pop()
                print(f'In {house}, {self} must also include {candidate}')
                fake_feature = SameValueFeature('', cells=(self.cells[0], candidate), name=f'[+= {candidate}]')
                fake_feature.initialize(self.grid)
                fake_feature.start()
                assert fake_feature not in self.shared_data.features
                self.__check()
                changed = True
                break
            else:
                return changed
        return changed

    def __str__(self):
        temp = '='.join(str(cell) for cell in self.cells)
        if self.has_real_name:
            return self.name + " " + temp
        else:
            return temp

    def __repr__(self):
        return str(self)

    def set_all_neighbors(self, neighbors: frozenset[Cell]):
        assert neighbors.isdisjoint(self.cells)
        for cell in self.cells:
            cell.neighbors = neighbors
        cells_as_list = set(self.cells)
        for cell in neighbors:
            cell.neighbors |= cells_as_list

    def draw(self, context: DrawContext) -> None:
        if self not in self.shared_data.features:
            return
        if all(cell.is_known for cell in self.cells):
            return
        self.color = self.color or self.shared_data.get_next_color()
        for cell in self.cells:
            y, x = cell.index
            context.draw_circle((x + .5, y + .2), radius=.1, fill=True, color=self.color)


class _SameValueSharedData:
    VERIFY = True

    COLORS = ('#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
              '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
              '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000')

    features: dict[SameValueFeature, int]   # Using the fact that dictionaries are Ordered
    union_find: UnionFind
    token_to_feature: dict[Cell, SameValueFeature]
    grid: grid

    owner: Optional[SameValueFeature]
    colors: deque[str]

    def __init__(self,  grid: Grid):
        self.union_find = UnionFind()
        self.token_to_feature = {}
        self.features = {}
        self.grid = grid

        self.owner = None
        self.colors = deque(self.COLORS)

    @staticmethod
    def get(grid: Grid) -> _SameValueSharedData:
        key = _SameValueSharedData
        shared_data = grid.get(key)
        if not shared_data:
            shared_data = grid[key] = _SameValueSharedData(grid)
        return shared_data

    def add_feature(self, feature: SameValueFeature) -> None:
        cell0 = feature.cells[0]
        for i in range(1, len(feature.cells)):
            self.union_find.union(cell0, feature.cells[i])
        self.features[feature] = 0
        self.token_to_feature[self.union_find.find(cell0)] = feature
        self.__fixup()

    def cell_to_feature(self, cell: Cell) -> Optional[SameValueFeature]:
        token = self.union_find.find(cell)
        return self.token_to_feature.get(token)

    def get_next_color(self):
        return self.colors.popleft()

    def __fixup(self):
        deletions = []
        self.token_to_feature.clear()
        for feature in self.features:
            token = self.union_find.find(feature.cells[0])
            prev_feature = self.token_to_feature.get(token)
            if not prev_feature:
                self.token_to_feature[token] = feature
            else:
                old_name = str(prev_feature)
                neighbors = feature.cells[0].neighbors | prev_feature.cells[0].neighbors
                prev_feature.set_all_neighbors(neighbors)
                prev_feature.cells = list(unique_everseen(itertools.chain(prev_feature.cells, feature.cells)))
                print(f'...Merging {feature} into {old_name} yielding {prev_feature}')
                deletions.append(feature)
        for feature in deletions:
            self.features.pop(feature)
            if feature.color:
                self.colors.append(feature.color)
        if self.VERIFY:
            self.__verify()

    def __verify(self) -> None:
        for cell in self.grid.cells:
            assert cell not in cell.neighbors
            for neighbor in cell.neighbors:
                assert cell in neighbor.neighbors

        nodes = set(self.union_find.all_nodes())
        for feature in self.features:
            tokens = [self.union_find.find(cell) for cell in feature.cells]
            token = tokens.pop()
            assert all(t == token for t in tokens)
            assert self.token_to_feature[token] == feature
            cells = set(feature.cells)
            assert cells <= nodes
            nodes -= cells


def unique_everseen(iterable):
    seen = set()
    seen_add = seen.add
    for element in itertools.filterfalse(seen.__contains__, iterable):
        seen_add(element)
        yield element



