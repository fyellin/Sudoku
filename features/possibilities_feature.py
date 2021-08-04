from __future__ import annotations

import abc
import math
from collections import defaultdict
from itertools import chain, combinations, product
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from cell import Cell, CellValue, House, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from grid import Grid
from tools.union_find import Node

Possibility = tuple[int, ...]


class PossibilitiesFeature(Feature, abc.ABC):
    """We are given a set of possible values for a set of cells"""
    squares: Sequence[Square]
    cells: Sequence[Cell]
    cells_as_set: set[Cell]
    possibilities: list[Possibility]
    handle_neighbors: bool
    handle_duplicates: bool
    shared_data: _PossibilitiesSharedData
    __value_only_in_feature: dict[House, SmallIntSet]
    __houses_to_indexes: Mapping[House, list[int]]
    __verified_cells: set[Cell]
    __check_cache: list[int]

    @classmethod
    def create(cls, squares: SquaresParseable, *,
               possibility_function: Callable[[], Iterable[Possibility]],
               name: Optional[str] = None, prefix: Optional[str] = None,
               neighbors: bool = False,
               duplicates: bool = False) -> Sequence[PossibilitiesFeature]:
        return [
            PossibilitiesFeature(squares, possibility_function=possibility_function,
                                 name=name, prefix=prefix,
                                 neighbors=neighbors, duplicates=duplicates)
        ]

    def __init__(self, squares: SquaresParseable, *,
                 possibility_function: Optional[Callable[[], Iterable[Possibility]]] = None,
                 name: Optional[str] = None, prefix: Optional[str] = None,
                 neighbors: bool = False, duplicates: bool = False) -> None:
        super().__init__(name=name, prefix=prefix)
        self.squares = self.parse_squares(squares) if isinstance(squares, str) else squares
        self.possibility_function = possibility_function or self.get_possibilities
        self.handle_neighbors = neighbors
        self.handle_duplicates = duplicates
        self.__value_only_in_feature = defaultdict(SmallIntSet)
        self.__check_cache = []
        self.__verified_cells = set()

    def initialize(self, grid: Grid, synthetic: bool = False) -> None:
        super().initialize(grid)
        self.cells = [grid.matrix[square] for square in self.squares]
        self.shared_data = _PossibilitiesSharedData.get_singleton(grid)
        if not synthetic:
            self.shared_data.owner = self
        self.shared_data.features.add(self)
        self.__finish_initialize()

    def __finish_initialize(self):
        self.cells_as_set = set(self.cells)
        house_to_indexes = defaultdict(list)
        for index, cell in enumerate(self.cells):
            for house in cell.houses:
                house_to_indexes[house].append(index)
        self.__houses_to_indexes = house_to_indexes

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        return ()

    def start(self, verbose: bool = False) -> None:
        possibilities = list(self.possibility_function())
        if self.handle_duplicates:
            possibilities = list(set(possibilities))
        if self.handle_neighbors:
            possibilities = self.__remove_bad_neighbors(possibilities)
        print(f'{self} has {len(possibilities)} possibilities')
        self.possibilities = possibilities
        self.__update_cells_for_possibilities(verbose)

    def check(self) -> bool:
        if self in self.shared_data.features and \
                self.cells_changed_since_last_invocation(self.__check_cache, self.cells):
            old_length = len(self.possibilities)
            if old_length == 1:
                self.shared_data.features.remove(self)
            else:
                # Only keep those possibilities that are still viable
                possibilities = [values for values in self.possibilities
                                 if all(value in square.possible_values for value, square in zip(values, self.cells))]
                if len(possibilities) != old_length:
                    print(f"Possibilities for {self} reduced from {old_length} to {len(possibilities)}")
                    self.possibilities = possibilities
                    change = self.__update_cells_for_possibilities()
                    change |= self.__handle_all_possibilities_use_value_in_house()
                    if change:
                        return True

        if self.shared_data.owner == self:
            return any(feature.check() for feature in self.shared_data.added_features)

    def check_special(self) -> bool:
        if self in self.shared_data.features:
            if self.__handle_value_in_house_only_occurs_in_possibility():
                return True

        if self.shared_data.owner == self:
            if any(feature.check_special() for feature in self.shared_data.added_features):
                return True

        if self.shared_data.owner == self:
            return self.shared_data.check_special()

        return False

    def get_weak_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        cell, value = cell_value
        if self in self.shared_data.features and cell in self.cells_as_set:
            index = self.cells.index(cell)
            # A weak pair says both conditions can't simultaneously be true.  Assume the cell has the indicated index
            # and see which values in other cells are no longer possibilities that had been before.
            possibilities = [possibility for possibility in self.possibilities if possibility[index] == value]
            for index2, cell2 in enumerate(self.cells):
                if index2 == index or cell2.is_known:
                    continue
                legal_values = SmallIntSet(values[index2] for values in possibilities)
                for value2 in cell2.possible_values - legal_values:
                    # By setting cell=value, it is no longer possible that cell2=value2
                    yield CellValue(cell2, value2)

        if self.shared_data.owner == self:
            for feature in self.shared_data.added_features:
                yield from feature.get_weak_pairs(cell_value)

    def simplify(self):
        seen = set()
        deleted = set()
        for index, cell in enumerate(self.cells):
            if cell.is_known or cell in seen:
                deleted.add(index)
            seen.add(cell)

        def trim(a_tuple: Sequence[Node]) -> tuple[Node, ...]:
            return tuple(item for ix, item in enumerate(a_tuple) if ix not in deleted)

        if not deleted:
            return
        self.cells = trim(self.cells)
        self.squares = trim(self.squares)
        self.possibilities = list(map(trim, self.possibilities))
        self.__finish_initialize()


    def __update_cells_for_possibilities(self, show: bool = True) -> bool:
        changed = False
        for index, cell in enumerate(self.cells):
            if cell.is_known:
                if cell not in self.__verified_cells:
                    assert all(values[index] == cell.known_value for values in self.possibilities)
                    self.__verified_cells.add(cell)
                continue
            legal_values = SmallIntSet(values[index] for values in self.possibilities)
            if not cell.possible_values <= legal_values:
                changed = True
                if len(legal_values) == 1:
                    cell.set_value_to(legal_values.unique(), show=True)
                    self.__verified_cells.add(cell)
                else:
                    Cell.keep_values_for_cell([cell], legal_values, show=show)
        return changed

    def __handle_all_possibilities_use_value_in_house(self) -> bool:
        """
        If all possibilities force a specific value for a house to occur in this feature, then that value
        must inside this feature, and all occurrences of that value inside the house but outside the feature can
        be removed.
        """
        change = False
        for house, indexes in self.__houses_to_indexes.items():
            locked_values = house.unknown_values - self.__value_only_in_feature[house]
            for possibility in self.possibilities:
                locked_values &= SmallIntSet(possibility[i] for i in indexes)
                if not locked_values:
                    break

            else:
                self.__value_only_in_feature[house] |= locked_values
                affected_cells = {cell for cell in house.unknown_cells
                                  if cell not in self.cells_as_set
                                  if not cell.possible_values.isdisjoint(locked_values)}
                if affected_cells:
                    plural = "s" if len(locked_values) != 1 else ""
                    print(f'Value{plural} {locked_values} locked into {self} in {house}')
                    Cell.remove_values_from_cells(affected_cells, locked_values)
                    change = True
        return change

    def __handle_value_in_house_only_occurs_in_possibility(self) -> bool:
        """
        If for a specific house, the location of a value doesn't occur outside this feature, then the value must
        occur as part of this feature.  We can prune the possibilities that not put the value somewhere inside
        the house.
        """
        updated = False
        length = len(self.possibilities)
        for house, house_indexes in self.__houses_to_indexes.items():
            # get list of all values that can be found in house cells outside this feature
            found_outside = {value for cell in house.unknown_cells if cell not in self.cells_as_set
                             for value in cell.possible_values}
            # These are the values that must be inside the feature
            required_inside = house.unknown_values - found_outside
            for value in required_inside:
                if value in self.__value_only_in_feature[house]:
                    # We've already ensured that this value occurs only inside this feature.  Ignore
                    continue
                self.__value_only_in_feature[house].add(value)
                self.possibilities = [values for values in self.possibilities
                                      if any(values[i] == value for i in house_indexes)]
                if len(self.possibilities) != length:
                    print(f'For {house}, value {value} only occurs inside {self}')
                    print(f"Possibilities for {self} reduced from {length} to {len(self.possibilities)}")
                    updated = True
                    length = len(self.possibilities)
        if updated:
            return self.__update_cells_for_possibilities()

    def __remove_bad_neighbors(self, possibilities: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]]:
        for (index1, cell1), (index2, cell2) in combinations(enumerate(self.cells), 2):
            if cell1.index == cell2.index:
                # For some reason, we have the same cell repeated twice
                possibilities = [p for p in possibilities if p[index1] == p[index2]]
            elif cell1.is_neighbor(cell2):
                possibilities = [p for p in possibilities if p[index1] != p[index2]]
        return possibilities


class _PossibilitiesSharedData:
    owner: PossibilitiesFeature
    features: set[PossibilitiesFeature]
    grid: Grid
    added_features: list[PossibilitiesFeature]

    @classmethod
    def get_singleton(cls, grid: Grid) -> _PossibilitiesSharedData:
        key = _PossibilitiesSharedData
        result = grid.get(key)
        if result is None:
            grid[key] = result = _PossibilitiesSharedData(grid)
        return result

    def __init__(self, grid: Grid):
        self.features = set()
        self.grid = grid
        self.added_features = []

    def check_special(self) -> bool:
        p_log = {feature: math.log(len(feature.possibilities)) for feature in self.features}

        def closeness(f1: PossibilitiesFeature, f2: PossibilitiesFeature) -> int:
            count = 0
            for cell in f2.cells:
                if cell in f1.cells_as_set:
                    count += 9
                else:
                    count += len(cell.neighbors & f1.cells_as_set)
            return count - p_log[f1] - p_log[f2]

        sorted_features = sorted((x for x in self.features if len(x.possibilities) > 1),
                                 key=lambda f: len(f.possibilities))
        try:
            _count, m1, m2 = max(((closeness(f1, f2), f1, f2) for f1, f2 in combinations(sorted_features[:10], 2)
                                 if len(f1.possibilities) * len(f2.possibilities) <= 10_000_000),
                                 key=lambda x: x[0])
        except ValueError:
            return False

        merged_feature = self.merge_features(m1, m2)
        self.features -= {m1, m2}
        if m1 in self.added_features:
            self.added_features.remove(m1)  # They mey not be there, but we can delete them if they are
        if m2 in self.added_features:
            self.added_features.remove(m2)
        self.added_features.append(merged_feature)
        return True

    def merge_features(self, feature1: PossibilitiesFeature, feature2: PossibilitiesFeature) -> PossibilitiesFeature:
        length1, length2 = len(feature1.possibilities), len(feature2.possibilities)

        def possibility_function() -> Iterable[Possibility]:
            index = {cell: index for index, cell in enumerate(feature1.cells)}
            results = (p1 + p2 for p1, p2 in product(feature1.possibilities, feature2.possibilities))
            # A simplified version of handling neighbors, since we only need to cross check between the features.
            for index2, cell2 in enumerate(feature2.cells, start=len(feature1.cells)):
                if (index1 := index.get(cell2)) is not None:
                    results = [possibility for possibility in results if possibility[index1] == possibility[index2]]
                else:
                    for cell1 in feature1.cells_as_set & cell2.neighbors:
                        index1 = index[cell1]
                        results = [possibility for possibility in results if possibility[index1] != possibility[index2]]
            length3 = len(results)
            temp = length3 * 100.0 / (length1 * length2)
            print(f'Merge {feature1} ({length1}) x {feature2} ({length2}) = {length1 * length2} '
                  f'--> {length3} {temp:.2f}%')
            return results

        owner = self.owner
        result = PossibilitiesFeature(tuple(chain(feature1.squares, feature2.squares)),
                                      prefix="Merge", possibility_function=possibility_function)
        result.initialize(self.grid, synthetic=True)
        assert self.owner == owner
        result.start(True)
        result.check()
        result.check_special()
        result.simplify()
        return result


class GroupedPossibilitiesFeature(Feature, abc.ABC):
    """We are given a set of possible values for a set of cells"""
    squares: Sequence[Square]
    cells: Sequence[Cell]
    possibilities: list[tuple[SmallIntSet, ...]]
    handle_neighbors: bool
    compressed: bool
    __check_cache: list[int]

    def __init__(self, squares: SquaresParseable, *,
                 name: Optional[str] = None, neighbors: bool = False, compressed: bool = False) -> None:
        super().__init__(name=name)
        if isinstance(squares, str):
            squares = self.parse_squares(squares)
        self.squares = squares
        self.handle_neighbors = neighbors
        self.compressed = compressed
        self.__check_cache = []

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        self.cells = [grid.matrix[square] for square in self.squares]

    @abc.abstractmethod
    def get_possibilities(self) -> list[tuple[Union[SmallIntSet, Iterable[int], int], ...]]:
        ...

    def start(self) -> None:
        def fixit_one(x: Union[SmallIntSet, Iterable[int], int]) -> SmallIntSet:
            if isinstance(x, int):
                return SmallIntSet([x])
            elif isinstance(x, SmallIntSet):
                return x
            else:
                return SmallIntSet(x)

        def fixit(items: tuple[Union[SmallIntSet, Iterable[int], int], ...]) -> tuple[SmallIntSet, ...]:
            return tuple(fixit_one(item) for item in items)

        possibilities = list(fixit(x) for x in self.get_possibilities())
        if self.handle_neighbors:
            possibilities = self.__remove_bad_neighbors(possibilities)
        print(f'{self} has {len(possibilities)} possibilities')
        self.possibilities = possibilities
        self.__update_for_possibilities(False)

    def check(self) -> bool:
        if not self.cells_changed_since_last_invocation(self.__check_cache, self.cells):
            return False

        old_length = len(self.possibilities)
        if old_length == 1:
            return False

        # Only keep those possibilities that are still available
        def is_viable(possibility: tuple[SmallIntSet, ...]) -> bool:
            choices = [value & square.possible_values for (value, square) in zip(possibility, self.cells)]
            if not all(choices):
                return False
            if self.compressed:
                open_choices = [choice for choice, cell in zip(choices, self.cells) if not cell.is_known]
                for length in range(2, len(open_choices)):
                    for subset in combinations(open_choices, length):
                        if len(SmallIntSet.union(*subset)) < length:
                            return False
            return True

        self.possibilities = list(filter(is_viable, self.possibilities))
        if len(self.possibilities) < old_length:
            print(f"Possibilities for {self} reduced from {old_length} to {len(self.possibilities)}")
            return self.__update_for_possibilities()
        return False

    def __update_for_possibilities(self, show: bool = True) -> bool:
        updated = False
        for index, cell in enumerate(self.cells):
            if cell.is_known:
                continue
            legal_values = SmallIntSet.union(*[possibility[index] for possibility in self.possibilities])
            if not cell.possible_values <= legal_values:
                updated = True
                Cell.keep_values_for_cell([cell], legal_values, show=show)
        return updated

    def __remove_bad_neighbors(self, possibilities: Sequence[tuple[SmallIntSet, ...]]
                               ) -> list[tuple[set[int], ...]]:
        for (index1, cell1), (index2, cell2) in combinations(enumerate(self.cells), 2):
            if cell1.is_neighbor(cell2):
                possibilities = [p for p in possibilities if len(p[index1]) > 1 or p[index1] != p[index2]]
            elif cell1.index == cell2.index:
                #  We're not sure if this works or not
                possibilities = [p for p in possibilities if p[index1] == p[index2]]
        return possibilities

    def to_possibility_feature(self):
        parent = self

        class ChildPossibilityFeature(PossibilitiesFeature):
            def __init__(self):
                super().__init__(parent.squares, name=parent.name, neighbors=True, duplicates=True)

            def draw(self, context: DrawContext):
                parent.draw(context)

            def get_possibilities(self) -> Iterable[tuple[int, ...]]:
                for element in parent.get_possibilities():
                    yield from product(*element)

        return ChildPossibilityFeature()
