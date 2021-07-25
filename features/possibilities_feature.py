from __future__ import annotations

import abc
from collections import defaultdict
from itertools import combinations, product
from typing import Sequence, Mapping, Union, Optional, Iterable

from cell import Cell, House, SmallIntSet
from feature import Feature, Square
from grid import Grid


class PossibilitiesFeature(Feature, abc.ABC):
    """We are given a set of possible values for a set of cells"""
    squares: Sequence[Square]
    cells: Sequence[Cell]
    possibilities: list[tuple[int, ...]]
    handle_neighbors: bool
    handle_duplicates: bool
    _houses_to_indexes: Mapping[House, list[int]]
    _cells_as_set: set[Cell]
    _value_only_in_feature: dict[House, SmallIntSet]
    is_primary: bool

    def __init__(self, squares: Union[Sequence[Square], str], *,
                 name: Optional[str] = None, neighbors: bool = False, duplicates: bool = False) -> None:
        super().__init__(name=name)
        self.squares = self.parse_squares(squares) if isinstance(squares, str) else squares
        self.handle_neighbors = neighbors
        self.handle_duplicates = duplicates
        self._value_only_in_feature = defaultdict(SmallIntSet)
        self.is_primary = False

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        self.cells = [grid.matrix[square] for square in self.squares]
        self._cells_as_set = set(self.cells)
        house_to_indexes = defaultdict(list)
        for index, cell in enumerate(self.cells):
            for house in cell.houses:
                house_to_indexes[house].append(index)
        self._houses_to_indexes = house_to_indexes
        features = self.grid.get((self.__class__, "feature"), None)
        if not features:
            self.is_primary = True
            self.grid[(self.__class__, "feature")] = features = set()
        features.add(self)

    def __initialize_after_merge(self, other: PossibilitiesFeature) -> None:
        self.cells += other.cells
        self._cells_as_set = set(self.cells)
        house_to_indexes = defaultdict(list)
        for index, cell in enumerate(self.cells):
            for house in cell.houses:
                house_to_indexes[house].append(index)
        self._houses_to_indexes = house_to_indexes
        for house, values in other._value_only_in_feature.items():
            self._value_only_in_feature[house] |= values

    @abc.abstractmethod
    def get_possibilities(self) -> Iterable[tuple[int, ...]]: ...

    def start(self) -> None:
        possibilities = list(self.get_possibilities())
        if self.handle_duplicates:
            possibilities = list(set(possibilities))
        if self.handle_neighbors:
            possibilities = self.__remove_bad_neighbors(possibilities)
        print(f'{self} has {len(possibilities)} possibilities')
        self.possibilities = possibilities
        self.__update_for_possibilities(False)

    @Feature.check_only_if_changed
    def check(self) -> bool:
        features: set[PossibilitiesFeature] = self.grid[self.__class__, "feature"]
        if self not in features:
            return False
        old_length = len(self.possibilities)
        if old_length == 1:
            features.remove(self)
            return False

        # Only keep those possibilities that are still viable
        self.possibilities = [values for values in self.possibilities
                              if all(value in square.possible_values for value, square in zip(values, self.cells))]

        if len(self.possibilities) == old_length:
            return False
        else:
            print(f"Possibilities for {self} reduced from {old_length} to {len(self.possibilities)}")
            change = self.__update_for_possibilities()

        change |= self.__handle_all_possibilities_use_value()
        return change

    def check_special(self) -> bool:
        features: set[PossibilitiesFeature] = self.grid[self.__class__, "feature"]

        if self in features and self.__force_value_into_feature():
            return True
        if self.is_primary and self.check_join_two():
            return True
        return False

    def __update_for_possibilities(self, show: bool = True) -> bool:
        updated = False
        for index, cell in enumerate(self.cells):
            if cell.is_known:
                continue
            legal_values = SmallIntSet(values[index] for values in self.possibilities)
            if not cell.possible_values <= legal_values:
                updated = True
                Cell.keep_values_for_cell([cell], legal_values, show=show)
        return updated

    def __handle_all_possibilities_use_value(self) -> bool:
        """
        If all possibilities force a specific value for a house to occur in this feature, then that value
        must inside this feature, and all occurrences of that value inside the house but outside the feature can
        be removed.
        """
        change = False
        for house, indexes in self._houses_to_indexes.items():
            locked_values = house.unknown_values - self._value_only_in_feature[house]
            for possibility in self.possibilities:
                locked_values &= SmallIntSet(possibility[i] for i in indexes)
                if not locked_values:
                    break

            else:
                self._value_only_in_feature[house] |= locked_values
                affected_cells = {cell for cell in house.unknown_cells
                                  if cell not in self._cells_as_set
                                  if not cell.possible_values.isdisjoint(locked_values)}
                if affected_cells:
                    print(f'Values {locked_values} locked into {self} in {house}')
                    Cell.remove_values_from_cells(affected_cells, locked_values)
                    change = True
        return change

    def __force_value_into_feature(self) -> bool:
        """
        If for a specific house, the location of a value doesn't occur outside this feature, then the value must
        occur as part of this feature.  We can prune the possibilities that not put the value somewhere inside
        the house.
        """
        updated = False
        length = len(self.possibilities)
        for house, house_indexes in self._houses_to_indexes.items():
            # get list of all values that can be found in house cells outside this feature
            found_outside = {value for cell in house.unknown_cells if cell not in self._cells_as_set
                             for value in cell.possible_values}
            # These are the values that must be inside the feature
            required_inside = house.unknown_values - found_outside
            for value in required_inside:
                if value in self._value_only_in_feature[house]:
                    # We've already ensured that this value occurs only inside this feature.  Ignore
                    continue
                self._value_only_in_feature[house].add(value)
                self.possibilities = [values for values in self.possibilities
                                      if any(values[i] == value for i in house_indexes)]
                if len(self.possibilities) != length:
                    print(f'For {house}, value {value} only occurs inside {self}')
                    print(f"Possibilities for {self} reduced from {length} to {len(self.possibilities)}")
                    updated = True
                    length = len(self.possibilities)
        if updated:
            return self.__update_for_possibilities()

    def __remove_bad_neighbors(self, possibilities: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]]:
        for (index1, cell1), (index2, cell2) in combinations(enumerate(self.cells), 2):
            if cell1.index == cell2.index:
                # For some reason, we have the same cell repeated twice
                possibilities = [p for p in possibilities if p[index1] == p[index2]]
            elif cell1.is_neighbor(cell2):
                possibilities = [p for p in possibilities if p[index1] != p[index2]]
        return possibilities

    def check_join_two(self):
        def clinginess(f1: PossibilitiesFeature, f2: PossibilitiesFeature) -> int:
            count = 0
            for cell in f2.cells:
                if cell in f1._cells_as_set:
                    count += 9
                else:
                    count += len(cell.neighbors & f1._cells_as_set)
            return count

        features: set[PossibilitiesFeature] = self.grid[self.__class__, "feature"]
        sorted_features = sorted((x for x in features if len(x.possibilities) > 1),
                                 key=lambda f: len(f.possibilities))
        try:
            _count, m1, m2 = max(((clinginess(f1, f2), f1, f2) for f1, f2 in combinations(sorted_features[:10], 2)
                                 if len(f1.possibilities) * len(f2.possibilities) <= 10_000),
                                 key = lambda x:x[0])
            m1.merge_into_me(m2)
            m1.__handle_all_possibilities_use_value()
            return True
        except ValueError:
            return False

    def merge_into_me(self, other: PossibilitiesFeature):
        length1, length2 = len(self.possibilities), len(other.possibilities)
        self.__initialize_after_merge(other)

        my_possibilities = [item1 + item2 for item1, item2 in product(self.possibilities, other.possibilities)]
        my_possibilities = self.__remove_bad_neighbors(my_possibilities)
        length3 = len(my_possibilities)
        self.possibilities = my_possibilities

        features: set[PossibilitiesFeature] = self.grid[self.__class__, "feature"]
        features.remove(other)
        temp = length3 * 100 / (length1 * length2)
        print(f'Merge {self.name} ({length1}) x {other.name} ({length2}) = ({length3}) {temp:.2f}%')


class GroupedPossibilitiesFeature(Feature, abc.ABC):
    """We are given a set of possible values for a set of cells"""
    squares: Sequence[Square]
    cells: Sequence[Cell]
    possibilities: list[tuple[SmallIntSet, ...]]
    handle_neighbors: bool
    compressed: bool

    def __init__(self, squares: Union[Sequence[Square], str], *,
                 name: Optional[str] = None, neighbors: bool = False, compressed: bool = False) -> None:
        super().__init__(name=name)
        if isinstance(squares, str):
            squares = self.parse_squares(squares)
        self.squares = squares
        self.handle_neighbors = neighbors
        self.compressed = compressed

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        self.cells = [grid.matrix[square] for square in self.squares]

    @abc.abstractmethod
    def get_possibilities(self) -> list[tuple[Union[SmallIntSet, Iterable[int], int], ...]]: ...

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

    @Feature.check_only_if_changed
    def check(self) -> bool:
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

