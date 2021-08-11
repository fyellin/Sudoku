from __future__ import annotations

import abc
import math
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from functools import cache
from itertools import chain, combinations, permutations, product
from typing import Callable, Iterable, Mapping, Optional, Sequence

from cell import Cell, CellValue, House, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from grid import Grid
from tools.itertool_recipes import all_equal, pairwise
from tools.union_find import Node, UnionFind

Possibility = tuple[int, ...]


class PossibilitiesFeature(Feature):
    """We are given a set of possible values for a set of cells"""
    squares: Sequence[Square]
    handle_neighbors: bool
    handle_duplicates: bool
    possibility_function: Callable[[], Iterable[Possibility]]

    @classmethod
    def create(cls, squares: SquaresParseable, *,
               possibility_function: Callable[[], Iterable[Possibility]],
               name: Optional[str] = None, prefix: Optional[str] = None,
               neighbors: bool = False, duplicates: bool = False) -> Sequence[PossibilitiesFeature]:
        return [
            PossibilitiesFeature(squares, possibility_function=possibility_function,
                                 name=name, prefix=prefix, neighbors=neighbors, duplicates=duplicates)
        ]

    def __init__(self, squares: SquaresParseable, *,
                 possibility_function: Optional[Callable[[], Iterable[Possibility]]] = None,
                 name: Optional[str] = None, prefix: Optional[str] = None,
                 neighbors: bool = False, duplicates: bool = False) -> None:
        super().__init__(name=name, prefix=prefix)
        self.squares = self.parse_squares(squares) if isinstance(squares, str) else squares
        self.handle_neighbors = neighbors
        self.handle_duplicates = duplicates
        self.possibility_function = possibility_function or self.get_possibilities

    def get_possibilities(self) -> Iterable[Possibility]:
        raise NotImplementedError()

    def start(self):
        cells = [self @ square for square in self.squares]
        possibilities = list(self.possibility_function())
        if self.handle_duplicates:
            possibilities = list(set(possibilities))
        if self.handle_neighbors:
            possibilities = self.__remove_bad_neighbors(cells, possibilities)
        print(f'{self} has {len(possibilities)} possibilities')
        possibility_info = PossibilityInfo(grid=self.grid, cells=cells, possibilities=possibilities, name=self.name)
        self.grid.possibilities_handler.add_info(possibility_info)

    def __remove_bad_neighbors(self, cells, possibilities: Sequence[Possibility]) -> list[Possibility]:
        for (index1, cell1), (index2, cell2) in combinations(enumerate(cells), 2):
            if cell1.index == cell2.index:
                # For some reason, we have the same cell repeated twice
                possibilities = [p for p in possibilities if p[index1] == p[index2]]
            elif cell1.is_neighbor(cell2):
                possibilities = [p for p in possibilities if p[index1] != p[index2]]
        return possibilities


@dataclass(eq=False)
class PossibilityInfo:
    grid: Grid
    cells: Sequence[Cell]
    possibilities: Sequence[Possibility]
    name: str
    verbose: InitVar[bool] = False

    cells_as_set: set[Cell] = field(init=False)
    # We have already handled the fact that for this house, the indicated values occur within this feature
    __value_only_in_feature: dict[House, SmallIntSet] = field(init=False)
    # A mapping from houses to  cell indexes that are in that house
    __house_to_indexes: Mapping[House, list[int]] = field(init=False)
    # These cells have a known value, and we have verified that all possibilities use that value
    __verified_cells: set[Cell] = field(default_factory=set)
    # These cells are known to have identical values in all possibilities
    __known_identical_cells: UnionFind[Cell] = field(default_factory=UnionFind)

    __cells_at_last_call_to_check: list[int] = field(default_factory=list)
    __weak_pair_cache_check: int = -1
    __strong_pair_cache_check: int = -1

    def __post_init__(self, verbose):
        self.cells_as_set = set(self.cells)
        self.__value_only_in_feature = defaultdict(SmallIntSet)
        self.__house_to_indexes = defaultdict(list)
        for index, cell in enumerate(self.cells):
            for house in cell.houses:
                self.__house_to_indexes[house].append(index)
        self.__update_cells_for_possibilities(show=verbose)

    def check(self):
        if not Feature.cells_changed_since_last_invocation(self.__cells_at_last_call_to_check, self.cells):
            return False
        old_length = len(self.possibilities)
        if old_length == 1:
            self.grid.possibilities_handler.remove_info(self)
            return False
        # Only keep those possibilities that are still viable
        possibilities = [values for values in self.possibilities
                         if all(value in square.possible_values for value, square in zip(values, self.cells))]
        if len(possibilities) != old_length:
            print(f"Possibilities for {self} reduced from {old_length} to {len(possibilities)}")
            self.possibilities = possibilities
            change = self.__update_cells_for_possibilities()
            change |= self.__handle_all_possibilities_use_value_in_house()
            return change

        return False

    def check_special(self) -> bool:
        return self.__handle_value_in_house_only_occurs_in_possibility() or \
               self.__check_values_identical_in_all_possibilities() or \
               self.__check_identical_cells()

    def get_weak_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        if cell_value.cell not in self.cells_as_set:
            return ()
        length = len(self.possibilities)
        if self.__weak_pair_cache_check != length:
            self.__get_weak_pairs.cache_clear()
            self.__weak_pair_cache_check = length
        return self.__get_weak_pairs(cell_value)

    def get_strong_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        if cell_value.cell not in self.cells_as_set:
            return ()
        length = len(self.possibilities)
        if self.__strong_pair_cache_check != length:
            self.__get_strong_pairs.cache_clear()
            self.__strong_pair_cache_check = length
        return self.__get_strong_pairs(cell_value)

    @cache
    def __get_weak_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        # A weak pair says both conditions can't simultaneously be true.  Assume the cell has the indicated value
        # and see which values in other cells are no longer possibilities that had been before.
        cell, value = cell_value
        index = self.cells.index(cell)
        iterator = (possibility for possibility in self.possibilities if possibility[index] == value)
        hopefuls = {index2: cell2.possible_values.copy() for index2, cell2 in enumerate(self.cells)
                    if cell2 != cell and not cell2.is_known}
        deletions = set()
        for possibility in iterator:
            deletions.clear()
            for index2, possible_values2 in hopefuls.items():
                possible_values2.discard(possibility[index2])
                if not possible_values2:
                    deletions.add(index2)
            for index2 in deletions:
                hopefuls.pop(index2)
            if not hopefuls:
                return ()

        return [CellValue(self.cells[index2], value2)
                for index2, values2 in hopefuls.items()
                for value2 in values2]

    @cache
    def __get_strong_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        # A strong pair says both conditions can't simultaneously be false.  Assume the cell doesn't have the
        # indicated value and see which values in other cells are forced.
        cell, value = cell_value
        index = self.cells.index(cell)
        iterator = (possibility for possibility in self.possibilities if possibility[index] != value)
        first = next(iterator)
        hopefuls = {index2: first[index2] for index2, cell2 in enumerate(self.cells)
                    if cell2 != cell and not cell2.is_known}
        deletions = set()
        for possibility in iterator:
            deletions.clear()
            for index2, value2 in hopefuls.items():
                if value2 != possibility[index2]:
                    deletions.add(index2)
            for index2 in deletions:
                hopefuls.pop(index2)
            if not hopefuls:
                return ()

        return [CellValue(self.cells[index2], value2) for index2, value2 in hopefuls.items()]

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
        If for a specific house and unknown value of that house, that value always occurs as part of this
        info.  That value must be part of this info (though we don't know which cell) and all occurrences
        of that value inside the house but outside this feature can be removed.
        be removed.
        """
        change = False
        for house, indexes in self.__house_to_indexes.items():
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

    def __check_values_identical_in_all_possibilities(self) -> bool:
        """If two cells are identical in all possibilities, then we can guarantee they have the same value"""
        same_value_handler = self.grid.same_value_handler
        hopefuls = {(index1, index2)
                    for (index1, cell1), (index2, cell2) in combinations(enumerate(self.cells), 2)
                    if not same_value_handler.are_cells_same_value(cell1, cell2)}
        for possibility in self.possibilities:
            deletions = {(ix1, ix2) for (ix1, ix2) in hopefuls if possibility[ix1] != possibility[ix2]}
            hopefuls -= deletions
            if not hopefuls:
                return False
        for index1, index2 in hopefuls:
            cell1, cell2 = self.cells[index1], self.cells[index2]
            same_value_handler.make_cells_same_value(cell1, cell2, f'{cell1}={cell2} {self}')
            self.__known_identical_cells.union(cell1, cell2)
        return True

    def __check_identical_cells(self) -> bool:
        """If two cells are guaranteed to have the same value, we can prune possibilities down to just those
        in which the cells are equal.
        """
        same_value_handler = self.grid.same_value_handler
        groups = same_value_handler.group_same_value_cells(self.cells)
        if not groups:
            return False
        updated = False
        length = len(self.possibilities)
        for group in groups:
            if all_equal(self.__known_identical_cells.find(cell) for cell in group):
                continue
            [self.__known_identical_cells.union(cell1, cell2) for cell1, cell2 in pairwise(group)]
            indexes = [self.cells.index(cell) for cell in group]
            self.possibilities = [p for p in self.possibilities if all_equal(p[index] for index in indexes)]
            if len(self.possibilities) < length:
                print(f"Possibilities for {self} reduced from {length} to {len(self.possibilities)} "
                      f"because  {'='.join(str(cell) for cell in group)}")
                length = len(self.possibilities)
                updated = True
        if updated:
            return self.__update_cells_for_possibilities()
        return False

    def __handle_value_in_house_only_occurs_in_possibility(self) -> bool:
        """
        If for a specific house, an unknown value only occurs inside this info, then that value must be
        part of of this info.  We can prune the possibilities that don't include that value inside the house.
        """
        updated = False
        length = len(self.possibilities)
        for house, house_indexes in self.__house_to_indexes.items():
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
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class PossibilitiesHandler(Feature):
    infos: dict[PossibilityInfo, bool]
    merge_count: int

    def __init__(self):
        super().__init__(name="Possibilities Handler")
        self.infos = {}
        self.merge_count = 0

    def add_info(self, info: PossibilityInfo):
        self.infos[info] = True

    def remove_info(self, info: PossibilityInfo):
        del self.infos[info]

    def check(self) -> bool:
        # We make a copy of self.infos, since an item might delete itself
        return any(info.check() for info in list(self.infos))

    def check_special(self) -> bool:
        # We make a copy of self.infos, since an item might delete itself
        if any(info.check_special() for info in list(self.infos)):
            return True
        return self.__try_to_merge()

    def get_weak_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        for info in self.infos:
            yield from info.get_weak_pairs(cell_value)

    def get_strong_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        for info in self.infos:
            yield from info.get_strong_pairs(cell_value)

    def __try_to_merge(self) -> bool:
        p_log = {info: math.log(len(info.possibilities)) for info in self.infos}

        def closeness(f1: PossibilityInfo, f2: PossibilityInfo) -> int:
            count = 0
            for cell in f2.cells:
                if cell in f1.cells_as_set:
                    count += 9
                else:
                    count += len(cell.neighbors & f1.cells_as_set)
            return count - p_log[f1] - p_log[f2]

        sorted_features = sorted((x for x in self.infos if len(x.possibilities) > 1),
                                 key=lambda f: len(f.possibilities))
        try:
            _count, m1, m2 = max(((closeness(f1, f2), f1, f2) for f1, f2 in combinations(sorted_features[:10], 2)
                                 if len(f1.possibilities) * len(f2.possibilities) <= 10_000_000),
                                 key=lambda x: x[0])
        except ValueError:
            return False

        merged_info = self.__perform_merge(m1, m2)
        self.remove_info(m1)
        self.remove_info(m2)
        self.add_info(merged_info)
        return True

    def __perform_merge(self, info1: PossibilityInfo, info2: PossibilityInfo) -> PossibilityInfo:
        length1, length2 = len(info1.possibilities), len(info2.possibilities)

        index = {cell: index for index, cell in enumerate(info1.cells)}
        possibilities = [p1 + p2 for p1, p2 in product(info1.possibilities, info2.possibilities)]
        deletions = set()
        # A simplified version of handling neighbors, since we only need to cross check between the features.
        for index2, cell2 in enumerate(info2.cells, start=len(info1.cells)):
            if (index1 := index.get(cell2)) is not None:
                possibilities = [p for p in possibilities if p[index1] == p[index2]]
                deletions.add(index2)
            else:
                for cell1 in info1.cells_as_set & cell2.neighbors:
                    index1 = index[cell1]
                    possibilities = [p for p in possibilities if p[index1] != p[index2]]

        length3 = len(possibilities)
        fraction = length3 * 100.0 / (length1 * length2)
        print(f'Merge {info1} ({length1}) x {info2} ({length2}) = {length1 * length2} --> {length3} {fraction:.2f}%')

        cells = tuple(chain(info1.cells, info2.cells))
        if deletions:
            def trim(sequence: Sequence[Node]) -> tuple[Node, ...]:
                return tuple(item for ix, item in enumerate(sequence) if ix not in deletions)
            cells = trim(cells)
            possibilities = [trim(p) for p in possibilities]

        self.merge_count += 1
        result = PossibilityInfo(grid=self.grid, cells=cells, possibilities=possibilities,
                                 name=f'Merge #{self.merge_count}', verbose=True)
        result.check()
        result.check_special()
        return result


class HousePossibilitiesFeature(PossibilitiesFeature, abc.ABC):
    """
    A simplified method of generating a Possibilities Feature when we know that the item
    takes up an entire house.  The user overrides generator() [which defaults to generating all
    permutations of 1..9], and match() which determines if a permutation should be included.  9! is
    smaller than it used to be.
    """
    htype: House.Type
    index: int

    def __init__(self, htype: House.Type, index: int, *,
                 name: Optional[str] = None, prefix: Optional[str] = None):
        if not name and prefix:
            name = f'{prefix} {htype.name} #{index}'
        super().__init__(self.get_house_squares(htype, index), name=name)
        self.htype = htype
        self.index = index

    def get_possibilities(self) -> Iterable[Possibility]:
        return filter(self.match, self.generator())

    @abc.abstractmethod
    def match(self, permutation: Possibility) -> bool:
        ...

    def generator(self):
        return permutations(range(1, 10))


class GroupedPossibilitiesFeature(Feature, abc.ABC):
    """We are given a set of possible values for a set of cells"""
    squares: Sequence[Square]
    cells: Sequence[Cell]
    possibilities: list[tuple[SmallIntSet, ...]]
    handle_neighbors: bool
    compressed: bool
    __cells_at_last_call_to_check: list[int]

    def __init__(self, squares: SquaresParseable, *,
                 name: Optional[str] = None, neighbors: bool = False, compressed: bool = False) -> None:
        super().__init__(name=name)
        if isinstance(squares, str):
            squares = self.parse_squares(squares)
        self.squares = squares
        self.handle_neighbors = neighbors
        self.compressed = compressed
        self.__cells_at_last_call_to_check = []

    @abc.abstractmethod
    def get_possibilities(self) -> list[tuple[SmallIntSet | Iterable[int] | int], ...]:
        ...

    def start(self) -> None:
        self.cells = [self@square for square in self.squares]

        def fixit_one(x: SmallIntSet | Iterable[int] | int) -> SmallIntSet:
            if isinstance(x, int):
                return SmallIntSet([x])
            elif isinstance(x, SmallIntSet):
                return x
            else:
                return SmallIntSet(x)

        def fixit(items: tuple[SmallIntSet | Iterable[int] | int, ...]) -> tuple[SmallIntSet, ...]:
            return tuple(fixit_one(item) for item in items)

        possibilities = list(fixit(x) for x in self.get_possibilities())
        if self.handle_neighbors:
            possibilities = self.__remove_bad_neighbors(possibilities)
        print(f'{self} has {len(possibilities)} possibilities')
        self.possibilities = possibilities
        self.__update_for_possibilities(False)

    def check(self) -> bool:
        if not self.cells_changed_since_last_invocation(self.__cells_at_last_call_to_check, self.cells):
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

            def get_possibilities(self) -> Iterable[Possibility]:
                for element in parent.get_possibilities():
                    yield from product(*element)

        return ChildPossibilityFeature()
