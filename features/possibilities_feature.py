from __future__ import annotations

import abc
import math
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from functools import cache
from itertools import chain, combinations, permutations, product
from typing import Callable, Iterable, Mapping, Optional, Sequence

from cell import Cell, CellValue, House, SmallIntSet
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

    def __init__(self, squares: SquaresParseable, *,
                 possibility_function: Optional[Callable[[], Iterable[Possibility]]] = None,
                 name: Optional[str] = None, prefix: Optional[str] = None,
                 neighbors: bool = False, duplicates: bool = False) -> None:
        super().__init__(name=name, prefix=prefix)
        self.squares = self.parse_squares(squares)
        self.handle_neighbors = neighbors
        self.handle_duplicates = duplicates
        self.possibility_function = possibility_function or self.get_possibilities

    def get_possibilities(self) -> Iterable[Possibility]:
        raise NotImplementedError()

    def start(self) -> None:
        cells = [self @ square for square in self.squares]
        possibilities = list(self.possibility_function())
        if self.handle_duplicates:
            possibilities = list(set(possibilities))
        if self.handle_neighbors:
            possibilities = self.__remove_bad_neighbors(cells, possibilities)
        print(f'{self} has {len(possibilities)} possibilities')
        possibility_info = PossibilityInfo(grid=self.grid, cells=cells, possibilities=possibilities, name=self.name)
        self.grid.possibilities_handler.add_info(possibility_info)

    @classmethod
    def __remove_bad_neighbors(cls, cells: Sequence[Cell], possibilities: list[Possibility]) -> list[Possibility]:
        for (index1, cell1), (index2, cell2) in combinations(enumerate(cells), 2):
            if cell1.square == cell2.square:
                # For some reason, we have the same cell repeated twice
                possibilities = [p for p in possibilities if p[index1] == p[index2]]
            elif cell1.is_neighbor(cell2):
                possibilities = [p for p in possibilities if p[index1] != p[index2]]
        return possibilities


class AdjacentRelationshipFeature(Feature, abc.ABC):
    squares: Sequence[Square]
    cyclic: bool

    def __init__(self, squares: SquaresParseable, cyclic: bool = False, prefix: Optional[str] = None) -> None:
        self.squares = self.parse_squares(squares)
        self.cyclic = cyclic
        super().__init__(prefix=prefix)

    def start(self) -> None:
        squares, cyclic, match = self.squares, self.cyclic, self.match
        if len(squares) == 2:
            pairs = [(i, j) for i, j in product(range(1, 10), repeat=2)
                     if match(i, j) and (not cyclic or match(j, i))]
            features = [PossibilitiesFeature(self.squares, prefix=self.name, neighbors=True,
                                             possibility_function=lambda: pairs)]
        else:
            x_squares = squares if not cyclic else list(chain(squares, squares[0:2]))
            triples = [(i, j, k) for i, j, k in product(range(1, 10), repeat=3) if match(i, j) and match(j, k)]
            features = [PossibilitiesFeature(x_squares[i:i + 3], prefix=self.name, neighbors=True,
                                             possibility_function=lambda: triples)
                        for i in range(0, len(x_squares) - 2)]
        for feature in features:
            feature.initialize(self.grid)
            feature.start()

    @abc.abstractmethod
    def match(self, i: int, j: int) -> bool:
        raise NotImplementedError()


class FullGridAdjacencyFeature(Feature, abc.ABC):
    def __init__(self, *, prefix: str) -> None:
        super().__init__(name=prefix)

    def start(self) -> None:
        match = self.match
        pairs = [(a, b) for a, b in permutations(range(1, 10), 2) if match(a, b)]
        quads = [(a, b, c, d) for (a, b) in pairs for (c, d) in pairs
                 if a != c and b != d and match(a, c) and match(b, d)]
        features = [PossibilitiesFeature(((i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)),
                                         name=f'{self.name}@r{i}c{j}',
                                         neighbors=True, possibility_function=lambda: quads)
                    for i, j in product(range(1, 9), repeat=2)]
        for feature in features:
            feature.initialize(self.grid)
            feature.start()

    @abc.abstractmethod
    def match(self, i: int, j: int) -> bool:
        raise NotImplementedError()


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
    __previous_possibility_value_pruning: dict[Cell, SmallIntSet] = field(init=False)
    __known_identical_cells: UnionFind[Cell] = field(default_factory=UnionFind)

    __cells_at_last_call_to_check: list[int] = field(default_factory=list)

    def __post_init__(self, verbose: bool) -> None:
        self.cells_as_set = set(self.cells)
        self.__value_only_in_feature = defaultdict(SmallIntSet)
        self.__house_to_indexes = defaultdict(list)
        for index, cell in enumerate(self.cells):
            for house in cell.houses:
                self.__house_to_indexes[house].append(index)
        # self.__handle_shrunken_possibilities(show=verbose)
        self.__previous_possibility_value_pruning = {cell: SmallIntSet.get_full_cell() for cell in self.cells}

    def check(self) -> bool:
        if not Feature.cells_changed_since_last_invocation(self.__cells_at_last_call_to_check, self.cells):
            return False

        old_length = len(self.possibilities)
        if old_length == 1:
            self.grid.possibilities_handler.remove_info(self)
            return False

        possibilities = self.possibilities
        for index, cell in enumerate(self.cells):
            cpv = cell.possible_values
            previous_prune = self.__previous_possibility_value_pruning[cell]
            if cpv != previous_prune:
                possibilities = [p for p in possibilities if p[index] in cpv]
                self.__previous_possibility_value_pruning[cell] = cpv.copy()

        change = False
        if len(possibilities) != old_length:
            print(f"Possibilities for {self} reduced from {old_length} to {len(possibilities)}")
            self.possibilities = possibilities
            change = self.__handle_shrunken_possibilities()

        return change

    def check_special(self) -> bool:
        return self.__handle_all_occurrences_of_value_in_house_are_within_possibility() or \
               self.__check_values_identical_in_all_possibilities() or \
               self.__check_identical_cells()

    # Note that this cache is cleared whenever the number of possibilities changes
    @cache
    def get_weak_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        cell, value = cell_value
        if cell not in self.cells_as_set:
            return ()
        # A weak pair says both conditions can't simultaneously be true.  Assume the cell has the indicated value
        # and see which values in other cells are no longer possibilities that had been before.
        index = self.cells.index(cell)
        iterator = (possibility for possibility in self.possibilities if possibility[index] == value)
        hopefuls = {index2: cell2.possible_values.copy() for index2, cell2 in enumerate(self.cells)
                    if cell2 != cell and not cell2.is_known}
        deletions: set[int] = set()
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

    # Note that this cache is cleared whenever the number of possibilities changes
    @cache
    def get_strong_pairs(self, cell_value: CellValue) -> Iterable[CellValue]:
        cell, value = cell_value
        if cell not in self.cells_as_set:
            return ()
        # A strong pair says both conditions can't simultaneously be false.  Assume the cell doesn't have the
        # indicated value and see which values in other cells are forced.
        cell, value = cell_value
        index = self.cells.index(cell)
        iterator = (possibility for possibility in self.possibilities if possibility[index] != value)
        first = next(iterator)
        hopefuls = {index2: first[index2] for index2, cell2 in enumerate(self.cells)
                    if cell2 != cell and not cell2.is_known}
        deletions: set[int] = set()
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

    def handle_one_of_values_in_cells(self, cells: set[Cell], values: SmallIntSet) -> bool:
        """
        We have determined that at least one of the cells (all of which are part of our group) must
        contain at least one of the indicated values.  See if we can use that information to decrease
        the number of possibilities.
        """
        assert cells <= self.cells_as_set
        indexes = [self.cells.index(cell) for cell in cells]
        length = len(self.possibilities)
        self.possibilities = [possibility for possibility in self.possibilities
                              if any(possibility[i] in values for i in indexes)]
        if length != len(self.possibilities):
            print(f'In {self},  {values} must occur inside cells {sorted(cells)}')
            print(f"Possibilities for {self} reduced from {length} to {len(self.possibilities)}")
            return self.__handle_shrunken_possibilities()
        return False

    def __handle_shrunken_possibilities(self, show: bool = True) -> bool:
        self.get_weak_pairs.cache_clear()
        self.get_strong_pairs.cache_clear()
        # See if we can change any of our own cells
        changed = self.__shrunken_possibilities_change_cells_inside_me(show)
        # See if we can change any cells outside of us.
        changed |= self.__shrunken_possibilities_change_cells_outside_me(show)
        return changed

    def __shrunken_possibilities_change_cells_inside_me(self, show: bool) -> bool:
        """
        Check for cells inside this PossibilityInfo whose possible values might have changed we the number
        of our possibilities has shrunk
        """
        changed = False
        for index, cell in enumerate(self.cells):
            if cell.is_known:
                if cell not in self.__verified_cells:
                    assert all(values[index] == cell.known_value for values in self.possibilities)
                    self.__verified_cells.add(cell)
                continue
            legal_values = SmallIntSet({values[index] for values in self.possibilities})
            if not cell.possible_values <= legal_values:
                changed = True
                if len(legal_values) == 1:
                    cell.set_value_to(legal_values.unique(), show=True)
                    self.__verified_cells.add(cell)
                else:
                    Cell.keep_values_for_cell([cell], legal_values, show=show)
        return changed

    def __shrunken_possibilities_change_cells_outside_me(self, show: bool) -> bool:
        """
        Check for cells outside of this PossibilityInfo whose possible values might have changed we the number
        of our possibilities has shrunk
        """

        # If for some house and value, that value occurs inside this PossibilityInfo for every possibility (though not
        # necessarily the same cell each time), then that value must be part of this PossibilityInfo, and cannot occur
        # anywhere in the house.
        changed = False
        for house, indexes in self.__house_to_indexes.items():
            # Ignore values that we already determined to be locked into this PossibilityInfo.
            locked_values = house.unknown_values - self.__value_only_in_feature[house]
            for possibility in self.possibilities:
                locked_values &= SmallIntSet({possibility[i] for i in indexes})
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
                    Cell.remove_values_from_cells(affected_cells, locked_values, show=show)
                    changed = True
        return changed

    def __handle_all_occurrences_of_value_in_house_are_within_possibility(self) -> bool:
        """
        If for a specific house and value, that value can only occur inside this PossibilityInfo, then that value
        must be inside this PossibilityInfo, and we can prune any possibilities that don't include that value inside
        the house.
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
                    print(f'Inside {house}, value {value} only occurs as part of {self}.')
                    print(f"Possibilities for {self} reduced from {length} to {len(self.possibilities)}")
                    updated = True
                    length = len(self.possibilities)
        if updated:
            return self.__handle_shrunken_possibilities()
        return False

    def __check_values_identical_in_all_possibilities(self) -> bool:
        """If two cells are identical in all possibilities, then we can guarantee they have the same value"""
        same_value_handler = self.grid.same_value_handler
        hopefuls = {(index1, index2)
                    for (index1, cell1), (index2, cell2) in combinations(enumerate(self.cells), 2)
                    if not len(cell1.possible_values) == 1
                    if not len(cell2.possible_values) == 1
                    if not same_value_handler.are_cells_same_value(cell1, cell2)}
        for possibility in self.possibilities:
            deletions = {(ix1, ix2) for (ix1, ix2) in hopefuls if possibility[ix1] != possibility[ix2]}
            hopefuls -= deletions
            if not hopefuls:
                return False
        for index1, index2 in hopefuls:
            cell1, cell2 = self.cells[index1], self.cells[index2]
            print(f"In every possibility of {self}, {cell1} = {cell2}, so they must have identical values")
            same_value_handler.make_cells_same_value(cell1, cell2, name=f'{cell1}={cell2} {self}')
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
            return self.__handle_shrunken_possibilities()
        return False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class PossibilitiesHandler(Feature):
    infos: dict[PossibilityInfo, bool]
    merge_count: int

    def __init__(self) -> None:
        super().__init__(name="Possibilities Handler")
        self.infos = {}
        self.merge_count = 0

    def add_info(self, info: PossibilityInfo) -> None:
        self.infos[info] = True

    def remove_info(self, info: PossibilityInfo) -> None:
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

    def handle_one_of_values_in_cells(self, cells: set[Cell], values: SmallIntSet) -> bool:
        """
        We have determined that at least one of the cells contains at least one of the
        values.  See if we can do anything about that.
        """
        return any(info.handle_one_of_values_in_cells(cells, values)
                   for info in self.infos if cells <= info.cells_as_set)

    def __try_to_merge(self) -> bool:
        p_log = {info: math.log(len(info.possibilities)) for info in self.infos}

        def closeness(f1: PossibilityInfo, f2: PossibilityInfo) -> float:
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
        self.merge_count += 1
        length1, length2 = len(info1.possibilities), len(info2.possibilities)

        index1 = {cell: index for index, cell in enumerate(info1.cells)}
        index2 = {cell: index for index, cell in enumerate(info2.cells, start=len(index1))}

        print(f'Merge {info1} ({length1}) x {info2} ({length2}) = {length1 * length2} --> ', end='')

        possibilities = [p1 + p2 for p1, p2 in product(info1.possibilities, info2.possibilities)]
        duplicates = info1.cells_as_set & info2.cells_as_set

        for cell in duplicates:
            if len(cell.possible_values) > 1:
                i1, i2 = index1[cell], index2[cell]
                possibilities = [p for p in possibilities if p[i1] == p[i2]]

        for cell2, i2 in index2.items():
            # We don't need to bother checking when either cell1 or cell2 is in duplicates.  We already know that
            # it can't have the same value as the neighbors.
            if cell2 not in duplicates:
                for cell1 in info1.cells_as_set & cell2.neighbors:
                    if cell1 not in duplicates:
                        i1 = index1[cell1]
                        possibilities = [p for p in possibilities if p[i1] != p[i2]]

        cells = tuple(chain(info1.cells, info2.cells))

        deleted_indices = {index2[cell] for cell in duplicates}
        deleted_indices.update(index for index, cell in enumerate(cells) if len(cell.possible_values) == 1)
        if deleted_indices:
            def trim(sequence: Sequence[Node]) -> tuple[Node, ...]:
                return tuple(item for ix, item in enumerate(sequence) if ix not in deleted_indices)
            cells = trim(cells)
            possibilities = [trim(p) for p in possibilities]

        length3 = len(possibilities)
        fraction = length3 * 100.0 / (length1 * length2)
        print(f'Merge #{self.merge_count} ({length3} {fraction:.2f}%)')

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
    house_type: House.Type
    house_index: int

    def __init__(self, house_type: House.Type, index: int, *,
                 name: Optional[str] = None, prefix: Optional[str] = None):
        if not name and prefix:
            name = f'{prefix} {house_type.name} #{index}'
        super().__init__(self.get_house_squares(house_type, index), name=name)
        self.house_type = house_type
        self.house_index = index

    def get_possibilities(self) -> Iterable[Possibility]:
        return filter(self.match, self.generator())

    def match(self, permutation: Possibility) -> bool:
        return True

    def generator(self) -> Iterable[Possibility]:
        return permutations(range(1, 10))
