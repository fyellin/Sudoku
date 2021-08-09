from __future__ import annotations

import functools
import itertools
import operator
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

from cell import Cell, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from grid import Grid
from tools.itertool_recipes import pairwise, unique_everseen
from tools.union_find import UnionFind


class SameValueFeature(Feature):
    """
    The sole goal is of SameValueFeature is to hold onto its information until start() is called.  At this point, it
    creates an Equivalence to hold the information and tells the SameValueHandler about this equivalence.
    Then its done.
    """
    squares: Sequence[Square]
    has_real_name: bool

    def __init__(self, squares: SquaresParseable, name: Optional[str] = None, *, prefix: Optional[str] = None) -> None:
        self.squares = self.parse_squares(squares)
        if not name and not prefix:
            name = '='.join(f'r{r}c{c}' for r, c in self.squares)
        super().__init__(name=name, prefix=prefix)
        self.has_real_name = name is not None or prefix is not None

    def start(self):
        cells = [self @ square for square in self.squares]
        equivalence = _Equivalence(grid=self.grid, cells=cells, name=self.name)
        self.grid.same_value_handler.add_equivalence(equivalence)


@dataclass(eq=False)
class _Equivalence:
    """
    An equivalence keeps track of all information about a group of cells that have the same value
    """
    grid: Grid
    cells: list[Cell]
    name: str
    color: Optional[str] = None
    __check_cache: list[int] = field(default_factory=list)

    def __post_init__(self):
        neighbors = frozenset.union(*(cell.neighbors for cell in self.cells))
        self.set_all_neighbors(neighbors)

    def set_all_neighbors(self, neighbors: frozenset[Cell]):
        assert neighbors.isdisjoint(self.cells)
        # Add these neighbors to all my cells
        for cell in self.cells:
            cell.neighbors = neighbors
        # Add my cells as neighbors to all the neighbors.
        cells_as_set = set(self.cells)
        for cell in neighbors:
            cell.neighbors |= cells_as_set

    def check(self) -> bool:
        if not Feature.cells_changed_since_last_invocation(self.__check_cache, self.cells):
            return False
        return self.__check()

    def __check(self):
        # Our possible values are values that can fit into any of cour cells
        result = functools.reduce(operator.__and__, (cell.possible_values for cell in self.cells))
        # value is non-None only if we have one remaining value.
        value = result.unique() if len(result) == 1 else None
        if value is not None:
            cells_to_update = [cell for cell in self.cells if not cell.is_known]
            # This item need never be looked at again.
            self.grid.same_value_handler.remove_equivalence(self)
        else:
            cells_to_update = [cell for cell in self.cells if cell.possible_values != result]
        if cells_to_update:
            if value:
                print(f"All cells of {self} have the value {value}")
                [cell.set_value_to(value) for cell in cells_to_update]
                print(f'  {", ".join(str(cell) for cell in sorted(cells_to_update))} := {value}')
            else:
                print(f"Intersection of possible values for {self}")
                Cell.keep_values_for_cell(cells_to_update, result)
            return True
        return False

    def check_special(self) -> bool:
        # return self.__check_all_values_legal_in_all_houses() | self.__check_try_to_expand_equivalence()
        return self.__check_try_to_expand_equivalence()

    def __check_all_values_legal_in_all_houses_maybe_obsolete_but_not_deleting_yet(self):
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

    def __check_try_to_expand_equivalence(self) -> bool:
        equivalences = self.grid.same_value_handler.equivalences
        changed = False
        while self in equivalences:  # We can remove ourselves once we've got a value
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
                if len(viable_candidates) == 1:
                    # There is only one possible candidate for me in this house
                    candidate = viable_candidates.pop()
                    print(f'In {house}, {self} must also include {candidate}')
                    equivalence = _Equivalence(grid=self.grid, cells=[self.cells[0], candidate], name=f'+{candidate}')
                    self.grid.same_value_handler.add_equivalence(equivalence)
                    assert equivalence not in equivalences
                    self.__check()
                    changed = True
                    break
                else:
                    all_possible_values = SmallIntSet.union(*(cell.possible_values for cell in viable_candidates))
                    # Can this happen?  Intersection removal may already take care of this!  All the occurrences
                    # of the value in the house are from cells that are all our neighbors. So the value should have
                    # been removed from us.
                    if not my_values <= all_possible_values:
                        excluded = my_values - all_possible_values
                        my_values &= all_possible_values
                        print(f'**********************')
                        print(f'In {house}, {self} must be one of {viable_candidates}.  Cannot include {excluded}.')
                        Cell.keep_values_for_cell(self.cells, my_values)
                        changed = True
                        break
            else:
                return changed
        return changed

    def __str__(self) -> str:
        return self.name


class SameValueHandler(Feature):
    VERIFY = True

    COLORS = ('#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
              '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
              '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000')

    equivalences: dict[_Equivalence, bool]  # Using the fact that dictionaries remember the order items are added.
    __union_find: UnionFind
    __token_to_equivalence: dict[Cell, _Equivalence]
    __colors: deque[str]

    def __init__(self):
        super().__init__(name="Same Value Handler")
        # equivalences is really a set, but we use a dictionary so that we iterate by order of entry
        self.equivalences = {}
        self.__union_find = UnionFind()
        self.__token_to_equivalence = {}
        self.__colors = deque(self.COLORS)

    def check(self):
        # Copy the list before iterating, as items may delete themselves.
        return any(equivalence.check() for equivalence in list(self.equivalences))

    def check_special(self):
        # Copy the list before iterating, as items may delete themselves.
        return any(equivalence.check_special() for equivalence in list(self.equivalences))

    def already_paired(self, cell1: Cell, cell2: Cell) -> bool:
        return self.__union_find.find(cell1) == self.__union_find.find(cell2)

    def add_pair(self, cell1: Cell, cell2: Cell, name: str) -> bool:
        if self.already_paired(cell1, cell2):
            return False
        equivalence = _Equivalence(grid=self.grid, cells=[cell1, cell2], name=name)
        self.add_equivalence(equivalence)

    def add_equivalence(self, equivalence: _Equivalence) -> None:
        for a, b in pairwise(equivalence.cells):
            self.__union_find.union(a, b)
        self.equivalences[equivalence] = True  # Add ourselves to what is actually an ordered set
        self.__reassign_tokens_to_equivalences_as_necessary()

    def remove_equivalence(self, equivalence: _Equivalence) -> None:
        del self.equivalences[equivalence]

    def __reassign_tokens_to_equivalences_as_necessary(self):
        deletions = []
        self.__token_to_equivalence.clear()
        # We go through the equivalences in the order that they were created
        for equivalence in self.equivalences:
            # Find the token associated with this equivalence
            token = self.__union_find.find(equivalence.cells[0])
            # Does a previous equivalence already claim that token?
            old_equivalence = self.__token_to_equivalence.get(token)
            if not old_equivalence:
                # Claim this token as our own.
                self.__token_to_equivalence[token] = equivalence
            else:
                # equivalence and old_equivalence have been united.  Merge equivalence into old_equivalence
                old_name = str(old_equivalence)
                neighbors = equivalence.cells[0].neighbors | old_equivalence.cells[0].neighbors
                old_equivalence.set_all_neighbors(neighbors)
                old_equivalence.cells = list(unique_everseen(itertools.chain(old_equivalence.cells, equivalence.cells)))
                print(f'...Merging {equivalence} into {old_name} yielding {old_equivalence}')
                # Delete us once we're through iterating through the dict
                deletions.append(equivalence)
        for equivalence in deletions:
            self.remove_equivalence(equivalence)
            if equivalence.color:
                # We can re-use its color, if we really start to run low
                self.__colors.append(equivalence.color)
        if self.VERIFY:
            self.__verify()

    def __verify(self) -> None:
        for cell in self.grid.cells:
            assert cell not in cell.neighbors
            for neighbor in cell.neighbors:
                assert cell in neighbor.neighbors

        nodes = set(self.__union_find.all_nodes())
        for equivalence in self.equivalences:
            tokens = [self.__union_find.find(cell) for cell in equivalence.cells]
            token = tokens.pop()
            assert all(t == token for t in tokens)
            assert self.__token_to_equivalence[token] == equivalence
            cells = set(equivalence.cells)
            assert cells <= nodes
            nodes -= cells

    def draw(self, context: DrawContext) -> None:
        for equivalence in self.equivalences:
            if all(cell.is_known for cell in equivalence.cells):
                continue
            color = equivalence.color = equivalence.color or self.__colors.popleft()
            for cell in equivalence.cells:
                y, x = cell.index
                context.draw_circle((x + .5, y + .2), radius=.1, fill=True, color=color)
