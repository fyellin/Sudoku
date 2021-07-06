from __future__ import annotations

import itertools
from collections import deque
from enum import Enum, auto
from typing import Set, Iterable, Tuple, Iterator, Sequence, Mapping, Dict, NamedTuple, FrozenSet, List, Deque, \
    TYPE_CHECKING

from cell import CellValue, Cell
from color import Color

if TYPE_CHECKING:
    from feature import Feature


class Chain:
    one: FrozenSet[CellValue]
    two: FrozenSet[CellValue]
    is_medusa: bool

    def __init__(self, one: Iterable[CellValue], two: Iterable[CellValue], is_medusa: bool):
        self.one = frozenset(one)
        self.two = frozenset(two)
        self.is_medusa = is_medusa

    class Group (Enum):
        ONE = auto()
        TWO = auto()

        def pick_set(self, chain: 'Chain') -> Iterable[CellValue]:
            return chain.one if self == Chain.Group.ONE else chain.two

        def pick_other_set(self, chain: 'Chain') -> Iterable[CellValue]:
            return chain.two if self == Chain.Group.ONE else chain.one

        def other(self) -> 'Chain.Group':
            return Chain.Group.ONE if self == Chain.Group.TWO else Chain.Group.TWO

        def color(self) -> str:
            return Color.blue if self == Chain.Group.ONE else Color.red

        def marker(self) -> str:
            return f'{self.color()}■{Color.reset}'

    @staticmethod
    def create(start: CellValue, medusa: bool) -> Chain:
        todo = deque([(start, 0)])
        seen = {start}
        one: List[CellValue] = []
        two: List[CellValue] = []
        while todo:
            cell_value, depth = todo.popleft()
            (one if depth % 2 == 0 else two).append(cell_value)
            (this_cell, this_value) = cell_value
            for house in this_cell.all_houses():
                next_cell = this_cell.strong_pair(house, this_value)
                if next_cell is None:
                    continue
                next_cell_value = CellValue(next_cell, this_value)
                if next_cell_value not in seen:
                    seen.add(next_cell_value)
                    todo.append((next_cell_value, depth + 1))
            if medusa and len(this_cell.possible_values) == 2:
                next_value = (this_cell.possible_values - {this_value}).pop()
                next_cell_value = CellValue(this_cell, next_value)
                if next_cell_value not in seen:
                    seen.add(next_cell_value)
                    todo.append((next_cell_value, depth + 1))
        return Chain(one, two, medusa)

    def check_colors(self, features: Sequence['Feature']) -> bool:
        """Pairwise look at each two elements on this chain and see if they lead to insight or a contradiction"""
        for (cell_value1, group1), (cell_value2, group2) in itertools.combinations(self.items(), 2):
            (cell1, value1), (cell2, value2) = cell_value1, cell_value2
            if group1 == group2:
                # Either cell1=value1 and cell2=value2 are both true or are both false
                if (cell1 == cell2 and value1 != value2) \
                        or (value1 == value2 and cell1.is_neighbor(cell2)) \
                        or (value1 == value2 and any(cell2 in feature.get_neighbors_for_value(cell1, value1)
                                                     for feature in features)):
                    # If both are true, it leads to a contradiction.  Both must be false.
                    print(f"Setting value of {self} to {group1.marker()} causes contradiction.")
                    print(f"Both {cell_value1} and {cell_value2} cannot be true")
                    self.set_true(group1.other())
                    return True
            else:
                # Precisely one of cell1 = value1 or cell2 = value2 is true
                if value1 == value2:
                    # The two cells have the same value.  See if they both see an element in common
                    fixers = [cell for cell in cell1.joint_neighbors(cell2) if value1 in cell.possible_values]
                    if fixers:
                        print(f"From {self.__sub_chain_string(cell_value1, cell_value2)}, "
                              f"either {cell1} or {cell2} is {value2}.")
                        Cell.remove_value_from_cells(fixers, value1)
                        return True
                elif cell1 == cell2:
                    # Two different possible values for the cell.  If there are any others, they can be tossed
                    assert {value1, value2} <= cell1.possible_values
                    if len(cell1.possible_values) >= 3:
                        print(f"From {self.__sub_chain_string(cell_value1, cell_value2)}, "
                              f"{cell1} is either ={value1} or {value2}")
                        delta = cell1.possible_values - {value1, value2}
                        Cell.remove_values_from_cells([cell1], delta)
                        return True
                elif cell1.is_neighbor(cell2):
                    # Since cell1 and cell2 are neighbors, and either cell1=value1 or cell2=value2, in either case
                    # cell1 ≠ value2 and cell2 ≠ value1
                    if value2 in cell1.possible_values or value1 in cell2.possible_values:
                        print(f"From {self.__sub_chain_string(cell_value1, cell_value2)}, "
                              f"{cell1}≠{value2} and {cell2}≠{value1}")
                        for value, cell in ((value1, cell2), (value2, cell1)):
                            if value in cell.possible_values:
                                Cell.remove_value_from_cells([cell], value)
                        return True
        return False

    def __sub_chain_string(self, start: CellValue, end: CellValue) -> str:
        """Given two cell-values in this chain, print out the piece of the chain from "start" to "end" """
        todo: Deque[CellValue] = deque([end])
        seen = {end: end}
        while todo:
            cell_value: CellValue = todo.popleft()
            if cell_value == start:
                break
            (this_cell, this_value) = cell_value
            for house in this_cell.all_houses():
                next_cell = this_cell.strong_pair(house, this_value)
                if next_cell is None:
                    continue
                next_cell_value = CellValue(next_cell, this_value)
                if next_cell_value not in seen:
                    seen[next_cell_value] = cell_value
                    todo.append(next_cell_value)
            if self.is_medusa and len(this_cell.possible_values) == 2:
                next_value = (this_cell.possible_values - {this_value}).pop()
                next_cell_value = CellValue(this_cell, next_value)
                if next_cell_value not in seen:
                    seen[next_cell_value] = cell_value
                    todo.append(next_cell_value)
        cell_value = start
        group = Chain.Group.ONE
        items = []
        while True:
            items.append(f'{group.color()}{cell_value.cell}={cell_value.value}{Color.reset}')
            if cell_value == end:
                break
            cell_value = seen[cell_value]
            group = group.other()
        return '<' + ' ⟺ '.join(items) + '>'

    def set_true(self, group: Chain.Group) -> None:
        """Make all cells that belong to the specified group be true, and the others be false."""
        for cell, value in group.pick_set(self):
            cell.set_value_to(value, show=True)
        for cell, value in group.pick_other_set(self):
            # It's possible the set_value above has made removing the value from the cell unnecessary.
            if value in cell.possible_values:
                Cell.remove_value_from_cells([cell], value)

    def items(self) -> Iterator[Tuple[CellValue, Chain.Group]]:
        """An enumeration of (cell_value, group) for all the cell values in this chain"""
        yield from ((cell, Chain.Group.ONE) for cell in self.one)
        yield from ((cell, Chain.Group.TWO) for cell in self.two)

    def to_string(self, group: Chain.Group) -> str:
        items: Set[Tuple[CellValue, str]] = set()
        items.update((cv, '=') for cv in group.pick_set(self))
        items.update((cv, '≠') for cv in group.pick_other_set(self))
        return ', '.join(f'{cell}{symbol}{value}' for (cell, value), symbol in sorted(items))

    def __repr__(self) -> str:
        joined = ', '.join(f'{group.color()}{cell}={value}{Color.reset}'
                           for (cell, value), group in sorted(self.items()))
        return '<' + joined + '>'

    def __len__(self) -> int:
        return len(self.one) + len(self.two)


class Chains (NamedTuple):
    chains: Sequence[Chain]
    mapping: Mapping[CellValue, Tuple[Chain, Chain.Group]]

    @staticmethod
    def create(all_cells: Iterable[Cell], medusa: bool) -> Chains:
        mapping: Dict[CellValue, Tuple[Chain, Chain.Group]] = {}
        chains = []
        for cell in all_cells:
            if cell.is_known:
                continue
            for value in cell.possible_values:
                cell_value = CellValue(cell, value)
                if cell_value not in mapping:
                    chain = Chain.create(cell_value, medusa)
                    chains.append(chain)
                    mapping.update((cv, (chain, group)) for cv, group in chain.items())
        chains.sort(key=lambda x: len(x), reverse=True)
        return Chains(chains, mapping)
