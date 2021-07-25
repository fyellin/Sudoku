from __future__ import annotations

import functools
import itertools
import operator
from collections.abc import Sequence, Iterable, Iterator
from enum import Enum, auto
from typing import Optional, Final, TYPE_CHECKING, NamedTuple, Union, AbstractSet

from color import Color

if TYPE_CHECKING:
    from feature import Feature
    from grid import Grid


class SmallIntSet:
    bits: int

    BITS_TO_TUPLE = {bits: items
                     for count in range(10)
                     for items in itertools.combinations(range(1, 10), count)
                     for bits in [functools.reduce(operator.__or__, (1 << i for i in items), 0)]
                     }

    def __init__(self, items: Union[int, Iterable[int]] = 0):
        self.set_to(items)

    def set_to(self, items: Union[int, Iterable[int]]):
        if isinstance(items, int):
            self.bits = items
        else:
            self.bits = self.__to_bits(items)
        assert self.bits & 1 == 0
        assert self.bits <= 1024

    def add(self, item: int) -> None:
        self.bits |= (1 << item)

    def clear(self) -> None:
        self.bits = 0

    def discard(self, item: int):
        self.bits &= ~(1 << item)

    def remove(self, item: int):
        if not self.bits & (1 << item):
            raise KeyError('item not in set')
        self.bits &= ~(1 << item)

    def unique(self) -> int:
        items = self.BITS_TO_TUPLE[self.bits]
        assert len(items) == 1
        return items[0]

    @classmethod
    def union(cls, *other_sets: SmallIntSet) -> SmallIntSet:
        values = functools.reduce(operator.__or__, (x.bits for x in other_sets), 0)
        return SmallIntSet(values)

    def isdisjoint(self, other: SmallIntSet) -> bool:
        return self.bits & other.bits == 0

    def __contains__(self, item: int) -> bool:
        return bool(self.bits & (1 << item))

    def __iter__(self) -> Iterable[int]:
        items = self.BITS_TO_TUPLE[self.bits]
        return iter(items)

    def __len__(self) -> int:
        return len(self.BITS_TO_TUPLE[self.bits])

    def __sub__(self, other: Union[SmallIntSet, Iterable[int]]) -> SmallIntSet:
        other_bits = other.bits if isinstance(other, SmallIntSet) else self.__to_bits(other)
        return SmallIntSet(self.bits & ~other_bits)

    def __isub__(self, other: Union[SmallIntSet, Iterable[int]]) -> SmallIntSet:
        other_bits = other.bits if isinstance(other, SmallIntSet) else self.__to_bits(other)
        self.bits &= ~other_bits
        return self

    def __and__(self, other: SmallIntSet) -> SmallIntSet:
        return SmallIntSet(self.bits & other.bits)

    def __iand__(self, other: SmallIntSet) -> SmallIntSet:
        self.bits &= other.bits
        return self

    def __or__(self, other: SmallIntSet) -> SmallIntSet:
        return SmallIntSet(self.bits | other.bits)

    def __ior__(self, other: SmallIntSet) -> SmallIntSet:
        self.bits |= other.bits
        return self

    def __le__(self, other: SmallIntSet) -> bool:
        return self.bits & ~other.bits == 0

    def __str__(self) -> str:
        elements = [str(x) for x in self]
        return "{" + ", ".join(elements) + "}"

    def __eq__(self, other: SmallIntSet):
        return self.bits == other.bits

    def __hash__(self):
        return hash(self.bits)

    @staticmethod
    def __to_bits(items):
        return functools.reduce(operator.__or__, (1 << i for i in items), 0)


class House:
    class Type(Enum):
        ROW = auto()
        COLUMN = auto()
        BOX = auto()
        EXTRA = auto()
        EGG = auto()

    house_type: Final[House.Type]
    index: Final[int]
    cells: Final[Sequence[Cell]]
    unknown_values: SmallIntSet
    unknown_cells: set[Cell]

    def __init__(self, house_type: House.Type, index: int, cells: Sequence[Cell]) -> None:
        self.house_type = house_type
        self.index = index
        self.cells = cells
        self.unknown_values = SmallIntSet()
        self.unknown_cells = set()
        for cell in self.cells:
            cell.houses.append(self)

    def start(self) -> None:
        self.unknown_values = SmallIntSet(range(1, 10))
        self.unknown_cells = set(self.cells)

    def __repr__(self) -> str:
        return self.house_type.name.title()[:3] + " " + str(self.index)

    def set_value_to(self, cell: Cell, value: int) -> None:
        try:
            self.unknown_cells.remove(cell)
            self.unknown_values.remove(value)
        except KeyError:
            print(f'Cannot remove {value} from {cell} in {self}')
            raise

    def __lt__(self, other: 'House') -> bool:
        return (self.house_type, self.index) < (other.house_type, other.index)


class Cell:
    houses: Final[list[House]]
    index: Final[tuple[int, int]]
    neighbors: frozenset[Cell]
    grid: Final[grid]

    known_value: Optional[int]
    possible_values: SmallIntSet

    def __init__(self, row: int, column: int, grid: Grid) -> None:
        self.index = (row, column)
        self.grid = grid
        self.known_value = None
        self.possible_values = SmallIntSet(range(1, 10))
        self.neighbors = frozenset()  # Filled in later
        self.houses = []

    def start(self) -> None:
        self.known_value = None
        self.possible_values = SmallIntSet(range(1, 10))

    def set_value_to(self, value: int, *, show: bool = False) -> str:
        for house in self.houses:
            house.set_value_to(self, value)
        for neighbor in self.neighbors:
            neighbor.possible_values.discard(value)
            assert neighbor.possible_values
        for neighbor in {cell
                         for feature in self.grid.neighborly_features
                         for cell in feature.get_neighbors_for_value(self, value)}:
            neighbor.possible_values.discard(value)
            assert neighbor.possible_values

        assert value in self.possible_values
        self.known_value = value
        self.possible_values.set_to([value])
        # self.possible_values.clear()
        # self.possible_values.add(value)
        output = f'{self} := {value}'  # Maybe use ⬅
        if show:
            print(f'  {output}')
        return output

    @property
    def is_known(self) -> bool:
        return self.known_value is not None

    @property
    def bitmap(self) -> int:
        return self.possible_values.bits

    def initialize_neighbors(self, _grid: 'Grid') -> None:
        neighbors = {n for house in self.all_houses() for n in house.cells}
        neighbors.update(n for feature in self.grid.neighborly_features for n in feature.get_neighbors(self))
        neighbors.remove(self)
        self.neighbors = frozenset(neighbors)

    def all_houses(self) -> Iterable[House]:
        return self.houses

    @functools.cache
    def house_of_type(self, house_type: House.Type) -> House:
        return next(house for house in self.houses if house.house_type == house_type)

    def strong_pair(self, value: int) -> Iterable[tuple[Cell, House]]:
        for house in self.houses:
            temp = [cell for cell in house.unknown_cells if cell != self and value in cell.possible_values]
            if len(temp) == 1:
                yield temp[0], house

    def extended_strong_pair(self, value: int) -> Iterable[tuple[Cell, int, Union[House, Feature, bool]]]:
        for cell2, house2 in self.strong_pair(value):
            yield cell2, value, house2
        if len(self.possible_values) == 2:
            yield self, (self.possible_values - {value}).unique(), True

    def weak_pair(self, value: int) -> Iterable[tuple[Cell, House]]:
        return ((cell, house) for house in self.houses
                for cell in house.unknown_cells if cell != self and value in cell.possible_values)

    def extended_weak_pair(self, value) -> Iterable[tuple[Cell, int, Union[House, Feature, bool]]]:
        for cell2, house2 in self.weak_pair(value):
            yield cell2, value, house2
        for value2 in self.possible_values - {value}:
            yield self, value2, True
        for feature in self.grid.weak_pair_features:
            for cell2, value2 in feature.weak_pair(self, value):
                yield cell2, value2, feature

    def is_neighbor(self, other: Cell) -> bool:
        return other in self.neighbors

    def is_neighbor_for_value(self, other: Cell, value: int):
        return other in self.neighbors or \
               any(other in feature.get_neighbors_for_value(self, value) for feature in self.grid.neighborly_features)

    def joint_neighbors(self, other: Cell) -> Iterator[Cell]:
        return (cell for cell in self.neighbors if other.is_neighbor(cell))

    def __repr__(self) -> str:
        row, column = self.index
        return f"r{row}c{column}"

    def possible_value_string(self) -> str:
        return ''.join(str(i) for i in sorted(self.possible_values))

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __lt__(self, other: Cell) -> bool:
        return self.index < other.index

    @staticmethod
    def __deleted(i: int) -> str:
        return f'[{Color.lightgrey}{Color.strikethrough}{i}{Color.reset}]'

    @staticmethod
    def remove_value_from_cells(cells: Iterable[Cell], value: int, *, show: bool = True) -> None:
        for cell in cells:
            foo = ''.join((Cell.__deleted(i) if i == value else str(i)) for i in sorted(cell.possible_values))
            cell.possible_values.remove(value)
            assert cell.possible_values
            if show:
                print(f'  {cell} = {foo}')

    @staticmethod
    def remove_values_from_cells(cells: Iterable[Cell], values: Union[SmallIntSet, set[int]], *,
                                 show: bool = True) -> None:
        if isinstance(values, AbstractSet):
            values = SmallIntSet(values)
        for cell in cells:
            if show:
                foo = ''.join((Cell.__deleted(i) if i in values else str(i)) for i in sorted(cell.possible_values))
                print(f'  {cell} = {foo}')
            cell.possible_values -= values
            assert cell.possible_values

    @staticmethod
    def keep_values_for_cell(cells: Iterable[Cell], values: Union[SmallIntSet, set[int]], *,
                             show: bool = True) -> None:
        if isinstance(values, AbstractSet):
            values = SmallIntSet(values)
        for cell in cells:
            if show:
                output = ''.join((Cell.__deleted(i) if i not in values else str(i))
                                 for i in sorted(cell.possible_values))
                print(f'  {cell} = {output}')
            cell.possible_values &= values
            assert cell.possible_values


class CellValue(NamedTuple):
    cell: Cell
    value: int

    def __repr__(self) -> str:
        return f'{self.cell}={self.value}'

    def to_string(self, truth: bool) -> str:
        char = '=' if truth else '≠'
        return f'{self.cell}{char}{self.value}'
