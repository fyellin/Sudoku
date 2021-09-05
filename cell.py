from __future__ import annotations

import functools
import itertools
import operator
from collections import Callable
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AbstractSet, Any, ClassVar, Final, Iterable, NamedTuple, Optional, TYPE_CHECKING

from color import Color

if TYPE_CHECKING:
    from feature import Feature, Square
    from grid import Grid


class SmallIntSet:
    bits: int

    BITS_TO_TUPLE: dict[int, tuple[int, ...]] = {
        bits: items
        for count in range(10)
        for items in itertools.combinations(range(1, 10), count)
        for bits in [functools.reduce(operator.__or__, (1 << i for i in items), 0)]
    }

    @staticmethod
    def get_full_cell() -> SmallIntSet:
        # (1 << 1) + ... (1 << 9).  Note, there is no 1 << 0
        return SmallIntSet(1022)

    def __init__(self, items: int | Iterable[int] = 0) -> None:
        self.set_to(items)

    def set_to(self, items: int | Iterable[int]) -> None:
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

    def discard(self, item: int) -> None:
        self.bits &= ~(1 << item)

    def remove(self, item: int) -> None:
        if not self.bits & (1 << item):
            raise KeyError('item not in set')
        self.bits &= ~(1 << item)

    def unique(self) -> int:
        items = self.BITS_TO_TUPLE[self.bits]
        assert len(items) == 1
        return items[0]

    def copy(self) -> SmallIntSet:
        return SmallIntSet(self.bits)

    @classmethod
    def union(cls, *other_sets: SmallIntSet) -> SmallIntSet:
        values = functools.reduce(operator.__or__, (x.bits for x in other_sets), 0)
        return SmallIntSet(values)

    def isdisjoint(self, other: SmallIntSet) -> bool:
        return self.bits & other.bits == 0

    def __contains__(self, item: int) -> bool:
        return bool(self.bits & (1 << item))

    def __iter__(self) -> Iterator[int]:
        items = self.BITS_TO_TUPLE[self.bits]
        return iter(items)

    def __len__(self) -> int:
        return len(self.BITS_TO_TUPLE[self.bits])

    def as_sorted_tuple(self) -> tuple[int, ...]:
        return self.BITS_TO_TUPLE[self.bits]

    def __sub__(self, other: SmallIntSet | Iterable[int]) -> SmallIntSet:
        other_bits = other.bits if isinstance(other, SmallIntSet) else self.__to_bits(other)
        return SmallIntSet(self.bits & ~other_bits)

    def __isub__(self, other: SmallIntSet | Iterable[int]) -> SmallIntSet:
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
        return "/" + "".join(elements) + "/"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SmallIntSet):
            return NotImplemented
        return self.bits == other.bits

    def __hash__(self) -> int:
        return hash(self.bits)

    @staticmethod
    def __to_bits(items: Iterable[int]) -> int:
        return functools.reduce(operator.__or__, (1 << i for i in items), 0)


@dataclass
class House:
    class Type(Enum):
        ROW = auto()
        COLUMN = auto()
        BOX = auto()
        EXTRA = auto()
        EGG = auto()

    house_type: House.Type
    house_index: int
    cells: Sequence[Cell]
    unknown_cells: set[Cell] = field(init=False)
    unknown_values: SmallIntSet = field(default_factory=SmallIntSet.get_full_cell)

    def __post_init__(self) -> None:
        self.unknown_cells = set(self.cells)
        for cell in self.cells:
            cell.houses.append(self)

    def set_value_to(self, cell: Cell, value: int) -> None:
        try:
            self.unknown_cells.remove(cell)
            self.unknown_values.remove(value)
        except KeyError:
            print(f'Cannot remove {value} from {cell} in {self}')
            raise

    def __repr__(self) -> str:
        return self.house_type.name.title()[:3] + " " + str(self.house_index)

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __lt__(self, other: House) -> bool:
        return (self.house_type, self.house_index) < (other.house_type, other.house_index)


@dataclass
class Cell:
    PROTOTYPE_CELL: ClassVar[SmallIntSet] = SmallIntSet(range(1, 10))

    square: Square
    grid: Grid
    houses: Final[list[House]] = field(default_factory=list)  # Can this be left uninitialized?
    neighbors: frozenset[Cell] = field(default_factory=frozenset)  # Can this be left uninitialized?
    known_value: Optional[int] = None
    possible_values: SmallIntSet = field(default_factory=SmallIntSet.get_full_cell)

    def set_value_to(self, value: int, *, show: bool = False) -> str:
        for house in self.houses:
            house.set_value_to(self, value)
        for neighbor in self.neighbors:
            neighbor.possible_values.discard(value)
            assert neighbor.possible_values, f'Deleted last value for {neighbor}'
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

    def get_equivalent_cells(self) -> tuple[Cell, ...]:
        return self.grid.same_value_handler.get_all_same_value_cells(self)

    def get_xor_pairs(self, value: int) -> Iterable[tuple[Cell, House]]:
        """A chain pair implies that exactly one of self=value or result=value is True."""
        return self.get_strong_pairs(value)

    def get_strong_pairs(self, value: int) -> Iterable[tuple[Cell, House]]:
        """A strong pair implies that at least one of self=value or result=value is True"""
        for house in self.houses:
            temp = [cell for cell in house.unknown_cells if cell != self and value in cell.possible_values]
            if len(temp) == 1:
                yield temp[0], house

    def get_weak_pairs(self, value: int) -> Iterable[tuple[Cell, House]]:
        """A strong pair implies that at most one of self=value or result=value is True.  Both may be False"""
        return ((cell, house) for house in self.houses
                for cell in house.unknown_cells if cell != self and value in cell.possible_values)

    def is_neighbor(self, other: Cell) -> bool:
        return other in self.neighbors

    def is_neighbor_for_value(self, other: Cell, value: int) -> bool:
        return other in self.neighbors or \
               any(other in feature.get_neighbors_for_value(self, value) for feature in self.grid.neighborly_features)

    def get_all_neighbors_for_value(self, value: int) -> frozenset[Cell]:
        immediate_neighbors = self.neighbors
        other_neighbors = {x for feature in self.grid.neighborly_features
                           for x in feature.get_neighbors_for_value(self, value)}
        if other_neighbors:
            return immediate_neighbors | other_neighbors
        else:
            return immediate_neighbors

    def joint_neighbors(self, other: Cell) -> Iterator[Cell]:
        return (cell for cell in self.neighbors if other.is_neighbor(cell))

    def __repr__(self) -> str:
        row, column = self.square
        return f"r{row}c{column}"

    def possible_value_string(self) -> str:
        return ''.join(str(i) for i in self.possible_values)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __lt__(self, other: Cell) -> bool:
        return self.square < other.square

    @staticmethod
    def __deleted(i: int) -> str:
        return f'[{Color.lightgrey}{Color.strikethrough}{i}{Color.reset}]'

    @staticmethod
    def remove_value_from_cells(cells: Iterable[Cell], value: int, *, show: bool = True) -> None:
        for cell in cells:
            foo = ''.join((Cell.__deleted(i) if i == value else str(i)) for i in cell.possible_values)
            cell.possible_values.remove(value)
            assert cell.possible_values
            if show:
                print(f'  {cell} = {foo}')

    @staticmethod
    def remove_values_from_cells(cells: Iterable[Cell], values: SmallIntSet | set[int], *, show: bool = True) -> None:
        if isinstance(values, AbstractSet):
            values = SmallIntSet(values)
        for cell in cells:
            if show:
                foo = ''.join((Cell.__deleted(i) if i in values else str(i)) for i in cell.possible_values)
                print(f'  {cell} = {foo}')
            cell.possible_values -= values
            assert cell.possible_values

    @staticmethod
    def keep_values_for_cell(cells: Iterable[Cell], values: SmallIntSet | set[int], *, show: bool = True) -> None:
        if isinstance(values, AbstractSet):
            values = SmallIntSet(values)
        for cell in cells:
            if show:
                output = ''.join((Cell.__deleted(i) if i not in values else str(i)) for i in cell.possible_values)
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

    def get_strong_pairs_extended(self) -> Iterable[tuple[CellValue, House | Feature | bool]]:
        """if cell ≠ value, then what cells are forced to have a value"""
        return self.__get_all_pairs_extended(lambda a, b: a.get_strong_pairs(b), False)

    def get_weak_pairs_extended(self) -> Iterable[tuple[CellValue, House | Feature | bool]]:
        """if cell == value, then what cells can't have which values?"""
        return self.__get_all_pairs_extended(lambda a, b: a.get_weak_pairs(b), True)

    def get_xor_pairs_extended(self) -> Iterable[tuple[CellValue, House | Feature | bool]]:
        """Which cells have which values if and only if this cell doesn't have the given value?"""
        return self.__get_all_pairs_extended(lambda a, b: a.get_xor_pairs(b), False)

    def __get_all_pairs_extended(self, func: Callable[..., Any], is_weak: bool) -> \
            Iterable[tuple[CellValue, House | Feature | bool]]:
        original_cell, value = self
        for cell in original_cell.get_equivalent_cells():
            yield from ((CellValue(cell2, value), house2) for cell2, house2 in func(cell, value))
            if is_weak or len(cell.possible_values) == 2:
                yield from ((CellValue(cell, value2), True) for value2 in cell.possible_values - {value})
            cell_value = CellValue(cell, value)
            yield from ((cv, feature)
                        for feature in cell.grid.pair_features
                        for cv in func(feature, cell_value))
