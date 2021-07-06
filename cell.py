from __future__ import annotations

from enum import Enum, auto
from collections.abc import Sequence, Iterable, Iterator
from typing import Optional, Final, TYPE_CHECKING, NamedTuple

from color import Color
if TYPE_CHECKING:
    from feature import Feature
    from grid import Grid


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
    unknown_values: set[int]
    unknown_cells: set[Cell]

    def __init__(self, house_type: House.Type, index: int, cells: Sequence[Cell]) -> None:
        self.house_type = house_type
        self.index = index
        self.cells = cells
        self.unknown_values = set()
        self.unknown_cells = set()
        for cell in self.cells:
            cell.houses.append(self)

    def reset(self) -> None:
        self.unknown_values = set(range(1, 10))
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
    neighbors: set[Cell]
    features: Final[Sequence['Feature']]

    known_value: Optional[int]
    possible_values: set[int]

    def __init__(self, row: int, column: int, features: Sequence['Feature']) -> None:
        self.index = (row, column)
        self.features = features
        self.known_value = None
        self.possible_values = set(range(1, 10))
        self.neighbors = set()  # Filled in later
        self.houses = []

    def reset(self) -> None:
        self.known_value = None
        self.possible_values = set(range(1, 10))

    def set_value_to(self, value: int, *, show: bool = False) -> str:
        for house in self.houses:
            house.set_value_to(self, value)
        for neighbor in self.neighbors:
            neighbor.possible_values.discard(value)
        for neighbor in {cell for feature in self.features for cell in feature.get_neighbors_for_value(self, value)}:
            neighbor.possible_values.discard(value)

        assert value in self.possible_values
        self.known_value = value
        self.possible_values.clear()
        self.possible_values.add(value)
        output = f'{self} := {value}'  # Maybe use ⬅
        if show:
            print(f'  {output}')
        return output

    @property
    def is_known(self) -> bool:
        return self.known_value is not None

    def initialize_neighbors(self, _grid: 'Grid') -> None:
        neighbors: set[Cell] = set()
        for house in self.all_houses():
            neighbors.update(house.cells)
        neighbors.update(cell for feature in self.features for cell in feature.get_neighbors(self))
        neighbors.remove(self)
        self.neighbors = neighbors

    def all_houses(self) -> Iterable[House]:
        return self.houses

    def all_houses_but(self, house: House) -> Iterable[House]:
        return (x for x in self.houses if x != house)

    def house_of_type(self, house_type: House.Type) -> House:
        return next(house for house in self.houses if house.house_type == house_type)

    def weak_pair(self, house: House, value: int) -> Sequence[Cell]:
        temp = [cell for cell in house.unknown_cells
                if cell != self and value in cell.possible_values]
        return temp

    def strong_pair(self, house: House, value: int) -> Optional[Cell]:
        temp = self.weak_pair(house, value)
        return temp[0] if len(temp) == 1 else None

    def is_neighbor(self, other: Cell) -> bool:
        return other in self.neighbors

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
            if show:
                print(f'  {cell} = {foo}')

    @staticmethod
    def remove_values_from_cells(cells: Iterable[Cell], values: set[int], *, show: bool = True) -> None:
        for cell in cells:
            if show:
                foo = ''.join((Cell.__deleted(i) if i in values else str(i)) for i in sorted(cell.possible_values))
                print(f'  {cell} = {foo}')
            cell.possible_values.difference_update(values)

    @staticmethod
    def keep_values_for_cell(cells: Iterable[Cell], values: set[int], *, show: bool = True) -> None:
        for cell in cells:
            if show:
                foo = ''.join((Cell.__deleted(i) if i not in values else str(i)) for i in sorted(cell.possible_values))
                print(f'  {cell} = {foo}')
            cell.possible_values.intersection_update(values)


class CellValue(NamedTuple):
    cell: Cell
    value: int

    def __repr__(self) -> str:
        return f'{self.cell}={self.value}'

    def to_string(self, truth: bool) -> str:
        char = '=' if truth else '≠'
        return f'{self.cell}{char}{self.value}'
