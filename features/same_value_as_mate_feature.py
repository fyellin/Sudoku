import abc
from typing import Sequence, Iterable

from cell import Cell, SmallIntSet
from feature import Feature, Square
from features.chess_move import KnightsMoveFeature
from grid import Grid


class AbstractMateFeature(Feature, abc.ABC):
    """Handles messages involving a square and its mates"""
    this_square: Square
    this_cell: Cell
    possible_mates: Sequence[Cell]
    done: bool

    def __init__(self, square: Square):
        super().__init__()
        self.this_square = square

    def initialize(self, grid: Grid) -> None:
        super().initialize(grid)
        self.this_cell = self @ self.this_square
        self.possible_mates = list(self.get_mates(self.this_cell))
        self.done = False

    def get_mates(self, cell: Cell) -> Iterable[Cell]:
        return self.neighbors_from_offsets(cell, KnightsMoveFeature.OFFSETS)

    def check(self) -> bool:
        if self.done:
            return False
        if self.this_cell.is_known:
            assert self.this_cell.known_value is not None
            return self._check_value_known(self.this_cell.known_value)
        else:
            return self._check_value_not_known()

    @abc.abstractmethod
    def _check_value_known(self, value: int) -> bool:
        """Handle the case of this cell having a known value"""
        ...

    @abc.abstractmethod
    def _check_value_not_known(self) -> bool:
        """Handle the case of this cell not having a value yet"""
        ...


class SameValueAsExactlyOneMateFeature(AbstractMateFeature):
    """The square must have the same value as exactly one of its mates"""
    def _check_value_known(self, value: int) -> bool:
        # We must make sure that the known value has exactly one mate
        count = sum(1 for cell in self.possible_mates if cell.is_known and cell.known_value == value)
        mates = [cell for cell in self.possible_mates if not cell.is_known and value in cell.possible_values]
        assert count < 2
        if count == 1:
            self.done = True
            if mates:
                print(f'Cell {self.this_cell} can only have one mate')
                Cell.remove_value_from_cells(mates, value)
                return True
            return False
        elif len(mates) == 1:
            print(f'Cell {self.this_cell} only has one possible mate')
            mates[0].set_value_to(value, show=True)
            self.done = True
            return True
        return False

    def _check_value_not_known(self) -> bool:
        # The only possible values for this cell are those values for which it can have one mate.
        impossible_values = set()
        for value in self.this_cell.possible_values:
            count = sum(1 for cell in self.possible_mates if cell.is_known and cell.known_value == value)
            mates = [cell for cell in self.possible_mates if not cell.is_known and value in cell.possible_values]
            if count >= 2 or (count == 0 and not mates):
                impossible_values.add(value)
        if impossible_values:
            print(f'Cell {self.this_cell} must have exactly one mate value')
            Cell.remove_values_from_cells([self.this_cell], impossible_values)
            return True
        return False


class SameValueAsMateFeature(AbstractMateFeature):
    """The square must have the same value as at least one of its mates"""
    def _check_value_known(self, value: int) -> bool:
        if any(cell.is_known and cell.known_value == value for cell in self.possible_mates):
            # We didn't change anything, but we've verified that this guy has a mate
            self.done = True
            return False
        mates = [cell for cell in self.possible_mates if not cell.is_known and value in cell.possible_values]
        assert len(mates) >= 1
        if len(mates) == 1:
            print(f'Cell {self.this_cell} has only one possible mate')
            mates[0].set_value_to(value, show=True)
            self.done = True
            return True
        return False

    def _check_value_not_known(self) -> bool:
        legal_values = SmallIntSet.union(*(cell.possible_values for cell in self.possible_mates))
        if not self.this_cell.possible_values <= legal_values:
            print(f'Cell {self.this_cell} must have a mate')
            Cell.keep_values_for_cell([self.this_cell], legal_values)
            return True
        return False
