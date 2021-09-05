import functools
from collections.abc import Iterable, Sequence
from typing import AbstractSet

from cell import Cell
from draw_context import DrawContext
from feature import Feature, Square


class KnightsMoveFeature(Feature):
    """No two squares within a knight's move of each other can have the same value."""
    OFFSETS = [(dr, dc) for dx in (-1, 1) for dy in (-2, 2) for (dr, dc) in ((dx, dy), (dy, dx))]

    def get_neighbors(self, cell: Cell) -> Iterable[Cell]:
        return self.neighbors_from_offsets(cell, self.OFFSETS)


class KingsMoveFeature(Feature):
    """No two pieces within a king's move of each other can have the same value."""
    OFFSETS = [(dr, dc) for dr in (-1, 1) for dc in (-1, 1)]

    def get_neighbors(self, cell: Cell) -> Iterable[Cell]:
        return self.neighbors_from_offsets(cell, self.OFFSETS)


class QueensMoveFeature(Feature):
    OFFSETS = [(dr, dc) for delta in range(1, 9) for dr in (-delta, delta) for dc in (-delta, delta)]
    values: AbstractSet[int]

    def __init__(self, values: AbstractSet[int] = frozenset({9})):
        super().__init__()
        self.values = values

    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        if value in self.values:
            return self.neighbors_from_offsets(cell, self.OFFSETS)
        else:
            return ()


class TaxicabFeature(Feature):
    """Two squares with the same value cannot have "value" as the taxicab distance between them."""
    taxis: set[int]

    def __init__(self, taxis: Sequence[int] = ()):
        super().__init__()
        self.taxis = set(taxis) if taxis else set(range(1, 10))

    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        if value in self.taxis:
            offsets = self.__get_offsets_for_value(value)
            return self.neighbors_from_offsets(cell, offsets)
        else:
            return ()

    @staticmethod
    @functools.lru_cache()
    def __get_offsets_for_value(value: int) -> Sequence[Square]:
        result = [square for i in range(0, value)
                  for square in [(i - value, i), (i, value - i), (value - i, -i), (-i, i - value)]]
        return result


def _draw_thermometer(squares: Sequence[Square], color: str, context: DrawContext) -> None:
    context.draw_line(squares, color=color, linewidth=10)
    row, column = squares[0]
    context.draw_circle((column + .5, row + .5), radius=.3, fill=True, facecolor=color)


class LittlePrincessFeature(Feature):
    """The taxicab distance between two like values cannot be that value"""
    def get_neighbors_for_value(self, cell: Cell, value: int) -> Iterable[Cell]:
        offsets = self.__get_offsets_for_value(value)
        return self.neighbors_from_offsets(cell, offsets)

    @staticmethod
    @functools.lru_cache
    def __get_offsets_for_value(value: int) -> Sequence[Square]:
        return [(dr, dc) for delta in range(1, value)
                for dr in (-delta, delta) for dc in (-delta, delta)]
