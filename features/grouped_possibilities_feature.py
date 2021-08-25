from __future__ import annotations

import abc
from itertools import combinations, product
from typing import Iterable, Optional, Sequence

from cell import Cell, SmallIntSet
from draw_context import DrawContext
from feature import Feature, Square, SquaresParseable
from features.possibilities_feature import PossibilitiesFeature, Possibility


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
    def get_possibilities(self) -> Iterable[tuple[SmallIntSet | set[int] | Iterable[int] | int], ...]:
        ...

    def start(self) -> None:
        self.cells = [self@square for square in self.squares]

        def fixit_one(x: SmallIntSet | Iterable[int] | set[int] | int) -> SmallIntSet:
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
            elif cell1.square == cell2.square:
                possibilities = [p for p in possibilities if p[index1] == p[index2]]
        return possibilities

    def to_possibility_feature(self) -> PossibilitiesFeature:
        parent = self

        class ChildPossibilityFeature(PossibilitiesFeature):
            def __init__(self) -> None:
                super().__init__(parent.squares, name=parent.name, neighbors=True, duplicates=True)

            def draw(self, context: DrawContext) -> None:
                parent.draw(context)

            def get_possibilities(self) -> Iterable[Possibility]:
                for element in parent.get_possibilities():
                    yield from product(*element)

        return ChildPossibilityFeature()
