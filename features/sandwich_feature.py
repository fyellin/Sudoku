from __future__ import annotations

import datetime
import functools
from collections import defaultdict
from itertools import permutations
from typing import Iterable, Mapping, Optional, Sequence

from cell import House, SmallIntSet
from draw_context import DrawContext
from features.possibilities_feature import HousePossibilitiesFeature, PossibilitiesFeature


class SandwichFeature(HousePossibilitiesFeature):
    """Specify the total sum of the squares between the 1 and the 9."""
    total: int

    @staticmethod
    def all(htype: House.Type, totals: Sequence[Optional[int]]) -> Sequence[SandwichFeature]:
        """Used to set sandwiches for an entire row or column.   A none indicates missing"""
        return [SandwichFeature(htype, rc, total) for rc, total in enumerate(totals, start=1) if total is not None]

    def __init__(self, htype: House.Type, index: int, total: int):
        super().__init__(htype, index, name="Sandwich")
        self.total = total

    def match(self, permutation: tuple[int, ...]) -> bool:
        return self.sandwich_sum(permutation) == self.total

    @classmethod
    def sandwich_sum(cls, permutation):
        index1 = permutation.index(1)
        index2 = permutation.index(9)
        if index1 < index2:
            return sum(permutation[i] for i in range(index1 + 1, index2))
        else:
            return sum(permutation[i] for i in range(index2 + 1, index1))

    ONE_AND_NINE = SmallIntSet((1, 9))

    def draw(self, context: DrawContext) -> None:
        context.draw_outside(self.total, self.htype, self.index, fontsize=20, weight='bold')
        if not context.get(self.__class__):
            context[self.__class__] = True
            special = [square for square in self.all_squares()
                       if (self @ square).possible_values.isdisjoint(self.ONE_AND_NINE)]
            context.draw_rectangles(special, color='lightgreen')


class SandwichXboxFeature(PossibilitiesFeature):
    htype: House.Type
    row_column: int
    value: int
    is_right: bool

    def __init__(self, htype: House.Type, row_column: int, value: int, right: bool = False) -> None:
        name = f'Skyscraper {htype.name.title()} #{row_column}'
        squares = self.get_house_squares(htype, row_column)
        self.htype = htype
        self.row_column = row_column
        self.value = value
        self.is_right = right
        super().__init__(squares, name=name)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        result = self._get_all_possibilities()[self.value]
        if not self.is_right:
            return result
        else:
            return (item[::-1] for item in result)

    @staticmethod
    @functools.lru_cache(None)
    def _get_all_possibilities() -> Mapping[int, Sequence[tuple[int, ...]]]:
        result: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        start = datetime.datetime.now()
        for values in permutations(range(1, 10)):
            index1 = values.index(1)
            index2 = values.index(9)
            if index2 < index1:
                index2, index1 = index1, index2
            sandwich = sum([values[index] for index in range(index1 + 1, index2)])
            xbox = sum([values[index] for index in range(values[0])])
            if sandwich == xbox:
                result[sandwich].append(values)
        end = datetime.datetime.now()
        print(f'Initialization = {end - start}.')
        return result

    def draw(self, context: DrawContext) -> None:
        args = dict(fontsize=20, weight='bold')
        context.draw_outside(self.value, self.htype, self.row_column, is_right=self.is_right, **args)
