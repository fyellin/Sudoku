from __future__ import annotations

import functools
import itertools
from collections import defaultdict
from itertools import permutations
from typing import Iterable, Mapping, Optional, Sequence

from cell import House, SmallIntSet
from draw_context import DrawContext
from .possibilities_feature import HousePossibilitiesFeature, Possibility


class SandwichFeature(HousePossibilitiesFeature):
    """Specify the total sum of the squares between the 1 and the 9."""
    total: int

    @staticmethod
    def create_all(house_type: House.Type, totals: Sequence[Optional[int]]) -> Sequence[SandwichFeature]:
        """Used to set sandwiches for an entire row or column.   A none indicates missing"""
        return [SandwichFeature(house_type, rc, total)
                for rc, total in enumerate(totals, start=1)
                if total is not None]

    def __init__(self, house_type: House.Type, index: int, total: int):
        super().__init__(house_type, index, prefix="Sandwich")
        self.total = total

    def get_possibilities(self) -> Iterable[Possibility]:
        return self.get_all_generators()[self.total]

    @classmethod
    def sandwich_sum(cls, permutation: Possibility) -> int:
        index1 = permutation.index(1)
        index2 = permutation.index(9)
        if index1 < index2:
            return sum(permutation[i] for i in range(index1 + 1, index2))
        else:
            return sum(permutation[i] for i in range(index2 + 1, index1))

    @classmethod
    @functools.cache
    def get_all_generators(cls) -> Mapping[int, Sequence[Possibility]]:
        result: dict[int, list[Possibility]] = defaultdict(list)
        for permutation in itertools.permutations(range(1, 10)):
            result[cls.sandwich_sum(permutation)].append(permutation)
        return result

    ONE_AND_NINE = SmallIntSet((1, 9))

    def draw(self, context: DrawContext) -> None:
        context.draw_outside(self.total, self.house_type, self.house_index, fontsize=20, weight='bold')
        if not context.get(self.__class__):
            context[self.__class__] = True
            special = [square for square in self.all_squares()
                       if (self @ square).possible_values.isdisjoint(self.ONE_AND_NINE)]
            context.draw_rectangles(special, facecolor='lightgreen')


class XSumFeature(HousePossibilitiesFeature):
    value: int
    is_right: bool

    def __init__(self, house_type: House.Type, house_index: int, value: int, right: bool = False) -> None:
        super().__init__(house_type, house_index, prefix="XSum")
        self.value = value
        self.is_right = right

    def match(self, permutation: Possibility) -> bool:
        return True

    def generator(self) -> Iterable[Possibility]:
        results = self.get_all_generators()[self.value]
        if self.is_right:
            yield from (result[::-1] for result in results)
        else:
            yield from results

    @classmethod
    @functools.cache
    def get_all_generators(cls) -> Mapping[int, Sequence[Possibility]]:
        result: dict[int, list[Possibility]] = defaultdict(list)
        for permutation in itertools.permutations(range(1, 10)):
            result[cls.xsum(permutation)].append(permutation)
        return result

    @classmethod
    def xsum(cls, permutation: Possibility) -> int:
        return sum(permutation[0:permutation[0]])

    def draw(self, context: DrawContext) -> None:
        args = dict(fontsize=20, weight='bold')
        context.draw_outside(self.value, self.house_type, self.house_index, is_right=self.is_right, **args)


class SandwichXSumFeature(HousePossibilitiesFeature):
    value: int
    is_right: bool

    def __init__(self, house_type: House.Type, house_index: int, value: int, right: bool = False) -> None:
        super().__init__(house_type, house_index, prefix="Sandwich")
        self.value = value
        self.is_right = right

    def match(self, permutation: Possibility) -> bool:
        return True

    def generator(self) -> Iterable[Possibility]:
        results = self.get_all_generators()[self.value]
        if self.is_right:
            yield from (result[::-1] for result in results)
        else:
            yield from results

    @staticmethod
    @functools.lru_cache(None)
    def get_all_generators() -> Mapping[int, Sequence[Possibility]]:
        result: dict[int, list[Possibility]] = defaultdict(list)
        for permutation in permutations(range(1, 10)):
            sandwich = SandwichFeature.sandwich_sum(permutation)
            xsum = XSumFeature.xsum(permutation)
            if sandwich == xsum:
                result[sandwich].append(permutation)
        return result

    def draw(self, context: DrawContext) -> None:
        args = dict(fontsize=20, weight='bold')
        context.draw_outside(self.value, self.house_type, self.house_index, is_right=self.is_right, **args)

