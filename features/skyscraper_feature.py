import itertools
from collections.abc import Sequence, Iterable
from typing import Optional, cast

from cell import House
from draw_context import DrawContext
from features.possibilities_feature import GroupedPossibilitiesFeature


class SkyscraperFeature(GroupedPossibilitiesFeature):
    htype: House.Type
    row_column: int
    left: Optional[int]
    right: Optional[int]
    basement: Sequence[tuple[int, int]]

    def __init__(self, htype: House.Type, row_column: int,
                 left: Optional[int] = None, right: Optional[int] = None,
                 *, basement: Sequence[tuple[int, int]] = ()):
        name = f'Skyscraper {htype.name.title()} #{row_column}'
        squares = [square for square in self.get_house_squares(htype, row_column) if square not in basement]
        super().__init__(squares, name=name)
        self.htype = htype
        self.row_column = row_column
        self.left = left
        self.right = right
        self.basement = basement

    def get_possibilities(self) -> Iterable[Sequence[set[int]]]:
        if self.left and self.right:
            return self.__get_possibilities2(self.left, self.right, len(self.squares))
        elif self.left:
            return self.__get_possibilities1(self.left, len(self.squares))
        else:
            return map(lambda x: x[::-1], self.__get_possibilities1(cast(int, self.right), len(self.squares)))

    def __get_possibilities1(self, left: int, length: int) -> Iterable[tuple[set[int], ...]]:
        shadowed_count = length - left  # Number of values that need to be overshadowed
        for towers in ((*values, peak) for peak in range(length, 10)
                       for values in itertools.combinations(range(1, peak), left - 1)):
            for groupings in self.__combine_towers_and_shadowed(towers, shadowed_count):
                yield tuple(x for grouping in groupings for x in grouping)

    def __get_possibilities2(self, left: int, right: int, length: int) -> Iterable[tuple[set[int], ...]]:
        shadowed_count = length - (left + right - 1)
        for towers in ((*values, off_peak)
                       for off_peak in range(length - 1, 9)  # need to leave space for peak
                       for values in itertools.combinations(range(1, off_peak), left + right - 3)):
            off_peak = towers[-1]
            for groupings in self.__combine_towers_and_shadowed(towers, shadowed_count):
                for left_indices in itertools.combinations(range(len(towers)), left - 1):
                    right_indices = [x for x in range(len(towers)) if x not in left_indices]
                    yield (
                        *(x for index in left_indices for x in groupings[index]),
                        set(range(off_peak + 1, 10)),
                        *(x for index in reversed(right_indices) for x in reversed(groupings[index]))
                    )

    @staticmethod
    def __combine_towers_and_shadowed(towers: Sequence[int], shadowed_count: int) -> \
            Iterable[Sequence[list[set[int]]]]:
        # if shadowed_count == 0:
        #     return [[{tower}] for tower in towers],
        shadowable_values = [x for x in range(1, towers[-1]) if x not in towers]
        shadowable_values_by_tower = [{x for x in shadowable_values if x < tower} for tower in towers]

        def inner(tower_index: int, currently_shadowed: int) -> Iterable[list[list[set[int]]]]:
            # Get the current tower, and the values that can hide behind it.
            this_tower = towers[tower_index]
            these_shadowable_values = shadowable_values_by_tower[tower_index]
            if tower_index == len(towers) - 1:
                # The last tower.  We include this tower and the right number of shadows behind it
                yield [[{this_tower}] + [these_shadowable_values] * (shadowed_count - currently_shadowed)]
                return
            # The amount of items we put behind the current tower is next_shadowed - currently_shadowed.
            # Any amount is okay, as long as our total doesn't go over shadowed_count, and we don't use more towers
            # than are available to hide behind the current tower.
            for next_shadowed in range(currently_shadowed, min(shadowed_count, len(these_shadowable_values)) + 1):
                this_group = [{this_tower}] + [these_shadowable_values] * (next_shadowed - currently_shadowed)
                for result in inner(tower_index + 1, next_shadowed):
                    result.insert(0, this_group)
                    yield result

        return inner(0, 0)

    def draw(self, context: DrawContext) -> None:
        args = dict(fontsize=20, weight='bold')
        if self.left:
            context.draw_outside(self.left, self.htype, self.row_column, **args)
        if self.right:
            context.draw_outside(context, self.right, self.htype, self.row_column, is_right=True, **args)
        context.draw_rectangles(self.basement, facecolor="lightgrey")


def main() -> None:
    basement = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9))
    temp = SkyscraperFeature(House.Type.ROW, 1, 5, 2, basement=basement[0:0])
    count = 0
    for foo in temp.get_possibilities():
        count += 1
        print(len(foo), foo)
    print(count)


if __name__ == '__main__':
    main()
