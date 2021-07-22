from itertools import combinations
from typing import Union, Sequence, Optional, Iterable

from draw_context import DrawContext
from feature import Square
from features.features import AdjacentRelationshipFeature
from features.chess_move import _draw_thermometer
from features.possibilities_feature import PossibilitiesFeature, GroupedPossibilitiesFeature


class Thermometer1Feature(AdjacentRelationshipFeature):
    """
    A sequence of squares that must monotonically increase.

    If slow is set, then this is a "slow" thermometer, and two adjacent numbers can be the same.  Typically,
    thermometers must be strictly monotonic.

    This implementation uses "adjacency"
    """
    def __init__(self, thermometer: Union[Sequence[Square], str], *,
                 name: Optional[str] = None, color: str = 'lightgrey') -> None:
        super().__init__(thermometer, name=name, color=color)

    def match(self, digit1: int, digit2: int) -> bool:
        return digit1 < digit2

    def draw(self, context: DrawContext) -> None:
        _draw_thermometer(self.squares, self.color, context)


class Thermometer2Feature(PossibilitiesFeature):
    """
    A sequence of squares that must monotonically increase.
    This is implemented as a subclass of Possibilities Feature.  Not sure which implementation is better.
    """
    color: str

    def __init__(self, thermometer: Union[Sequence[Square], str], *,
                 name: Optional[str] = None, color: str = 'lightgrey'):
        super().__init__(thermometer, name=name)
        self.color = color

    def draw(self, context: DrawContext) -> None:
        _draw_thermometer(self.squares, self.color, context)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        return combinations(range(1, 10), len(self.squares))


class Thermometer3Feature(GroupedPossibilitiesFeature):
    """
    A sequence of squares that must monotonically increase.
    This is implemented as a subclass of Possibilities Feature.  Not sure which implementation is better.
    """
    color: str

    def __init__(self, thermometer: Union[Sequence[Square], str],
                 name: Optional[str] = None, color: str = 'lightgrey'):
        super().__init__(thermometer, name=name)
        self.color = color

    def draw(self, context: DrawContext) -> None:
        _draw_thermometer(self.squares, self.color, context)

    def get_possibilities(self) -> Iterable[tuple[set[int], ...]]:
        length = len(self.squares)
        if length > 2:
            for permutation in combinations(range(2, 9), length - 2):
                yield (set(range(1, permutation[0])),
                       *({x} for x in permutation),
                       set(range(permutation[-1] + 1, 10)))
        else:
            for i in range(1, 9):
                yield {i}, set(range(i + 1, 10))


class ThermometerFeature(Thermometer3Feature):
    pass


class ThermometerAsLessThanFeature(ThermometerFeature):
    """A Thermometer of two squares, where we draw a < sign between them"""
    def __init__(self, thermometer: Union[Sequence[Square], str], *, name: Optional[str] = None) -> None:
        super().__init__(thermometer, name=name)
        assert len(self.squares) == 2

    def draw(self, context: DrawContext) -> None:
        assert len(self.squares) == 2
        (r1, c1), (r2, c2) = self.squares
        y, x = (r1 + r2) / 2, (c1 + c2) / 2
        dy, dx = (r2 - r1), (c2 - c1)
        context.draw_text(x + .5, y + .5, '>' if (dy == 1 or dx == -1) else '<',
                          verticalalignment='center', horizontalalignment='center',
                          rotation=(90 if dx == 0 else 0),
                          fontsize=20, weight='bold')


class SlowThermometerFeature(Thermometer1Feature):
    """A thermometer in which the digits only need to be ≤ rather than <"""
    def match(self, digit1: int, digit2: int) -> bool:
        return digit1 <= digit2
