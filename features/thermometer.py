from itertools import combinations, combinations_with_replacement
from typing import Iterable, Optional, Sequence, Union

from draw_context import DrawContext
from feature import Square, SquaresParseable
from .chess_move import _draw_thermometer
from .grouped_possibilities_feature import GroupedPossibilitiesFeature
from .possibilities_feature import PossibilitiesFeature


class ThermometerFeature(PossibilitiesFeature):
    """
    A sequence of squares that must monotonically increase.
    This is implemented as a subclass of Possibilities Feature.  Not sure which implementation is better.
    Thermometers typically have bulbs.  Ones that don't can increase in either direction.
    """
    color: str
    bulb: bool

    def __init__(self, thermometer: SquaresParseable, *, name: Optional[str] = None, color: str = 'lightgrey',
                 bulb: bool = True):
        super().__init__(thermometer, name=name)
        self.color = color
        self.bulb = bulb

    def draw(self, context: DrawContext) -> None:
        if self.bulb:
            _draw_thermometer(self.squares, self.color, context)
        else:
            context.draw_line(self.squares)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        yield from combinations(range(1, 10), len(self.squares))
        if not self.bulb:
            yield from combinations(range(9, 0, -1), len(self.squares))


class OldThermometerFeature(GroupedPossibilitiesFeature):
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


class SlowThermometerFeature(PossibilitiesFeature):
    def __init__(self, thermometer: Union[Sequence[Square], str], *,
                 name: Optional[str] = None, color: str = 'lightgrey'):
        super().__init__(thermometer, name=name, neighbors=True)
        self.color = color

    def draw(self, context: DrawContext) -> None:
        _draw_thermometer(self.squares, self.color, context)

    def get_possibilities(self) -> Iterable[tuple[int, ...]]:
        return combinations_with_replacement(range(1, 10), len(self.squares))
