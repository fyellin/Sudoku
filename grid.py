from __future__ import annotations

from collections import UserDict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from cell import Cell, House

if TYPE_CHECKING:
    from feature import Feature
    import features.same_value_feature as svf

class Grid(UserDict):
    matrix: dict[tuple[int, int], Cell]
    houses: list[House]
    features: list[Feature]
    neighborly_features: Sequence[Feature]
    pair_features: Sequence[Feature]
    has_alternative_boxes: bool
    same_value_handler: 'svf.SameValueHandler'

    def __init__(self, features: list[Feature]) -> None:
        super().__init__()

        import features.same_value_feature as svf
        same_value_handler = svf.SameValueHandler()
        features.append(same_value_handler)

        # Features that affect neighbors of a cell
        self.neighborly_features = [feature for feature in features if feature.has_neighbor_method()]
        # Features that have strong/weak/chain pairs
        self.pair_features = [feature for feature in features if feature.has_any_pair_method()]
        self.matrix = {(row, column): Cell(row, column, self)
                       for row in range(1, 10) for column in range(1, 10)}
        self.features = features
        self.same_value_handler = same_value_handler

        def items_in_row(row: int) -> Sequence[Cell]:
            return [self.matrix[row, column] for column in range(1, 10)]

        def items_in_column(column: int) -> Sequence[Cell]:
            return [self.matrix[row, column] for row in range(1, 10)]

        def items_in_box(box: int) -> Sequence[Cell]:
            q, r = divmod(box - 1, 3)
            return [self.matrix[row, column]
                    for row in range(3 * q + 1, 3 * q + 4)
                    for column in range(3 * r + 1, 3 * r + 4)]

        rows = [House(House.Type.ROW, row, items_in_row(row)) for row in range(1, 10)]
        columns = [House(House.Type.COLUMN, column, items_in_column(column)) for column in range(1, 10)]
        boxes = [House(House.Type.BOX, box, items_in_box(box)) for box in range(1, 10)]

        self.houses = [*rows, *columns, *boxes]

        for feature in features:
            feature.initialize(self)

        for cell in self.matrix.values():
            cell.initialize_neighbors(self)

    def start(self) -> None:
        for cell in self.cells:
            cell.start()
        for house in self.houses:
            house.start()
        for feature in self.features:
            feature.start()

    def is_solved(self) -> bool:
        return all(cell.is_known for cell in self.cells)

    def delete_normal_boxes(self) -> None:
        boxes = [house for house in self.houses if house.house_type == House.Type.BOX]
        for box in boxes:
            self.delete_house(box)

    def delete_house(self, house: House) -> None:
        self.houses.remove(house)
        for cell in house.cells:
            cell.houses.remove(house)

    @property
    def cells(self) -> Iterable[Cell]:
        return self.matrix.values()

    def print(self, marks: bool = True) -> None:
        import sys
        out = sys.stdout
        matrix = self.matrix
        max_length = max(len(cell.possible_values) for cell in self.cells)
        is_solved = max_length == 1
        max_length = 1 if is_solved or not marks else max(max_length, 3)
        for row in range(1, 10):
            for column in range(1, 10):
                cell = matrix[row, column]
                if max_length == 1:
                    if cell.is_known:
                        out.write(f'{cell.known_value}')
                    else:
                        out.write('*')
                elif cell.is_known:
                    string = f'*{cell.known_value}*'
                    out.write(string.center(max_length, ' '))
                else:
                    string = ''.join(str(i) for i in sorted(cell.possible_values))
                    out.write(string.center(max_length, ' '))
                out.write(' | ' if column == 3 or column == 6 else ' ')
            out.write('\n')
            if row == 3 or row == 6:
                out.write('-' * (3 * max_length + 2))
                out.write('-+-')
                out.write('-' * (3 * max_length + 2))
                out.write('-+-')
                out.write('-' * (3 * max_length + 2))
                out.write('\n')
