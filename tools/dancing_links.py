import collections
import copy
import random
import sys
from collections.abc import Hashable, Iterator
from typing import Any, Callable, Generic, Optional, Sequence, TextIO, TypeVar

Row = TypeVar('Row', bound=Hashable)
Constraint = TypeVar('Constraint', bound=Hashable)


class DancingLinks(Generic[Row, Constraint]):
    row_to_constraints: dict[Row, list[Constraint]]
    optional_constraints: set[Constraint]
    inclusive_constraints: set[Constraint]
    row_printer: Any

    count: int
    max_debugging_depth: int
    output: TextIO
    debug: int
    constraint_to_rows: dict[Constraint, set[Row]]

    def __init__(self, constraints: dict[Row, list[Constraint]],
                 *, row_printer: Optional[Callable[[Sequence[Row]], None]] = None,
                 optional_constraints: Optional[set[Constraint]] = None,
                 inclusive_constraints: Optional[set[Constraint]] = None):
        """The entry to the Dancing Links code.  constraints should be a dictionary.  Each key
        is the name of the row (something meaningful to the user).  The value should
        be a list/tuple of the row_to_constraints satisfied by this row.

        The row names and constraint names can be anything immutable and hashable.
        Typically they are strings, but feel free to use whatever works best.
        """
        self.row_to_constraints = constraints
        self.optional_constraints = optional_constraints or set()
        self.inclusive_constraints = inclusive_constraints or set()
        self.row_printer = row_printer or (lambda solution: print(sorted(solution)))

    def solve(self, output: TextIO = sys.stdout, debug: Optional[int] = None,
              recursive: Optional[bool] = False) -> None:
        # Create the cross reference giving the rows in which each constraint appears
        constraint_to_rows: dict[Constraint, set[Row]] = collections.defaultdict(set)
        for row, constraints in self.row_to_constraints.items():
            for constraint in constraints:
                constraint_to_rows[constraint].add(row)

        self.optional_constraints = {x for x in self.optional_constraints if x in constraint_to_rows}
        unused_constraints = {x for x in self.optional_constraints if len(constraint_to_rows[x]) == 1}
        for constraint in unused_constraints:
            row = constraint_to_rows.pop(constraint).pop()
            self.row_to_constraints[row].remove(constraint)
            self.optional_constraints.remove(constraint)

        runner = copy.copy(self)

        runner.constraint_to_rows = constraint_to_rows
        runner.output = output
        runner.count = 0
        runner.debug = debug is not None
        runner.max_debugging_depth = debug if debug is not None else -1

        if runner.debug:
            optional_count = len(self.optional_constraints)
            required_count = len(constraint_to_rows) - optional_count
            output.write(f"There are {len(runner.row_to_constraints)} rows; "
                         f"{required_count} required constraints; "
                         f"{optional_count} optional constraints\n")

        if recursive:
            recursion_depth = len(constraint_to_rows) + 100
            if sys.getrecursionlimit() < recursion_depth:
                sys.setrecursionlimit(recursion_depth)

            solutions_count = 0
            for solution in runner.__solve_constraints_recursive(0):
                solutions_count += 1
                self.row_printer(solution)
        else:
            solutions_count = runner.__solve_constraints_iterative()

        if runner.debug:
            print("Count =", runner.count, file=output)
            print("Solutions =", solutions_count, file=output)

    def __solve_constraints_recursive(self, depth: int) -> Iterator[list[Row]]:
        """Returns a set of rows that satisfies the row_to_constraints of constraint_to_rows
        """
        # Note that "depth" is meaningful only when debugging.
        self.count += 1
        is_debugging = depth < self.max_debugging_depth

        try:
            min_count, _, min_constraint = min((len(rows), 0 * random.random(), constraint)
                                               for constraint, rows in self.constraint_to_rows.items()
                                               if constraint not in self.optional_constraints)
        except ValueError:
            if is_debugging:
                self.output.write(f"{self.__indent(depth)}✓ SOLUTION\n")
            yield []
            return

        old_depth = depth

        depth += (min_count != 1)
        if min_count == 0:
            if is_debugging:
                self.output.write(f"{self.__indent(depth)}✕ {min_constraint}\n")
            return

        # Look at each possible row that can resolve the min_constraint.
        min_constraint_rows = self.__cover_constraint(min_constraint)

        for index, row in enumerate(min_constraint_rows, start=1):
            cols = [self.__cover_constraint(row_constraint)
                    for row_constraint in self.row_to_constraints[row] if row_constraint != min_constraint]

            if is_debugging:
                self.__print_debug_info(min_constraint, row, index, min_count, old_depth)

            for solution in self.__solve_constraints_recursive(depth):
                solution.append(row)
                yield solution

            for row_constraint in reversed(self.row_to_constraints[row]):
                if row_constraint != min_constraint:
                    self.__uncover_constraint(row_constraint, cols.pop())

        self.__uncover_constraint(min_constraint, min_constraint_rows)

    def __solve_constraints_iterative(self) -> int:
        # Note that "depth" is meaningful only when debugging.
        stack: list[tuple[Callable[..., None], Sequence[Any]]] = []
        solution_count = 0

        def run() -> int:
            stack.append((find_minimum_constraint, (0,)))
            while stack:
                function, args = stack.pop()
                function(*args)
            return solution_count

        def find_minimum_constraint(depth: int) -> None:
            nonlocal solution_count
            self.count += 1
            try:
                count, _, constraint = min((len(rows), 0.0 * random.random(), constraint)
                                           for constraint, rows in self.constraint_to_rows.items()
                                           if constraint not in self.optional_constraints)
            except ValueError:
                # There is nothing left but optional constraints.  We have a solution!
                if depth < self.max_debugging_depth:
                    self.output.write(f"{self.__indent(depth)}✓ SOLUTION\n")
                # row_cleanup on the stack indicates that we are currently working on that item
                solution = [args[1] for (func, args) in stack if func == row_cleanup]
                solution_count += 1
                self.row_printer(solution)
                return

            if count > 0:
                stack.append((look_at_constraint, (constraint, depth)))
            else:
                # No rows satisfy this constraint.  Dead end.
                if depth < self.max_debugging_depth:
                    self.output.write(f"{self.__indent(depth)}✕ {constraint}\n")

        def look_at_constraint(constraint: Constraint, depth: int) -> None:
            # Look at each possible row that can resolve the constraint.
            rows = self.__cover_constraint(constraint)
            count = len(rows)

            stack.append((constraint_cleanup, (constraint, rows)))
            entries = [(look_at_row, (constraint, row, index, count, depth)) for index, row in enumerate(rows, start=1)]
            stack.extend(reversed(entries))

        def look_at_row(constraint: Constraint, row: Row, index: int, count: int, depth: int) -> None:
            cleanups = [(row_constraint, self.__cover_constraint(row_constraint))
                        for row_constraint in self.row_to_constraints[row]
                        if row_constraint != constraint]
            if depth < self.max_debugging_depth:
                self.__print_debug_info(constraint, row, index, count, depth)

            # Remember we are adding things in reverse order.  Recurse on the smaller sub-problem, and then cleanup
            # what we just did above.
            stack.append((row_cleanup, (cleanups, row)))
            stack.append((find_minimum_constraint, (depth + (count > 1),)))

        def row_cleanup(cleanups: list[tuple[Constraint, set[Row]]], _row: Row, ) -> None:
            for constraint, rows in reversed(cleanups):
                self.__uncover_constraint(constraint, rows)

        def constraint_cleanup(constraint: Constraint, rows: set[Row]) -> None:
            self.__uncover_constraint(constraint, rows)

        return run()

    def __cover_constraint(self, constraint: Constraint) -> set[Row]:
        rows = self.constraint_to_rows.pop(constraint)
        for row in rows:
            # For each constraint in this row about to be deleted
            for row_constraint in self.row_to_constraints[row]:
                # Mark this feature as now longer available in the row,
                # unless we're looking at the feature we just chose!
                if row_constraint != constraint:
                    self.constraint_to_rows[row_constraint].remove(row)
        return rows

    def __uncover_constraint(self, constraint: Constraint, rows: set[Row]) -> None:
        for row in rows:
            for row_constraint in self.row_to_constraints[row]:
                if row_constraint != constraint:
                    self.constraint_to_rows[row_constraint].add(row)
        self.constraint_to_rows[constraint] = rows

    def __print_debug_info(self, min_constraint: Constraint, row: Row, index: int, count: int, depth: int) -> None:
        indent = self.__indent(depth)
        live_rows = {x for rows in self.constraint_to_rows.values() for x in rows}
        if count == 1:
            self.output.write(f"{indent}• ")
        else:
            self.output.write(f"{indent}{index}/{count} ")
        self.output.write(f"{min_constraint}: Row {row} ({len(live_rows)} rows)\n")

    @staticmethod
    def __indent(depth: int) -> str:
        return ' | ' * depth
