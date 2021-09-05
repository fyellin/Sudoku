import collections
import operator
from itertools import chain, combinations, count, filterfalse, groupby, islice, repeat, tee, zip_longest
from typing import Callable, Iterable, Optional, TypeVar

T = TypeVar('T')


def take(n: int, iterable: Iterable[T]) -> list[T]:
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))


def prepend(value: T, iterator: Iterable[T]) -> Iterable[T]:
    """Prepend a single value in front of an iterator"""
    # prepend(1, [2, 3, 4]) -> 1 2 3 4
    return chain([value], iterator)


def tabulate(function:Callable[[int], T], start: int = 0) -> Iterable[T]:
    """Return function(0), function(1), ..."""
    return map(function, count(start))


def tail(n: int, iterable: Iterable[T]) -> Iterable[T]:
    """Return an iterator over the last n items"""
    # tail(3, 'ABCDEFG') --> E F G
    return iter(collections.deque(iterable, maxlen=n))


def consume(iterator: Iterable[T], n: Optional[int] = None) -> None:
    """Advance the iterator n-steps ahead. If n is None, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def nth(iterable: Iterable[T], n: int, default:Optional[T] = None) -> Optional[T]:
    """Returns the nth item or a default value"""
    return next(islice(iterable, n, None), default)


def all_equal(iterable: Iterable[T]) -> bool:
    """Returns True if all the elements are equal to each other"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def quantify(iterable: Iterable[T], pred: Callable[[T], bool] = bool) -> int:
    """Count how many times the predicate is true"""
    return sum(map(pred, iterable))


def pad_none(iterable: Iterable[T]) -> Iterable[Optional[T]]:
    """Returns the sequence elements and then returns None indefinitely.

    Useful for emulating the behavior of the built-in map() function.
    """
    return chain(iterable, repeat(None))


def ncycles(iterable: Iterable[T], n: int) -> Iterable[T]:
    """Returns the sequence elements n times"""
    return chain.from_iterable(repeat(tuple(iterable), n))


def dotproduct(vec1: Iterable[float], vec2: Iterable[float]) -> float:
    return sum(map(operator.mul, vec1, vec2))


def flatten(list_of_lists: list[list[T]]) -> Iterable[T]:
    """Flatten one level of nesting"""
    return chain.from_iterable(list_of_lists)


def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(iterable: Iterable[T], n: int, fillvalue=None) -> Iterable[tuple[T, ...]]:
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def partition(pred: Callable[[T], bool], iterable: Iterable[T]) -> tuple[Iterable[T], Iterable[T]]:
    """Use a predicate to partition entries into false entries and true entries"""
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def powerset(iterable: Iterable[T]) -> Iterable[tuple[T, ...]]:
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def unique_everseen(iterable: Iterable[T], key=None) -> Iterable[T]:
    """List unique elements, preserving order. Remember all elements ever seen."""
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(iterable: Iterable[T], key=None) -> Iterable[T]:
    """List unique elements, preserving order. Remember only the element just seen."""
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))


def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


# Added by FY.  This is my version of something in more_iterables
def one(iterable: Iterable[T]) -> Optional[T]:
    it = iter(iterable)
    on_error = None
    try:
        on_error = next(it)
        next(it)
        return None
    except StopIteration:
        return on_error
