from collections import Hashable, Iterable
from typing import TypeVar, Generic

Node = TypeVar('Node', bound=Hashable)


class UnionFind(Generic[Node]):
    parent: dict[Node, Node]
    rank: dict[Node, int]

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def all_nodes(self) -> Iterable[Node]:
        return self.parent.keys()

    def find(self, x: Node):
        parent = self.parent
        parent.setdefault(x, x)
        root = x
        while (next_node := parent[root]) != root:
            root = next_node
        while (next_node := parent[x]) != root:
            x, parent[x] = next_node, root
        return root

    def union(self, x: Node, y: Node):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            rank_x = self.rank.setdefault(x, 1)
            rank_y = self.rank.setdefault(y, 1)
            if rank_x > rank_y:
                # We dont' swap rank_x and rank_y because after this, we only care if they're equal
                x, y = y, x
            self.parent[x] = y
            if rank_x == rank_y:
                self.rank[y] = rank_y + 1
