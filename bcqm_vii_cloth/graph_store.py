\
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Node:
    created_at: int
    indeg: int = 0
    outdeg: int = 0


class GraphStore:
    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Tuple[int, int, float, int]] = []  # (u,v,w,epoch)
        self._next_id: int = 0

    def new_node(self, created_at: int) -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(created_at=created_at)
        return nid

    def add_edge(self, u: int, v: int, w: float, epoch: int) -> None:
        self.edges.append((u, v, float(w), int(epoch)))
        self.nodes[u].outdeg += 1
        self.nodes[v].indeg += 1
