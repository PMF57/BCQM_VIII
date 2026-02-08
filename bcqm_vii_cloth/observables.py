\
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any
import math
import numpy as np

from .graph_store import GraphStore


def weighted_junction_stat(indegs: List[int], beta: float) -> float:
    if not indegs:
        return 0.0
    s = 0.0
    for k in indegs:
        if k < 2:
            continue
        if k == 2:
            s += 1.0
        else:
            s += float(k) ** float(beta)
    return s / float(len(indegs))


def hubshare(indegs: List[int]) -> float:
    if not indegs:
        return 0.0
    tot = float(sum(indegs))
    if tot <= 0:
        return 0.0
    return float(max(indegs)) / tot


def clustering_coefficient_undirected(adj: Dict[int, Set[int]], sample: int | None = None) -> float:
    nodes = list(adj.keys())
    if not nodes:
        return 0.0
    if sample is not None and len(nodes) > sample:
        rng = np.random.default_rng(0)
        nodes = list(rng.choice(nodes, size=sample, replace=False))
    coeffs: List[float] = []
    for v in nodes:
        nbrs = list(adj.get(v, set()))
        k = len(nbrs)
        if k < 2:
            continue
        links = 0
        for i in range(k):
            ni = nbrs[i]
            ai = adj.get(ni, set())
            for j in range(i + 1, k):
                if nbrs[j] in ai:
                    links += 1
        coeffs.append((2.0 * links) / (k * (k - 1)))
    if not coeffs:
        return 0.0
    return float(sum(coeffs)) / float(len(coeffs))


def induced_active_set(cfg: Dict[str, Any], g: GraphStore, frontier: List[int], epoch: int) -> List[int]:
    aw = cfg["active_window"]
    mode = aw["mode"]
    if mode == "recency":
        hops = int(aw["hops"])
        cutoff = max(0, epoch - hops)
        active = [nid for nid, nd in g.nodes.items() if nd.created_at >= cutoff]
        aset = set(active)
        aset.update(frontier)
        return list(aset)

    d_h = int(aw["d_horizon"])
    adj: Dict[int, Set[int]] = {}
    for u, v, w, t in g.edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    seen: Set[int] = set(frontier)
    q: List[Tuple[int, int]] = [(f, 0) for f in frontier]
    while q:
        node, dist = q.pop(0)
        if dist >= d_h:
            continue
        for nb in adj.get(node, set()):
            if nb not in seen:
                seen.add(nb)
                q.append((nb, dist + 1))
    return list(seen)


def s_perc(g: GraphStore, active: List[int]) -> float:
    if not active:
        return 0.0
    aset = set(active)
    adj: Dict[int, Set[int]] = {v: set() for v in aset}
    for u, v, w, t in g.edges:
        if u in aset and v in aset:
            adj[u].add(v)
            adj[v].add(u)

    seen: Set[int] = set()
    best = 0
    for v in aset:
        if v in seen:
            continue
        stack = [v]
        seen.add(v)
        size = 0
        while stack:
            x = stack.pop()
            size += 1
            for nb in adj.get(x, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        best = max(best, size)
    return float(best) / float(len(aset))


def compute_Q_clock(tick_epochs: List[int]) -> float:
    if len(tick_epochs) < 3:
        return 0.0
    deltas = np.diff(np.array(tick_epochs, dtype=float))
    if deltas.size == 0:
        return 0.0
    mu = float(np.mean(deltas))
    sd = float(np.std(deltas))
    if sd <= 1e-12:
        return float("inf") if mu > 0 else 0.0
    return mu / sd
