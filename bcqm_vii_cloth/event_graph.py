from __future__ import annotations

"""
event_graph.py (BCQM VI)

Event graph bookkeeping + spatial observables + spectral-dimension estimator.

Notes:
- Geometry is computed on the largest weakly connected component of V_active.
- Spectral dimension is estimated from the random-walk return probability P0(t) ~ t^{-ds/2}.
- We log diagnostics: component size, fit window, slope, r2, ds_valid flags, and a downsampled P0(t) curve.

This module is designed to remain compatible with Path A usage:
- new_event, add_edge, v_active, s_perc, s_junc_w, hubshare, max_indegree, clustering_coeff
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import math
import random


@dataclass
class EventNode:
    created_at: int
    domain: Optional[int] = None
    indeg: int = 0
    outdeg: int = 0


class EventGraph:
    def __init__(self) -> None:
        self.nodes: Dict[int, EventNode] = {}
        self.edges: List[Tuple[int, int, int]] = []  # (u,v,t)
        self._next_id: int = 0

    def new_event(self, t: int, domain: Optional[int] = None) -> int:
        eid = self._next_id
        self._next_id += 1
        self.nodes[eid] = EventNode(created_at=int(t), domain=domain, indeg=0, outdeg=0)
        return eid

    def add_edge(self, u: int, v: int, t: int) -> None:
        self.edges.append((int(u), int(v), int(t)))
        self.nodes[u].outdeg += 1
        self.nodes[v].indeg += 1

    def v_active(self, t: int, W_coh: int, frontiers: List[int]) -> Set[int]:
        cutoff = int(t) - int(W_coh)
        active = {eid for eid, nd in self.nodes.items() if nd.created_at >= cutoff}
        active.update(int(e) for e in frontiers)
        return active

    def indegrees(self, active: Set[int]) -> List[int]:
        return [self.nodes[e].indeg for e in active]

    def max_indegree(self, active: Set[int]) -> int:
        return max((self.nodes[e].indeg for e in active), default=0)

    def hubshare(self, active: Set[int]) -> float:
        indegs = self.indegrees(active)
        if not indegs:
            return 0.0
        s = sum(indegs)
        return (max(indegs) / float(s)) if s > 0 else 0.0

    def s_junc_w(self, active: Set[int], beta: float = 1.5) -> float:
        if not active:
            return 0.0
        s = 0.0
        for e in active:
            k = self.nodes[e].indeg
            if k < 2:
                continue
            s += 1.0 if k == 2 else float(k) ** float(beta)
        return s / float(len(active))

    def _adj_undirected(self, active: Set[int]) -> Dict[int, Set[int]]:
        adj: Dict[int, Set[int]] = {e: set() for e in active}
        for u, v, t in self.edges:
            if u in active and v in active:
                adj[u].add(v)
                adj[v].add(u)
        return adj

    def s_perc(self, active: Set[int]) -> float:
        if not active:
            return 0.0
        adj = self._adj_undirected(active)
        seen: Set[int] = set()
        best = 0
        for e in active:
            if e in seen:
                continue
            stack = [e]
            seen.add(e)
            size = 0
            while stack:
                x = stack.pop()
                size += 1
                for nb in adj.get(x, set()):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            best = max(best, size)
        return best / float(len(active))

    def clustering_coeff(self, active: Set[int], sample: Optional[int] = 500) -> float:
        if not active:
            return 0.0
        adj = self._adj_undirected(active)
        nodes = list(active)
        if sample is not None and len(nodes) > sample:
            step = max(1, len(nodes) // sample)
            nodes = nodes[::step][:sample]
        coeffs: List[float] = []
        for v in nodes:
            nbrs = list(adj.get(v, set()))
            k = len(nbrs)
            if k < 2:
                continue
            links = 0
            for i in range(k):
                ai = adj.get(nbrs[i], set())
                for j in range(i + 1, k):
                    if nbrs[j] in ai:
                        links += 1
            coeffs.append((2.0 * links) / (k * (k - 1)))
        return sum(coeffs) / len(coeffs) if coeffs else 0.0

    def largest_component_nodes(self, active: Set[int]) -> Set[int]:
        """Largest weakly connected component in the undirected projection."""
        if not active:
            return set()
        adj = self._adj_undirected(active)
        seen: Set[int] = set()
        best_comp: Set[int] = set()
        for e in active:
            if e in seen:
                continue
            stack = [e]
            seen.add(e)
            comp: Set[int] = set([e])
            while stack:
                x = stack.pop()
                for nb in adj.get(x, set()):
                    if nb not in seen:
                        seen.add(nb)
                        comp.add(nb)
                        stack.append(nb)
            if len(comp) > len(best_comp):
                best_comp = comp
        return best_comp
    def ball_growth_profile(
        self,
        active: Set[int],
        r_max: int = 25,
        samples: int = 25,
        seed: int = 0,
    ) -> Dict[str, object]:
        """Estimate mean ball volume |B(r)| vs r on the largest component of active (undirected)."""
        rng = random.Random(int(seed))
        comp = self.largest_component_nodes(active)
        comp_size = len(comp)
        if comp_size == 0:
            return {"comp_size": 0, "r_max": int(r_max), "samples": 0, "mean_ball": []}

        adj = self._adj_undirected(comp)
        nodes = list(comp)
        k = min(int(samples), len(nodes))
        roots = [nodes[rng.randrange(len(nodes))] for _ in range(k)]

        mean_ball = [0.0] * (int(r_max) + 1)
        for root in roots:
            dist = {root: 0}
            frontier = [root]
            while frontier:
                v = frontier.pop()
                dv = dist[v]
                if dv >= r_max:
                    continue
                for nb in adj.get(v, ()):
                    if nb not in dist:
                        dist[nb] = dv + 1
                        if dist[nb] <= r_max:
                            frontier.append(nb)

            counts = [0] * (int(r_max) + 1)
            for d in dist.values():
                if d <= r_max:
                    counts[d] += 1

            cum = 0
            for r in range(r_max + 1):
                cum += counts[r]
                mean_ball[r] += cum

        mean_ball = [x / float(k) for x in mean_ball]
        return {"comp_size": comp_size, "r_max": int(r_max), "samples": int(k), "mean_ball": mean_ball}
    @staticmethod
    def _linear_fit_loglog(ts: List[int], ys: List[float]) -> Tuple[float, float, float]:
        """Fit log y = a + b log t; returns (b, a, r2)."""
        import math
        lx = [math.log(float(t)) for t in ts]
        ly = [math.log(float(y)) for y in ys]
        n = len(lx)
        mx = sum(lx)/n
        my = sum(ly)/n
        sxx = sum((x-mx)**2 for x in lx)
        sxy = sum((x-mx)*(y-my) for x,y in zip(lx,ly))
        b = sxy / sxx if sxx>0 else 0.0
        a = my - b*mx
        # r2
        ss_tot = sum((y-my)**2 for y in ly)
        ss_res = sum((y-(a+b*x))**2 for x,y in zip(lx,ly))
        r2 = 1.0 - (ss_res/ss_tot) if ss_tot>0 else 0.0
        return b,a,r2

    def estimate_spectral_dimension(
        self,
        active: Set[int],
        walkers: int = 200,
        steps: int = 4000,
        fit_t_min: int = 30,
        fit_t_max: int = 400,
        seed: int = 0,
        t_max: int | None = None,
        **kwargs,
    ) -> Dict[str, object]:
        """
        Return spectral dimension estimate on largest component of active set.

        We simulate 'walkers' random walks of length 'steps' starting at random nodes.
        P0(t) = Pr(X_t = X_0) estimated by fraction of walkers returned at time t.

        Fit ds from P0(t) ~ t^{-ds/2}.
        """
        rng = random.Random(int(seed))
        comp = self.largest_component_nodes(active)
        comp_size = len(comp)
        out: Dict[str, object] = {
            "comp_size": comp_size,
            "ds_est": None,
            "ds_valid": False,
            "fit_t_min": int(fit_t_min),
            "fit_t_max": int(fit_t_max),
            "slope": None,
            "r2": None,
            "notes": "",
        }

        # Compatibility: accept t_max as an alias for fit_t_max when provided.
        if t_max is not None:
            fit_t_max = int(t_max)
        if comp_size < 30:
            out["notes"] = "component_too_small"
            return out

        adj = self._adj_undirected(comp)
        nodes = list(comp)

        # Adaptive t_max to avoid deep finite-size saturation
        tmax = int(min(fit_t_max, max(60, comp_size // 2), steps))
        tmin = int(min(fit_t_min, max(10, tmax // 3)))
        if tmax <= tmin + 5:
            out["notes"] = "fit_window_too_small"
            out["fit_t_min"] = tmin
            out["fit_t_max"] = tmax
            return out
        out["fit_t_min"] = tmin
        out["fit_t_max"] = tmax

        # Preselect start nodes
        starts = [nodes[rng.randrange(comp_size)] for _ in range(int(walkers))]
        pos = list(starts)

        # Track returns for times up to tmax
        returns = [0]*(tmax+1)
        returns[0] = int(walkers)

        for t in range(1, tmax+1):
            # one step
            for i in range(int(walkers)):
                v = pos[i]
                nbrs = list(adj.get(v, ()))
                if not nbrs:
                    # stuck: stay
                    nxt = v
                else:
                    nxt = nbrs[rng.randrange(len(nbrs))]
                pos[i] = nxt
            # count returns
            ret = sum(1 for i in range(int(walkers)) if pos[i] == starts[i])
            returns[t] = ret

        P0 = [returns[t]/float(walkers) for t in range(tmax+1)]

        # Choose fit points where P0>0 and t in [tmin,tmax]
        ts = []
        ys = []
        for t in range(tmin, tmax+1):
            if P0[t] > 0:
                ts.append(t)
                ys.append(P0[t])
        if len(ts) < 8:
            out["notes"] = "insufficient_positive_returns"
            out["P0_downsample"] = [(t, P0[t]) for t in range(0, tmax+1, max(1, tmax//40))]
            # Plateau estimate (mixing): mean return prob over last 10% of t.
            tail_start = max(1, int(0.9 * tmax))
            out["plateau_est"] = float(sum(P0[tail_start: tmax+1]) / max(1, (tmax+1 - tail_start)))
            return out

        b,a,r2 = self._linear_fit_loglog(ts, ys)
        ds = -2.0 * b
        out["slope"] = float(b)
        out["r2"] = float(r2)
        out["P0_downsample"] = [(t, P0[t]) for t in range(0, tmax+1, max(1, tmax//40))]

        # Plateau estimate (mixing): mean return prob over last 10% of t.
        tail_start = max(1, int(0.9 * tmax))
        out["plateau_est"] = float(sum(P0[tail_start: tmax+1]) / max(1, (tmax+1 - tail_start)))
        # Validity checks
        if (ds <= 0) or (ds > 10) or (r2 < 0.85):
            out["notes"] = "fit_invalid"
            out["ds_est"] = float(ds)
            out["ds_valid"] = False
            return out

        out["ds_est"] = float(ds)
        out["ds_valid"] = True
        out["notes"] = "ok"
        return out
