\
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from .graph_store import GraphStore
from .io import write_json


def write_edges_csv(path: Path, edges: List[tuple[int, int, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "w", "epoch"])
        for u, v, wt, ep in edges:
            w.writerow([u, v, wt, ep])


def write_nodes_json(path: Path, g: GraphStore) -> None:
    obj: Dict[str, Any] = {
        "nodes": {
            str(nid): {
                "created_at": nd.created_at,
                "indeg": nd.indeg,
                "outdeg": nd.outdeg,
            }
            for nid, nd in g.nodes.items()
        }
    }
    write_json(path, obj)
