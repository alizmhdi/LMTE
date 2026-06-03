#!/usr/bin/env python3
"""Generate synthetic traffic matrices for LMTE.

Outputs a CSV where each row is a flattened NxN traffic matrix (no header),
including diagonal entries.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def _read_topology(topology_path: Path):
    with topology_path.open("r") as f:
        data = json.load(f)
    graph = nx.readwrite.json_graph.node_link_graph(data)
    nodes = sorted(graph.nodes())
    return graph, nodes


def _total_directed_capacity(graph: nx.DiGraph) -> float:
    total = 0.0
    for _, _, attr in graph.edges(data=True):
        total += float(attr.get("capacity", 0.0))
    return total


def _sample_tm_gravity(n: int, rng: np.random.Generator) -> np.ndarray:
    out_strength = rng.lognormal(mean=0.0, sigma=0.7, size=n)
    in_strength = rng.lognormal(mean=0.0, sigma=0.7, size=n)
    tm = np.outer(out_strength, in_strength).astype(np.float32)
    np.fill_diagonal(tm, 0.0)
    return tm


def _sample_tm_normal(
    n: int,
    rng: np.random.Generator,
    max_demand: float,
) -> np.ndarray:
    tm = rng.uniform(0.0, max_demand, size=(n, n)).astype(np.float32)
    np.fill_diagonal(tm, 0.0)
    return tm


def _sample_tm_gravity_capacity(
    graph: nx.DiGraph,
    nodes: list,
    rng: np.random.Generator,
    max_demand: float = 500.0,
    random: bool = True,
) -> np.ndarray:
    """Gravity TM based on per-node in/out capacity fractions within each SCC.

    For each SCC the outgoing fraction of node u is proportional to its total
    outgoing capacity, and the incoming fraction of node v is proportional to
    its total incoming capacity (excluding u's own share from the denominator).
    When *random* is True the entry is sampled from N(frac, frac/4) and clipped
    to non-negative values; otherwise the deterministic fraction is used.
    The matrix is scaled by *max_demand* and clipped so no entry exceeds it.
    """
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    tm = np.zeros((n, n), dtype=np.float32)

    for scc in nx.strongly_connected_components(graph):
        in_cap_sum: dict = defaultdict(float)
        out_cap_sum: dict = defaultdict(float)
        for u in scc:
            for v in graph.predecessors(u):
                in_cap_sum[u] += float(graph[v][u].get("capacity", 0.0))
            for v in graph.successors(u):
                out_cap_sum[u] += float(graph[u][v].get("capacity", 0.0))

        in_total_cap = sum(in_cap_sum.values())
        out_total_cap = sum(out_cap_sum.values())
        if out_total_cap == 0.0:
            continue

        for u in scc:
            norm_u = out_cap_sum.get(u, 0.0) / out_total_cap
            for v in scc:
                if u == v:
                    continue
                denom = in_total_cap - in_cap_sum.get(u, 0.0)
                if denom == 0.0:
                    continue
                frac = norm_u * in_cap_sum.get(v, 0.0) / denom
                if random:
                    val = float(rng.normal(frac, frac / 4)) if frac > 0.0 else 0.0
                    tm[node_idx[u], node_idx[v]] = max(val, 0.0)
                else:
                    tm[node_idx[u], node_idx[v]] = frac

    tm_max = tm.max()
    if tm_max > 0.0:
        tm = tm / tm_max * max_demand
    np.clip(tm, 0.0, max_demand, out=tm)
    return tm

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TM CSV for LMTE")
    parser.add_argument("--topology", type=Path, required=True, help="Path to topology.json")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--num-samples", type=int, default=48384, help="Number of TM snapshots")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gravity", "normal", "gravity_capacity"],
        default="gravity",
        help="Traffic generation model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--normal-max-demand",
        type=float,
        default=500.0,
        help="Maximum per-entry demand for normal model after clipping",
    )
    parser.add_argument(
        "--gravity-max-demand",
        type=float,
        default=500.0,
        help="Maximum per-entry demand for gravity_capacity model",
    )
    parser.add_argument(
        "--gravity-no-random",
        action="store_true",
        default=False,
        help="Use deterministic gravity_capacity TMs instead of Gaussian-sampled ones",
    )

    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.normal_max_demand <= 0:
        raise ValueError("--normal-max-demand must be > 0")

    graph, nodes = _read_topology(args.topology)
    n = len(nodes)
    total_capacity = _total_directed_capacity(graph)

    rng = np.random.default_rng(args.seed)
    rows = np.zeros((args.num_samples, n * n), dtype=np.float32)

    for t in range(args.num_samples):
        if args.model == "gravity":
            tm = _sample_tm_gravity(n, rng)
            tm = np.clip(tm, 0.0, args.normal_max_demand)
        elif args.model == "normal":
            tm = _sample_tm_normal(n, rng, args.normal_max_demand)
            tm = np.clip(tm, 0.0, args.normal_max_demand)
        else:  # gravity_capacity
            tm = _sample_tm_gravity_capacity(
                graph,
                nodes,
                rng,
                max_demand=args.gravity_max_demand,
                random=not args.gravity_no_random,
            )
        rows[t] = tm.reshape(-1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, header=False, index=False)

    print(f"Saved: {args.output}")
    print(f"Shape: {rows.shape}")
    print(f"Nodes: {n}")
    print(f"Total directed capacity: {total_capacity}")
    print(f"Model: {args.model}")


if __name__ == "__main__":
    main()
