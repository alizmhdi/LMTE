#!/usr/bin/env python3
"""Generate synthetic traffic matrices for LMTE.

Outputs a CSV where each row is a flattened NxN traffic matrix (no header),
including diagonal entries.
"""

import argparse
import json
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


def _rescale_tm_to_load(tm: np.ndarray, target_total: float) -> np.ndarray:
    current = float(tm.sum())
    if current <= 0:
        return tm
    return tm * (target_total / current)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TM CSV for LMTE")
    parser.add_argument("--topology", type=Path, required=True, help="Path to topology.json")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--num-samples", type=int, default=48384, help="Number of TM snapshots")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gravity", "normal"],
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
        else:
            tm = _sample_tm_normal(
                n,
                rng,
                args.normal_max_demand,
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
