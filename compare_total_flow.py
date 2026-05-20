#!/usr/bin/env python
"""
Compare total routed traffic: LMTE model vs. LP-optimal routing.

The script loads the same test split (70/10/20) used by run_baseline.py,
runs both the LMTE model and the LP-optimal solver on every test TM, and
prints a side-by-side summary.

Run from the LMTE/ root directory:

    python compare_total_flow.py \
        --checkpoint checkpoints-1/Abilene_lmte_0/checkpoint.pt \
        --topology_filepath data/Abilene/topology.json \
        --tm_filepath data/Abilene/Abilene_normal.csv
"""

import os
import sys
import argparse

import numpy as np
import torch
from tqdm import tqdm

# Make LMTE's src/ package and cl_baselines/ importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_CL_BASELINES = os.path.join(_HERE, 'cl_baselines')
if _CL_BASELINES not in sys.path:
    sys.path.insert(0, _CL_BASELINES)

from src.build_dataloader import build_dataloader           # noqa: E402
from src.model import LmteModel                             # noqa: E402
from src.objective import compute_total_flows               # noqa: E402
from src.utils.read_data import read_graph_from_json, read_paths_from_file   # noqa: E402
from src.utils.useful_functions import (                    # noqa: E402
    compute_ksp_paths,
    get_paths_to_edges,
    get_commodities_to_paths,
    get_capacities_from_graph,
)
from useful_functions import Get_edge_to_path, expand_tms   # noqa: E402
from optimal_total_flow import TotalFlowOptimal             # noqa: E402


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare LMTE vs. LP-optimal total-flow on the test set'
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to LMTE checkpoint.pt file')
    p.add_argument('--topology_filepath', type=str,
                   default='./data/Abilene/topology.json')
    p.add_argument('--tm_filepath', type=str,
                   default='./data/Abilene/Abilene_normal.csv')
    p.add_argument('--topology', type=str, default='Abilene',
                   help='Topology name (used to locate paths.txt)')
    p.add_argument('--num_paths', type=int, default=8,
                   help='K-shortest paths per OD pair (must match checkpoint)')
    p.add_argument('--window_size', type=int, default=12,
                   help='History window length (must match checkpoint)')
    p.add_argument('--scale', type=float, default=1e9,
                   help='Normalisation scale (must match checkpoint training)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--eval_batch_size', type=int, default=1)
    # Model architecture – must match the checkpoint.
    p.add_argument('--d_model', type=int, default=32)
    p.add_argument('--d_keys', type=int, default=32)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--llm_dim', type=int, default=4096)
    p.add_argument('--llm_layers', type=int, default=8)
    p.add_argument('--num_gnn_layers', type=int, default=3)
    p.add_argument('--num_rnn_layers', type=int, default=2)
    p.add_argument('--num_dnn_layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--llm_model', type=str, default='llama-8b')
    p.add_argument('--use_divide_head', action='store_true')
    p.add_argument('--device', type=str, default=None,
                   help='Torch device string, e.g. "cuda:0" or "cpu"')
    p.add_argument('--seed', type=int, default=2025)
    return p.parse_args()


# ── topology helper functions ─────────────────────────────────────────────────

def _build_edge_index(topology):
    node_idx = {n: i for i, n in enumerate(sorted(topology.nodes()))}
    src_list, dst_list = [], []
    for u, v in topology.edges():
        src_list.append(node_idx[u])
        dst_list.append(node_idx[v])
    return torch.tensor([src_list, dst_list], dtype=torch.int64)


def _build_node_features(topology, scale):
    nodes_ordered = sorted(topology.nodes())
    degrees = torch.tensor(
        [topology.in_degree(n) for n in nodes_ordered], dtype=torch.float32
    ).reshape(-1, 1)
    cap_sums = [
        sum(topology[n][v].get('capacity', 0.0) for v in topology.successors(n)) / scale
        for n in nodes_ordered
    ]
    return torch.cat(
        [degrees, torch.tensor(cap_sums, dtype=torch.float32).reshape(-1, 1)], dim=1
    )


def _build_edge_ids_per_path(paths_converted, topology):
    """Build padded [total_paths, max_path_len] edge-ID tensor.
    Expects paths_converted in edge-tuple format, same as lmte_solver.py."""
    edges_map = {(u, v): eid for eid, (u, v) in enumerate(topology.edges())}
    paths_edges_list = []
    for key in sorted(paths_converted.keys()):
        for path in paths_converted[key]:
            edge_ids = [edges_map[e] for e in path]
            paths_edges_list.append(torch.tensor(edge_ids, dtype=torch.int32))
    padded = torch.nn.utils.rnn.pad_sequence(
        paths_edges_list, batch_first=True, padding_value=-1
    )
    return padded.to(dtype=torch.int64)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (
        torch.device(args.device) if args.device
        else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )
    print(f'Device: {device}')

    # ── topology & k-shortest paths ───────────────────────────────────────
    topo, _ = read_graph_from_json(args.topology_filepath)
    num_nodes = len(topo.nodes())
    paths_dir = f'./data/{args.topology}'

    try:
        # Node-ID format: used by TotalFlowOptimal and Get_edge_to_path.
        te_paths = read_paths_from_file(
            filepath=f'{paths_dir}/paths.txt', num_nodes=num_nodes
        )
    except FileNotFoundError:
        pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        te_paths = compute_ksp_paths(
            k=args.num_paths, pairs=pairs, graph=topo,
            save2txt=True, filepath=paths_dir, transform=True,
        )

    # Edge-tuple format: used by matrix builders and LMTE model.
    paths_converted = read_paths_from_file(
        filepath=f'{paths_dir}/paths.txt', num_nodes=num_nodes, convert=True
    )

    # ── sparse matrices (LMTE objective evaluation) ───────────────────────
    p_matrix = get_paths_to_edges(topo, paths=paths_converted)
    pm_coo = p_matrix.tocoo()
    paths_to_edges = torch.sparse_coo_tensor(
        np.vstack((pm_coo.row, pm_coo.col)),
        torch.FloatTensor(pm_coo.data),
        torch.Size(pm_coo.shape),
    ).to(device)

    c_matrix = get_commodities_to_paths(topo, num_paths=p_matrix.shape[0], paths=paths_converted)
    cm_coo = c_matrix.tocoo()
    commodities_to_paths = torch.sparse_coo_tensor(
        np.vstack((cm_coo.row, cm_coo.col)),
        torch.FloatTensor(cm_coo.data),
        torch.Size(cm_coo.shape),
    ).to(device)

    # ── static tensors for LMTE model forward pass ────────────────────────
    edge_index       = _build_edge_index(topo).to(device)
    raw_caps         = torch.tensor(get_capacities_from_graph(topo), dtype=torch.float32)
    capacities       = (raw_caps / args.scale).to(device)
    node_features    = _build_node_features(topo, args.scale).to(device)
    edge_ids_per_path = _build_edge_ids_per_path(paths_converted, topo).to(device)
    max_path_length  = int(edge_ids_per_path.shape[-1])

    # ── build & load LMTE model ───────────────────────────────────────────
    print(f'Building LmteModel …')
    model = LmteModel(
        d_model=args.d_model,
        d_llm=args.llm_dim,
        num_nodes=num_nodes,
        num_paths=args.num_paths,
        num_eages=topo.number_of_edges(),
        num_gnn_layers=args.num_gnn_layers,
        num_rnn_layers=args.num_rnn_layers,
        num_dense_layers=args.num_dnn_layers,
        d_keys=args.d_keys,
        seq_len=args.window_size,
        n_heads=args.n_heads,
        dropout=args.dropout,
        d_middle=args.d_model,
        llm_layers=args.llm_layers,
        llm_model=args.llm_model,
        max_length=max_path_length,
        use_divide_head=args.use_divide_head,
    ).float()

    print(f'Loading checkpoint: {args.checkpoint}')
    state_dict = torch.load(args.checkpoint, map_location=device)
    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    for name, param in state_dict.items():
        key = name.replace('module.', '')
        if key in model_sd:
            model_sd[key].copy_(param)
            loaded += 1
        else:
            skipped += 1
    if skipped:
        print(f'  checkpoint: loaded {loaded} params, skipped {skipped} unrecognised keys.')
    model.eval().to(device)

    # ── load test data ────────────────────────────────────────────────────
    _, _, test_loader = build_dataloader(
        args.topology_filepath, args.tm_filepath,
        args.batch_size, args.scale, args.eval_batch_size, args.window_size,
        split_ratio=(0.7, 0.1, 0.2),
    )
    # histories: (N_test, window_size, num_commodities) — normalised
    # targets:   (N_test, num_commodities)               — normalised
    histories = test_loader.dataset.tm_seqences.float()
    targets   = test_loader.dataset.tm_preds.float()
    n_test    = len(targets)
    print(f'Test set: {n_test} samples')

    # Pre-expand targets to (N_test, N*N) in raw capacity units for the LP.
    raw_tms = expand_tms(targets.cpu().numpy(), num_nodes) * args.scale  # (N_test, N*N)

    # ── LP-optimal solver ─────────────────────────────────────────────────
    edge_to_path = Get_edge_to_path(topo, te_paths)
    lp_solver    = TotalFlowOptimal(topo, te_paths, edge_to_path)

    # ── inference loop ────────────────────────────────────────────────────
    lmte_flows = []
    lp_flows   = []

    for i in tqdm(range(n_test), desc='Evaluating'):
        # ── LMTE ──────────────────────────────────────────────────────────
        tm_window = histories[i].unsqueeze(0).to(device)  # (1, window, C)
        tm_tensor = targets[i].unsqueeze(0).to(device)    # (1, C)

        with torch.no_grad():
            split_ratios = model(
                tm_window, node_features, edge_index, capacities,
                topo, None, edge_ids_per_path,
            )  # (1, total_paths)

        # compute_total_flows returns normalised flow; multiply by scale → raw units.
        lmte_total = compute_total_flows(
            split_ratios, tm_tensor, topo,
            paths_to_edges, commodities_to_paths,
            scale=args.scale, normalize_by_demand=False,
        )[0] * args.scale

        # ── LP optimal ────────────────────────────────────────────────────
        lp_total, _ = lp_solver.maximize_total_flow(raw_tms[i])

        lmte_flows.append(lmte_total)
        lp_flows.append(lp_total if lp_total is not None else float('nan'))

    # ── summary ───────────────────────────────────────────────────────────
    lmte_arr = np.array(lmte_flows)
    lp_arr   = np.array(lp_flows)
    valid    = ~np.isnan(lp_arr)
    ratio    = lmte_arr[valid] / (lp_arr[valid] + 1e-12)

    gaps        = lp_arr[valid] - lmte_arr[valid]   # positive = LP routes more
    max_gap_idx = int(np.argmax(gaps))
    max_gap_val = gaps[max_gap_idx]
    # Map back to original sample index
    orig_indices = np.where(valid)[0]
    max_gap_sample = orig_indices[max_gap_idx]

    print('\n' + '=' * 60)
    print('  Total-Flow Comparison  (raw capacity units)')
    print('=' * 60)
    print(f'  Test samples          : {n_test}  (LP solved: {valid.sum()})')
    print(f'  LMTE   — mean : {lmte_arr.mean():.4e}   std : {lmte_arr.std():.4e}')
    print(f'  LP opt — mean : {lp_arr[valid].mean():.4e}   std : {lp_arr[valid].std():.4e}')
    print(f'  LMTE / LP-opt — mean : {ratio.mean():.4f}   std : {ratio.std():.4f}'
          f'   (1.0 = LP-optimal)')
    print(f'  Gap (LP − LMTE) — mean : {gaps.mean():.4e}   std : {gaps.std():.4e}')
    print(f'  Maximum gap           : {max_gap_val:.4e}'
          f'   (sample #{max_gap_sample},'
          f' LMTE={lmte_arr[max_gap_sample]:.4e},'
          f' LP={lp_arr[max_gap_sample]:.4e})')
    print('=' * 60)

    # Optionally save raw numbers for further analysis.
    out_dir = './results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.topology}_total_flow_comparison.npz')
    np.savez(out_path, lmte=lmte_arr, lp_optimal=lp_arr)
    print(f'Raw results saved to {out_path}')


if __name__ == '__main__':
    main()
