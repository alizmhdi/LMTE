#!/usr/bin/env python
"""
Standalone FGSM/PGD adversarial traffic-matrix search for LMTE.

The attack is white-box: it differentiates through the LMTE model input and the
total-flow objective, then perturbs traffic history/features within an L-inf
budget to find inputs with poor routed-flow performance.

Example, run from the LMTE directory or from anywhere:

    python adversarial_tm_attack.py \
        --checkpoint checkpoints/Abilene_lmte_0/checkpoint.pt \
        --topology Abilene \
        --topology_filepath data/Abilene/topology.json \
        --tm_filepath data/Abilene/Abilene_normal.csv \
        --attack pgd \
        --attack_surface history \
        --epsilon 0.05 \
        --steps 20 \
        --num_samples 50

Notes:
  * epsilon/alpha are in normalized TM units after dividing by --scale.
  * Use --epsilon_raw/--alpha_raw if you prefer raw traffic units.
  * For target/history_target attacks, total demand is preserved by default to
    avoid the trivial solution of reducing all demands.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.build_dataloader import build_dataloader  # noqa: E402
from src.model import LmteModel  # noqa: E402
from src.utils.read_data import read_graph_from_json, read_paths_from_file  # noqa: E402
from src.utils.useful_functions import (  # noqa: E402
    compute_ksp_paths,
    get_capacities_from_graph,
    get_commodities_to_paths,
    get_paths_to_edges,
    mask_invalid_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find adversarial traffic matrices for LMTE total-flow robustness."
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint.pt, or a checkpoint directory containing checkpoint.pt.")
    parser.add_argument("--topology", type=str, default="Abilene",
                        help="Topology name, used to locate data/<topology>/paths.txt.")
    parser.add_argument("--topology_filepath", type=str, default="data/Abilene/topology.json")
    parser.add_argument("--tm_filepath", type=str, default="data/Abilene/Abilene_normal.csv")
    parser.add_argument("--num_paths", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Same normalization scale used for LMTE training/evaluation.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    parser.add_argument("--attack", choices=("fgsm", "pgd"), default="pgd")
    parser.add_argument("--attack_surface",
                        choices=("history", "target", "history_target"),
                        default="history",
                        help="Perturb history features, target demand, or both.")
    parser.add_argument("--attack_objective",
                        choices=("routed_fraction", "total_flow"),
                        default="routed_fraction",
                        help="Quantity to minimize. routed_fraction avoids trivial low-demand attacks.")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="L-inf perturbation budget in normalized TM units.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="PGD step size in normalized TM units. Defaults to epsilon for FGSM, epsilon/5 for PGD.")
    parser.add_argument("--epsilon_raw", type=float, default=None,
                        help="Alternative L-inf budget in raw TM units. Overrides --epsilon after division by --scale.")
    parser.add_argument("--alpha_raw", type=float, default=None,
                        help="Alternative step size in raw TM units. Overrides --alpha after division by --scale.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--random_start", action="store_true",
                        help="Start PGD from a random point inside the epsilon ball.")
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--clip_max", type=float, default=None,
                        help="Maximum normalized demand after attack. Default: observed max * --clip_max_factor.")
    parser.add_argument("--clip_max_factor", type=float, default=1.2)

    preserve = parser.add_mutually_exclusive_group()
    preserve.add_argument("--preserve_total_demand", dest="preserve_total_demand",
                          action="store_true", default=True,
                          help="Rescale adversarial target demand to keep the original total demand.")
    preserve.add_argument("--allow_total_demand_change", dest="preserve_total_demand",
                          action="store_false",
                          help="Allow target-demand attacks to change total demand.")

    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of test samples to attack. Use <=0 for all test samples.")
    parser.add_argument("--start_index", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/adversarial_tm")
    parser.add_argument("--save_npz", action="store_true",
                        help="Save clean/adversarial history and target tensors as NPZ.")
    lp_group = parser.add_mutually_exclusive_group()
    lp_group.add_argument("--run_lp", dest="run_lp", action="store_true", default=True,
                          help="Compute LP-optimal total flow and performance gaps. Enabled by default.")
    lp_group.add_argument("--skip_lp", dest="run_lp", action="store_false",
                          help="Skip LP-optimal total flow and performance-gap computation.")

    parser.add_argument("--model_objective", choices=("mlu", "total_flow"), default="total_flow",
                        help="Objective string passed into LmteModel. Use the value from training.")
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_keys", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--llm_dim", type=int, default=4096)
    parser.add_argument("--llm_layers", type=int, default=8)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_rnn_layers", type=int, default=2)
    parser.add_argument("--num_dnn_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--llm_model", type=str, default="llama-8b")
    parser.add_argument("--use_divide_head", action="store_true")
    parser.add_argument("--keep_prompt_grad_stats", action="store_true",
                        help="By default prompt text is built from detached histories; this keeps the original behavior.")

    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (_HERE / path).resolve()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_detach_prompt_stats() -> None:
    """Keep tokenized prompt statistics out of the input-gradient graph."""
    import src.model as model_module

    original_prepare_prompts = model_module.prepare_prompts

    def detached_prepare_prompts(topology, histories, *args, **kwargs):
        return original_prepare_prompts(topology, histories.detach(), *args, **kwargs)

    model_module.prepare_prompts = detached_prepare_prompts


def load_paths(topology, topology_name: str, num_paths: int, num_nodes: int) -> Tuple[dict, dict]:
    paths_dir = resolve_path(f"data/{topology_name}")
    paths_file = paths_dir / "paths.txt"
    if not paths_file.exists():
        paths_dir.mkdir(parents=True, exist_ok=True)
        pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        compute_ksp_paths(
            k=num_paths,
            pairs=pairs,
            graph=topology,
            save2txt=True,
            filepath=str(paths_dir),
            transform=True,
        )
    node_paths = read_paths_from_file(str(paths_file), num_nodes=num_nodes, convert=False)
    edge_paths = read_paths_from_file(str(paths_file), num_nodes=num_nodes, convert=True)
    return node_paths, edge_paths


def sparse_scipy_to_torch(matrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape), device=device).coalesce()


def build_model_and_tensors(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    topology_path = resolve_path(args.topology_filepath)
    tm_path = resolve_path(args.tm_filepath)

    topo, _ = read_graph_from_json(str(topology_path))
    num_nodes = len(topo.nodes())
    num_edges = len(topo.edges())
    _, edge_paths = load_paths(topo, args.topology, args.num_paths, num_nodes)

    _, _, test_loader = build_dataloader(
        str(topology_path),
        str(tm_path),
        args.batch_size,
        args.scale,
        args.eval_batch_size,
        args.window_size,
        split_ratio=(0.7, 0.1, 0.2),
    )
    dataset = test_loader.dataset

    p_matrix = get_paths_to_edges(topo, paths=edge_paths)
    c_matrix = get_commodities_to_paths(topo, num_paths=p_matrix.shape[0], paths=edge_paths)
    paths_to_edges = sparse_scipy_to_torch(p_matrix, device)
    commodities_to_paths = sparse_scipy_to_torch(c_matrix, device)

    edge_index = dataset.get_edge_index().to(device)
    node_features = dataset.get_node_features().float().to(device)
    capacities = dataset.capacities.float().to(device)
    edge_ids_per_path = dataset.get_padded_edge_ids_per_path(edge_paths).to(device)
    max_path_length = int(edge_ids_per_path.shape[-1])

    model = LmteModel(
        d_model=args.d_model,
        d_llm=args.llm_dim,
        num_nodes=num_nodes,
        num_paths=args.num_paths,
        num_eages=num_edges,
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
        objective=args.model_objective,
    ).float().to(device)

    checkpoint = resolve_path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoint.pt"
    state_dict = torch.load(str(checkpoint), map_location=device)
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for name, param in state_dict.items():
        key = name.replace("module.", "")
        if key in model_state and model_state[key].shape == param.shape:
            model_state[key].copy_(param.to(device=model_state[key].device, dtype=model_state[key].dtype))
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state, strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Loaded parameters: {loaded}; skipped: {skipped}")
    print(f"Test samples: {len(dataset)}")

    return {
        "model": model,
        "topology": topo,
        "num_nodes": num_nodes,
        "test_dataset": dataset,
        "paths_to_edges": paths_to_edges,
        "commodities_to_paths": commodities_to_paths,
        "edge_index": edge_index,
        "node_features": node_features,
        "capacities": capacities,
        "edge_ids_per_path": edge_ids_per_path,
    }


def differentiable_total_flow(
    split_ratios: torch.Tensor,
    tm: torch.Tensor,
    paths_to_edges: torch.Tensor,
    commodities_to_paths: torch.Tensor,
    capacities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return total routed flow and routed fraction with gradients intact."""
    c_2_p = mask_invalid_paths(commodities_to_paths, paths_to_edges, capacities)
    capacity = capacities.float().view(-1, 1)
    valid_capacity = capacity != 0

    total_flows = []
    routed_fractions = []
    for i in range(tm.shape[0]):
        true_tm = tm[[i]]
        w_p = split_ratios[[i]].transpose(0, 1)

        commodity_total_weight = c_2_p.matmul(w_p).clamp_min(1e-8)
        paths_over_total = c_2_p.transpose(0, 1).matmul(1.0 / commodity_total_weight)
        x_p = w_p.mul(paths_over_total)

        demand_on_paths = c_2_p.transpose(0, 1).matmul(true_tm.transpose(0, 1)).mul(x_p)
        edge_loads = paths_to_edges.transpose(0, 1).matmul(demand_on_paths)
        if bool(valid_capacity.any()):
            gamma = (edge_loads[valid_capacity] / capacity[valid_capacity]).max().clamp_min(1e-8)
        else:
            gamma = torch.tensor(1e-8, dtype=tm.dtype, device=tm.device)

        omega_p = demand_on_paths / gamma
        max_flow_per_commodity = c_2_p.matmul(omega_p)
        total_flow = torch.minimum(max_flow_per_commodity, true_tm.transpose(0, 1)).sum()
        routed_fraction = total_flow / true_tm.sum().clamp_min(1e-8)

        total_flows.append(total_flow)
        routed_fractions.append(routed_fraction)

    return torch.stack(total_flows), torch.stack(routed_fractions)


def forward_metrics(
    model: LmteModel,
    history: torch.Tensor,
    target: torch.Tensor,
    tensors: Dict[str, object],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_ratios = model(
        history,
        tensors["node_features"],
        tensors["edge_index"],
        tensors["capacities"],
        tensors["topology"],
        None,
        tensors["edge_ids_per_path"],
    )
    total_flow, routed_fraction = differentiable_total_flow(
        split_ratios,
        target,
        tensors["paths_to_edges"],
        tensors["commodities_to_paths"],
        tensors["capacities"],
    )
    return split_ratios, total_flow, routed_fraction


def project_linf(
    candidate: torch.Tensor,
    clean: torch.Tensor,
    epsilon: float,
    clip_min: Optional[float],
    clip_max: Optional[float],
) -> torch.Tensor:
    candidate = torch.max(torch.min(candidate, clean + epsilon), clean - epsilon)
    if clip_min is not None or clip_max is not None:
        min_value = -float("inf") if clip_min is None else clip_min
        max_value = float("inf") if clip_max is None else clip_max
        candidate = candidate.clamp(min=min_value, max=max_value)
    return candidate


def preserve_total_demand(adv_target: torch.Tensor, clean_target: torch.Tensor) -> torch.Tensor:
    clean_sum = clean_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    adv_sum = adv_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return adv_target * (clean_sum / adv_sum)


def attack_one_sample(
    model: LmteModel,
    clean_history: torch.Tensor,
    clean_target: torch.Tensor,
    tensors: Dict[str, object],
    args: argparse.Namespace,
    epsilon: float,
    alpha: float,
    clip_max: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    attack_history = args.attack_surface in ("history", "history_target")
    attack_target = args.attack_surface in ("target", "history_target")
    steps = 1 if args.attack == "fgsm" else max(1, args.steps)

    adv_history = clean_history.detach().clone()
    adv_target = clean_target.detach().clone()

    if args.random_start and args.attack == "pgd":
        if attack_history:
            adv_history = adv_history + torch.empty_like(adv_history).uniform_(-epsilon, epsilon)
            adv_history = project_linf(adv_history, clean_history, epsilon, args.clip_min, clip_max)
        if attack_target:
            adv_target = adv_target + torch.empty_like(adv_target).uniform_(-epsilon, epsilon)
            adv_target = project_linf(adv_target, clean_target, epsilon, args.clip_min, clip_max)
            if args.preserve_total_demand:
                adv_target = preserve_total_demand(adv_target, clean_target)
                adv_target = project_linf(adv_target, clean_target, epsilon, args.clip_min, clip_max)

    best_history = adv_history.detach().clone()
    best_target = adv_target.detach().clone()
    with torch.no_grad():
        _, initial_total_flow, initial_fraction = forward_metrics(model, best_history, best_target, tensors)
        if args.attack_objective == "routed_fraction":
            best_badness = -float(initial_fraction.detach().cpu().mean())
        else:
            best_badness = -float(initial_total_flow.detach().cpu().mean())
        best_total_flow = float(initial_total_flow.detach().cpu().mean())
        best_fraction = float(initial_fraction.detach().cpu().mean())

    for _ in range(steps):
        adv_history = adv_history.detach().requires_grad_(attack_history)
        adv_target = adv_target.detach().requires_grad_(attack_target)

        model.train()  # cuDNN RNN backward requires training mode
        _, total_flow, routed_fraction = forward_metrics(model, adv_history, adv_target, tensors)
        minimized_metric = routed_fraction.mean() if args.attack_objective == "routed_fraction" else total_flow.mean()
        badness = -minimized_metric

        grad_inputs = []
        if attack_history:
            grad_inputs.append(adv_history)
        if attack_target:
            grad_inputs.append(adv_target)
        grads = torch.autograd.grad(badness, grad_inputs, allow_unused=True)
        model.eval()

        with torch.no_grad():
            grad_iter = iter(grads)
            if attack_history:
                grad = next(grad_iter)
                if grad is not None:
                    adv_history = adv_history + alpha * grad.sign()
                adv_history = project_linf(adv_history, clean_history, epsilon, args.clip_min, clip_max)
            if attack_target:
                grad = next(grad_iter)
                if grad is not None:
                    adv_target = adv_target + alpha * grad.sign()
                adv_target = project_linf(adv_target, clean_target, epsilon, args.clip_min, clip_max)
                if args.preserve_total_demand:
                    adv_target = preserve_total_demand(adv_target, clean_target)
                    adv_target = project_linf(adv_target, clean_target, epsilon, args.clip_min, clip_max)

            _, eval_total_flow, eval_fraction = forward_metrics(model, adv_history, adv_target, tensors)
            if args.attack_objective == "routed_fraction":
                current_badness = -float(eval_fraction.detach().cpu().mean())
            else:
                current_badness = -float(eval_total_flow.detach().cpu().mean())
            if current_badness > best_badness:
                best_badness = current_badness
                best_history = adv_history.detach().clone()
                best_target = adv_target.detach().clone()
                best_total_flow = float(eval_total_flow.detach().cpu().mean())
                best_fraction = float(eval_fraction.detach().cpu().mean())

    return best_history, best_target, {
        "attack_badness": best_badness,
        "attack_inner_total_flow": best_total_flow,
        "attack_inner_routed_fraction": best_fraction,
    }


@torch.no_grad()
def evaluate_one_sample(
    model: LmteModel,
    history: torch.Tensor,
    target: torch.Tensor,
    tensors: Dict[str, object],
) -> Dict[str, float]:
    _, total_flow, routed_fraction = forward_metrics(model, history, target, tensors)
    return {
        "total_flow": float(total_flow.detach().cpu().mean()),
        "routed_fraction": float(routed_fraction.detach().cpu().mean()),
        "total_demand": float(target.detach().cpu().sum()),
    }


def expand_flat_od(flat_tms: np.ndarray, num_nodes: int) -> np.ndarray:
    mask = np.ones((num_nodes, num_nodes), dtype=bool)
    np.fill_diagonal(mask, 0)
    full = np.zeros((flat_tms.shape[0], num_nodes * num_nodes), dtype=flat_tms.dtype)
    full[:, mask.flatten()] = flat_tms
    return full


def expand_history_od(histories: np.ndarray, num_nodes: int) -> np.ndarray:
    if histories.ndim != 3:
        raise ValueError(f"Expected histories with shape [samples, window, commodities], got {histories.shape}")
    num_samples, window_size, num_commodities = histories.shape
    full = expand_flat_od(histories.reshape(num_samples * window_size, num_commodities), num_nodes)
    return full.reshape(num_samples, window_size * num_nodes * num_nodes)


def as_float_array(items, shape_tail: Tuple[int, ...]) -> np.ndarray:
    if items:
        return np.asarray(items, dtype=np.float32)
    return np.empty((0, *shape_tail), dtype=np.float32)


def total_flow_performance_gap(optimal_raw: float, lmte_raw: float, total_capacity: float) -> float:
    if total_capacity <= 0 or not np.isfinite(optimal_raw) or not np.isfinite(lmte_raw):
        return float("nan")
    return (optimal_raw - lmte_raw) / total_capacity


def build_gap_history(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "attack_order",
        "sample_idx",
        "elapsed_seconds",
        "sample_runtime_seconds",
        "attack_runtime_seconds",
        "lp_runtime_seconds",
        "initial_performance_gap",
        "adv_performance_gap",
        "performance_gap_increase",
        "best_adv_performance_gap_so_far",
        "best_adv_gap_sample_idx_so_far",
        "best_gap_increase_so_far",
        "best_gap_increase_sample_idx_so_far",
        "clean_routed_fraction",
        "adv_routed_fraction",
        "routed_fraction_drop",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    base_columns = [col for col in columns if col not in {
        "attack_order",
        "best_adv_performance_gap_so_far",
        "best_adv_gap_sample_idx_so_far",
        "best_gap_increase_so_far",
        "best_gap_increase_sample_idx_so_far",
    }]
    history = df[base_columns].copy()
    history.insert(0, "attack_order", np.arange(len(history), dtype=np.int64))

    best_adv_gap = float("nan")
    best_adv_sample = float("nan")
    best_gap_increase = float("nan")
    best_gap_increase_sample = float("nan")
    best_adv_gaps = []
    best_adv_samples = []
    best_gap_increases = []
    best_gap_increase_samples = []

    for _, row in history.iterrows():
        adv_gap = float(row["adv_performance_gap"])
        gap_increase = float(row["performance_gap_increase"])
        sample_idx = float(row["sample_idx"])

        if np.isfinite(adv_gap) and (not np.isfinite(best_adv_gap) or adv_gap > best_adv_gap):
            best_adv_gap = adv_gap
            best_adv_sample = sample_idx
        if np.isfinite(gap_increase) and (
            not np.isfinite(best_gap_increase) or gap_increase > best_gap_increase
        ):
            best_gap_increase = gap_increase
            best_gap_increase_sample = sample_idx

        best_adv_gaps.append(best_adv_gap)
        best_adv_samples.append(best_adv_sample)
        best_gap_increases.append(best_gap_increase)
        best_gap_increase_samples.append(best_gap_increase_sample)

    history["best_adv_performance_gap_so_far"] = best_adv_gaps
    history["best_adv_gap_sample_idx_so_far"] = best_adv_samples
    history["best_gap_increase_so_far"] = best_gap_increases
    history["best_gap_increase_sample_idx_so_far"] = best_gap_increase_samples
    return history[columns]


def maybe_build_lp_solver(args: argparse.Namespace, topology, num_nodes: int):
    if not args.run_lp:
        return None
    try:
        cl_baselines_dir = _HERE / "cl_baselines"
        if str(cl_baselines_dir) not in sys.path:
            sys.path.insert(0, str(cl_baselines_dir))
        from optimal_total_flow import TotalFlowOptimal
        from useful_functions import Get_edge_to_path
    except Exception as exc:
        print(f"LP comparison disabled; could not import LP baseline: {exc}")
        return None

    node_paths, _ = load_paths(topology, args.topology, args.num_paths, num_nodes)
    edge_to_path = Get_edge_to_path(topology, node_paths)
    return TotalFlowOptimal(topology, node_paths, edge_to_path)


def lp_total_flow(lp_solver, target: torch.Tensor, scale: float, num_nodes: int) -> float:
    if lp_solver is None:
        return float("nan")
    flat = target.detach().cpu().numpy() * scale
    raw_full = expand_flat_od(flat, num_nodes)[0]
    try:
        value, _ = lp_solver.maximize_total_flow(raw_full)
        return float(value) if value is not None else float("nan")
    except Exception as exc:
        print(f"LP solve failed for one sample: {exc}")
        return float("nan")


def choose_sample_indices(dataset_len: int, start_index: int, num_samples: int) -> Iterable[int]:
    start = max(0, start_index)
    stop = dataset_len if num_samples <= 0 else min(dataset_len, start + num_samples)
    return range(start, stop)


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    os.chdir(_HERE)
    set_seed(args.seed)
    if not args.keep_prompt_grad_stats:
        maybe_detach_prompt_stats()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    epsilon = args.epsilon_raw / args.scale if args.epsilon_raw is not None else args.epsilon
    if args.alpha_raw is not None:
        alpha = args.alpha_raw / args.scale
    elif args.alpha is not None:
        alpha = args.alpha
    else:
        alpha = epsilon if args.attack == "fgsm" else epsilon / 5.0

    tensors = build_model_and_tensors(args, device)
    model = tensors["model"]
    dataset = tensors["test_dataset"]
    histories = dataset.tm_seqences.float()
    targets = dataset.tm_preds.float()
    observed_max = float(torch.max(torch.stack([histories.max(), targets.max()])).item())
    clip_max = args.clip_max if args.clip_max is not None else observed_max * args.clip_max_factor
    total_capacity = float(sum(get_capacities_from_graph(tensors["topology"])))

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lp_solver = maybe_build_lp_solver(args, tensors["topology"], tensors["num_nodes"])
    if not args.run_lp:
        lp_status = "skipped"
    elif lp_solver is None:
        lp_status = "unavailable"
    else:
        lp_status = "enabled"

    print(f"Device: {device}")
    print(f"Attack: {args.attack.upper()} on {args.attack_surface}, epsilon={epsilon:g}, alpha={alpha:g}, clip_max={clip_max:g}")
    print(f"Attack objective: minimize {args.attack_objective}")
    print(f"Total graph capacity: {total_capacity:.6e}")
    print(f"LP optimal comparison: {lp_status}")

    rows = []
    attacked_sample_indices = []
    clean_histories, clean_targets = [], []
    adv_histories, adv_targets = [], []
    sample_indices = list(choose_sample_indices(len(dataset), args.start_index, args.num_samples))

    for sample_idx in tqdm(sample_indices, desc="Attacking"):
        sample_start = time.perf_counter()
        clean_history = histories[sample_idx:sample_idx + 1].to(device)
        clean_target = targets[sample_idx:sample_idx + 1].to(device)

        clean_metrics = evaluate_one_sample(model, clean_history, clean_target, tensors)
        attack_start = time.perf_counter()
        adv_history, adv_target, attack_stats = attack_one_sample(
            model,
            clean_history,
            clean_target,
            tensors,
            args,
            epsilon=epsilon,
            alpha=alpha,
            clip_max=clip_max,
        )
        attack_runtime = time.perf_counter() - attack_start
        adv_metrics = evaluate_one_sample(model, adv_history, adv_target, tensors)

        lp_start = time.perf_counter()
        clean_lp = lp_total_flow(lp_solver, clean_target, args.scale, tensors["num_nodes"])
        adv_lp = lp_total_flow(lp_solver, adv_target, args.scale, tensors["num_nodes"])
        lp_runtime = time.perf_counter() - lp_start

        clean_total_flow_raw = clean_metrics["total_flow"] * args.scale
        adv_total_flow_raw = adv_metrics["total_flow"] * args.scale
        initial_gap = total_flow_performance_gap(clean_lp, clean_total_flow_raw, total_capacity)
        adv_gap = total_flow_performance_gap(adv_lp, adv_total_flow_raw, total_capacity)
        gap_increase = adv_gap - initial_gap if np.isfinite(initial_gap) and np.isfinite(adv_gap) else np.nan

        history_linf = float((adv_history - clean_history).abs().max().detach().cpu())
        target_linf = float((adv_target - clean_target).abs().max().detach().cpu())
        routed_drop = clean_metrics["routed_fraction"] - adv_metrics["routed_fraction"]
        total_flow_drop = clean_metrics["total_flow"] - adv_metrics["total_flow"]
        sample_runtime = time.perf_counter() - sample_start
        elapsed_seconds = time.perf_counter() - run_start

        rows.append({
            "sample_idx": sample_idx,
            "total_capacity_raw": total_capacity,
            "clean_total_flow_normalized": clean_metrics["total_flow"],
            "adv_total_flow_normalized": adv_metrics["total_flow"],
            "clean_total_flow_raw": clean_total_flow_raw,
            "adv_total_flow_raw": adv_total_flow_raw,
            "initial_lmte_total_flow_raw": clean_total_flow_raw,
            "adv_lmte_total_flow_raw": adv_total_flow_raw,
            "initial_optimal_total_flow_raw": clean_lp,
            "adv_optimal_total_flow_raw": adv_lp,
            "initial_performance_gap": initial_gap,
            "adv_performance_gap": adv_gap,
            "performance_gap_increase": gap_increase,
            "clean_routed_fraction": clean_metrics["routed_fraction"],
            "adv_routed_fraction": adv_metrics["routed_fraction"],
            "routed_fraction_drop": routed_drop,
            "total_flow_drop_normalized": total_flow_drop,
            "total_flow_drop_raw": total_flow_drop * args.scale,
            "clean_total_demand_normalized": clean_metrics["total_demand"],
            "adv_total_demand_normalized": adv_metrics["total_demand"],
            "history_linf": history_linf,
            "target_linf": target_linf,
            "lp_clean_total_flow_raw": clean_lp,
            "lp_adv_total_flow_raw": adv_lp,
            "clean_lmte_over_lp": clean_total_flow_raw / (clean_lp + 1e-12) if np.isfinite(clean_lp) else np.nan,
            "adv_lmte_over_lp": adv_total_flow_raw / (adv_lp + 1e-12) if np.isfinite(adv_lp) else np.nan,
            "elapsed_seconds": elapsed_seconds,
            "sample_runtime_seconds": sample_runtime,
            "attack_runtime_seconds": attack_runtime,
            "lp_runtime_seconds": lp_runtime,
            **attack_stats,
        })
        attacked_sample_indices.append(sample_idx)
        clean_histories.append(clean_history.detach().cpu().numpy()[0])
        clean_targets.append(clean_target.detach().cpu().numpy()[0])
        adv_histories.append(adv_history.detach().cpu().numpy()[0])
        adv_targets.append(adv_target.detach().cpu().numpy()[0])

    df = pd.DataFrame(rows)
    ranked_df = df
    if not df.empty:
        sort_column = "adv_performance_gap" if np.isfinite(df['adv_performance_gap'].to_numpy(dtype=float)).any() else "adv_routed_fraction"
        ranked_df = df.sort_values(sort_column, ascending=(sort_column != "adv_performance_gap"))

    csv_path = out_dir / "adversarial_tm_attack_summary.csv"
    df.to_csv(csv_path, index=False)
    gap_history_path = out_dir / "adversarial_tm_gap_history.csv"
    gap_history_df = build_gap_history(df)
    gap_history_df.to_csv(gap_history_path, index=False)

    clean_histories_arr = as_float_array(clean_histories, tuple(histories.shape[1:]))
    clean_targets_arr = as_float_array(clean_targets, tuple(targets.shape[1:]))
    adv_histories_arr = as_float_array(adv_histories, tuple(histories.shape[1:]))
    adv_targets_arr = as_float_array(adv_targets, tuple(targets.shape[1:]))

    adv_csv_path = out_dir / "adversarial_targets_raw_full.csv"
    adv_history_csv_path = out_dir / "adversarial_histories_raw_full_flat.csv"
    pd.DataFrame(expand_flat_od(adv_targets_arr * args.scale, tensors["num_nodes"])).to_csv(
        adv_csv_path, header=False, index=False
    )
    pd.DataFrame(expand_history_od(adv_histories_arr * args.scale, tensors["num_nodes"])).to_csv(
        adv_history_csv_path, header=False, index=False
    )

    npz_path = None
    if args.save_npz:
        npz_path = out_dir / "adversarial_tm_attack_tensors.npz"
        np.savez_compressed(
            npz_path,
            clean_histories_normalized=clean_histories_arr,
            clean_targets_normalized=clean_targets_arr,
            adversarial_histories_normalized=adv_histories_arr,
            adversarial_targets_normalized=adv_targets_arr,
            clean_targets_raw_full=expand_flat_od(clean_targets_arr * args.scale, tensors["num_nodes"]),
            adversarial_targets_raw_full=expand_flat_od(adv_targets_arr * args.scale, tensors["num_nodes"]),
            adversarial_histories_raw_full_flat=expand_history_od(adv_histories_arr * args.scale, tensors["num_nodes"]),
            sample_indices=np.asarray(attacked_sample_indices, dtype=np.int64),
            initial_performance_gaps=df['initial_performance_gap'].to_numpy(dtype=np.float32) if "initial_performance_gap" in df else np.empty(0, dtype=np.float32),
            adv_performance_gaps=df['adv_performance_gap'].to_numpy(dtype=np.float32) if "adv_performance_gap" in df else np.empty(0, dtype=np.float32),
            total_capacity=np.asarray(total_capacity, dtype=np.float32),
            scale=np.asarray(args.scale, dtype=np.float32),
            epsilon=np.asarray(epsilon, dtype=np.float32),
            alpha=np.asarray(alpha, dtype=np.float32),
        )
        print(f"Saved tensors: {npz_path}")

    total_runtime_seconds = time.perf_counter() - run_start
    metadata_path = out_dir / "adversarial_tm_attack_run_summary.csv"
    pd.DataFrame([{
        "total_runtime_seconds": total_runtime_seconds,
        "num_attacked_samples": len(df),
        "total_capacity_raw": total_capacity,
        "lp_status": lp_status,
        "attack": args.attack,
        "attack_surface": args.attack_surface,
        "attack_objective": args.attack_objective,
        "epsilon": epsilon,
        "alpha": alpha,
        "steps": args.steps,
        "results_csv": str(csv_path),
        "gap_history_csv": str(gap_history_path),
        "adversarial_targets_csv": str(adv_csv_path),
        "adversarial_histories_csv": str(adv_history_csv_path),
        "tensors_npz": str(npz_path) if npz_path is not None else "",
    }]).to_csv(metadata_path, index=False)

    if len(df):
        has_gap = np.isfinite(df['adv_performance_gap'].to_numpy(dtype=float)).any()
        if has_gap:
            print("\nWorst adversarial samples by performance gap:")
            display_cols = [
                "sample_idx",
                "initial_performance_gap",
                "adv_performance_gap",
                "performance_gap_increase",
                "clean_routed_fraction",
                "adv_routed_fraction",
                "history_linf",
                "target_linf",
            ]
        else:
            print("\nWorst adversarial samples by routed fraction:")
            display_cols = [
                "sample_idx",
                "clean_routed_fraction",
                "adv_routed_fraction",
                "routed_fraction_drop",
                "history_linf",
                "target_linf",
            ]
        print(ranked_df.head(10)[display_cols].to_string(index=False))
        print("\nSummary:")
        print(f"  clean routed fraction mean: {df['clean_routed_fraction'].mean():.6f}")
        print(f"  adv routed fraction mean:   {df['adv_routed_fraction'].mean():.6f}")
        print(f"  mean routed fraction drop:  {df['routed_fraction_drop'].mean():.6f}")
        if has_gap:
            print(f"  initial performance gap mean: {df['initial_performance_gap'].mean():.6f}")
            print(f"  adv performance gap mean:     {df['adv_performance_gap'].mean():.6f}")
            print(f"  mean performance gap increase: {df['performance_gap_increase'].mean():.6f}")
            final_best_gap = gap_history_df['best_adv_performance_gap_so_far'].iloc[-1]
            final_best_sample = gap_history_df['best_adv_gap_sample_idx_so_far'].iloc[-1]
            print(f"  best adv performance gap found: {final_best_gap:.6f} (sample #{int(final_best_sample)})")
        else:
            print("  performance gaps are NaN because LP optimal totals were unavailable or skipped.")

    print(f"\nSaved summary CSV: {csv_path}")
    print(f"Saved gap history CSV: {gap_history_path}")
    print(f"Saved adversarial targets CSV: {adv_csv_path}")
    print(f"Saved adversarial histories CSV: {adv_history_csv_path}")
    print(f"Saved run summary CSV: {metadata_path}")
    print(f"Total runtime: {total_runtime_seconds:.2f}s")


if __name__ == "__main__":
    main()
