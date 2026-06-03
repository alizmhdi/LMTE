# Adversarial Attacks for LMTE

This note explains how [`adversarial_tm_attack.py`](/home/yizhuoliang/MetaRL/src/problems/traffic_engineering/solvers/LMTE/adversarial_tm_attack.py) searches for adversarial inputs for LMTE, and how the implemented `fgsm` and `pgd` methods work in this codebase.

## Goal

The script treats LMTE as a white-box model. That means it has access to the model parameters and can differentiate the final evaluation metric with respect to the model input.

For one test sample, the clean input is:

- `history`: a traffic-history window with shape `[1, window_size, num_commodities]`
- `target`: the next-step traffic demand with shape `[1, num_commodities]`

The attack searches for a perturbed input:

- `adv_history`
- `adv_target`

such that LMTE performs worse on the attacked sample than on the clean sample.

Depending on `--attack_surface`, the perturbation may affect:

- `history`: perturb only the history window
- `target`: perturb only the next demand vector
- `history_target`: perturb both

Across the dataset, `--num_samples` controls how many test samples are attacked sequentially. The script attacks one sample at a time and records one result row per attacked sample.

## End-to-End Attack Pipeline

For each test sample, the script does the following:

1. Load the clean `history` and `target`.
2. Run LMTE to produce path split ratios.
3. Evaluate the routing quality through a differentiable total-flow objective.
4. Compute gradients of the chosen attack objective with respect to the attacked input surface.
5. Update the input with either one FGSM step or multiple PGD steps.
6. Project the result back into the allowed `L_inf` perturbation set and valid demand range.
7. Keep the worst attacked input found during the attack loop.
8. Re-evaluate the final attacked sample and record LMTE flow, LP-optimal flow, and performance gaps.

In short, the model weights stay fixed, and the script optimizes the input instead.

## What LMTE Produces

LMTE outputs path split ratios:

```text
split_ratios = model(history, ...)
```

These ratios define how each source-destination commodity distributes demand across its candidate paths.

The attack does not directly perturb the split ratios. Instead, it perturbs the traffic input that LMTE consumes, which indirectly changes the predicted split ratios.

## Differentiable Routing Objective

The core differentiable evaluator is `differentiable_total_flow(...)`.

For each sample:

1. Invalid paths are masked out according to topology and capacities.
2. LMTE path weights are normalized commodity-wise.
3. Demand is placed onto paths.
4. Edge loads are computed.
5. A congestion scaling factor `gamma` is computed from the maximum edge load-to-capacity ratio.
6. Path flows are scaled by `1 / gamma` to obtain a feasible routed flow.
7. The routed flow for each commodity is capped by the true demand.
8. The total routed flow is summed across all commodities.

This gives two metrics:

- `total_flow`: total routed traffic volume
- `routed_fraction = total_flow / total_demand`

The attack can minimize either one, depending on `--attack_objective`.

## Attack Objective

The script supports:

- `--attack_objective routed_fraction`
- `--attack_objective total_flow`

Internally, the attack defines:

```text
minimized_metric = routed_fraction.mean()   or   total_flow.mean()
badness = -minimized_metric
```

Then it differentiates `badness` with respect to the attacked input.

So the attack is trying to make:

- routed fraction smaller, or
- total routed flow smaller

which means it is trying to make LMTE route traffic less effectively.

## Perturbation Budget

The attack is constrained by an `L_inf` budget:

```text
||adv - clean||_inf <= epsilon
```

This is enforced separately for the history tensor and target tensor.

The helper `project_linf(...)` does two things:

1. Projects the perturbed tensor back into the `epsilon`-ball around the clean input.
2. Clips the values into `[clip_min, clip_max]`.

By default, `epsilon` and `alpha` are in normalized traffic units. If raw traffic units are preferred, `--epsilon_raw` and `--alpha_raw` are converted by dividing with `--scale`.

## Preserving Total Demand

For `target` and `history_target` attacks, the script preserves total demand by default:

```text
adv_target = adv_target * (clean_sum / adv_sum)
```

This avoids a trivial attack where the adversary simply reduces all demands so that the routed-flow objective also becomes artificially small.

You can disable this with:

```bash
--allow_total_demand_change
```

## FGSM in This Script

FGSM stands for Fast Gradient Sign Method.

In this codebase, FGSM is the one-step version of the attack:

1. Start from the clean input.
2. Compute the gradient of `badness` with respect to the attacked input.
3. Move the input in the sign direction of that gradient:

```text
adv = adv + alpha * sign(grad)
```

4. Project back into the `L_inf` budget and valid range.

For FGSM:

- the number of steps is forced to `1`
- by default `alpha = epsilon`

So the usual FGSM update is:

```text
adv = Project(clean + epsilon * sign(grad))
```

In this implementation, the update is applied to whichever surface is enabled:

- `history`
- `target`
- or both

Pseudocode:

```text
input: clean input x, budget epsilon
grad = d badness(x) / d x
adv = x + epsilon * sign(grad)
adv = Project(adv)
return adv
```

## PGD in This Script

PGD stands for Projected Gradient Descent. In adversarial ML, it is commonly used as an iterative stronger version of FGSM.

In this codebase, PGD works as follows:

1. Initialize `adv_history` and `adv_target` from the clean input.
2. Optionally apply a random start inside the `L_inf` ball if `--random_start` is enabled.
3. Repeat for `steps` iterations:
   - compute gradients of `badness`
   - update the attacked tensor by `alpha * sign(grad)`
   - project back to the feasible `L_inf` ball
   - optionally preserve total target demand
   - evaluate the current attacked sample
4. Keep the best attack found during the loop, not just the final iterate.

The update rule is:

```text
adv_{t+1} = Project(adv_t + alpha * sign(grad_t))
```

where `Project(...)` includes:

- `L_inf` projection around the clean input
- clipping to `[clip_min, clip_max]`
- optional demand-preserving rescaling for the target

By default:

- `steps = max(1, args.steps)`
- if `alpha` is not provided, `alpha = epsilon / 5`

That makes PGD a multi-step attack with smaller steps than FGSM.

Pseudocode:

```text
input: clean input x, budget epsilon, step size alpha, steps T
adv = x
if random_start:
    adv = Project(x + Uniform(-epsilon, epsilon))

best = adv
for t = 1 ... T:
    grad = d badness(adv) / d adv
    adv = adv + alpha * sign(grad)
    adv = Project(adv)
    if badness(adv) is worse than badness(best):
        best = adv
return best
```

## Random Start

When `--random_start` is enabled and `--attack pgd` is used, the attack starts from a random point inside the `L_inf` ball:

```text
adv = clean + Uniform(-epsilon, epsilon)
```

followed by projection and clipping.

This is useful because PGD can otherwise stay tied to a specific local path from the clean point. Random starts often make the attack stronger.

## Why the Script Switches the Model to Train Mode

Inside the attack loop, the script temporarily uses:

```python
model.train()
```

before calling backward, and then switches back with:

```python
model.eval()
```

This is not training the model weights. The weights remain frozen. The reason is a PyTorch/cuDNN requirement: RNN backward can require training mode for gradient computation through the input.

## Best-Attack Tracking

The script does not assume the last PGD iterate is always the worst one.

Instead, after every update it re-evaluates the attacked sample and stores the best adversarial example found so far according to:

- lowest routed fraction, or
- lowest total flow

This is handled through:

- `best_history`
- `best_target`
- `best_badness`

This is important because projection and demand-preserving rescaling can make intermediate iterates better attack points than the final one.

## Performance Gap After the Attack

After the adversarial input is found, the script compares LMTE against the LP-optimal total-flow solution.

For total-flow experiments, the recorded gap is:

```text
performance_gap = (optimal_total_flow - lmte_total_flow) / total_capacity
```

The script records:

- `initial_performance_gap`
- `adv_performance_gap`
- `performance_gap_increase`

So the attack is useful in two related ways:

1. It finds an input that degrades LMTE performance.
2. It shows how much farther LMTE moves away from the optimal solver under attack.

## Outputs Produced by the Script

The attack run writes several artifacts under `--out_dir`:

- `adversarial_tm_attack_summary.csv`: per-sample attack results
- `adversarial_tm_gap_history.csv`: attack-order history for plotting gap over time
- `adversarial_targets_raw_full.csv`: adversarial target TMs in raw units
- `adversarial_histories_raw_full_flat.csv`: adversarial history windows in raw units
- `adversarial_tm_attack_run_summary.csv`: run-level metadata and total runtime

If `--save_npz` is enabled, it also writes:

- `adversarial_tm_attack_tensors.npz`

## Choosing Between FGSM and PGD

FGSM is useful when:

- you want a fast baseline attack
- you want one gradient step per sample
- you want a weaker but cheap perturbation method

PGD is useful when:

- you want a stronger attack
- you want to search more thoroughly inside the allowed perturbation region
- you want the standard iterative white-box robustness test

In practice, PGD is usually the more informative robustness evaluation because it can find stronger adversarial inputs than FGSM under the same `epsilon` bound.

## Practical Interpretation

For LMTE, these attacks are not changing the topology or capacities. They are changing the traffic information the model sees.

That means the attack answers a robustness question of the form:

> If the observed traffic history or target demand is adversarially perturbed within a small bounded budget, how much can LMTE’s routing quality degrade?

This is especially meaningful because LMTE is making path-allocation decisions from traffic patterns, so sensitivity to small structured demand perturbations directly reflects sensitivity in the control pipeline.

## Minimal Example

```bash
python adversarial_tm_attack.py \
  --checkpoint checkpoints/Abilene_lmte_0/checkpoint.pt \
  --topology Abilene \
  --topology_filepath data/Abilene/topology.json \
  --tm_filepath data/Abilene/Abilene_normal.csv \
  --attack pgd \
  --attack_surface history_target \
  --attack_objective routed_fraction \
  --epsilon 0.05 \
  --steps 20 \
  --random_start \
  --num_samples 50
```

This runs an iterative PGD attack on both traffic history and target demand, under an `L_inf` budget of `0.05` in normalized units, while trying to reduce LMTE's routed fraction.
