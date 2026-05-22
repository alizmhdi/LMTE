"""
Report the min and max total demand (sum of all OD pairs) across the test set.
Mirrors the TEDataset split: last 20% of windows with default window_size=12.

Usage:
    python check_demand_range.py [--tm_filepath PATH] [--window_size N] [--scale S]
"""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tm_filepath',
        default='data/Abilene/Abilene_normal.csv',
    )
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--scale', type=float, default=1,
                        help='Scale factor used to normalize the raw demands')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    args = parser.parse_args()

    # Load raw CSV (each row is one flattened TM, diagonal already excluded or not)
    tms = np.loadtxt(args.tm_filepath, delimiter=',', dtype=np.float32)
    print(f'Loaded {len(tms)} TMs, {tms.shape[1]} OD pairs each')

    # Remove diagonal entries (self-loops): for n nodes, n*(n-1) off-diag entries
    # The CSV already has diagonal removed (see TEDataset tm_mask), so no extra step needed.

    # Normalize (same as TEDataset)
    tms = tms / args.scale

    # Build tm_preds: the prediction target for each window (same as TEDataset)
    tm_preds = tms[args.window_size:]           # shape: (N - window_size, OD)
    total = len(tm_preds)

    # Test split: last test_ratio fraction
    test_start = total - int(args.test_ratio * total)
    test_preds = tm_preds[test_start:]          # shape: (n_test, OD)
    print(f'Test set size: {len(test_preds)} samples (indices {test_start}..{total-1})')

    # Sum of all OD demands per TM
    demand_sums = test_preds.sum(axis=1)        # shape: (n_test,)

    print(f'\nNormalized (divided by scale={args.scale:.2e}):')
    print(f'  min_sum = {demand_sums.min():.6f}')
    print(f'  max_sum = {demand_sums.max():.6f}')
    print(f'  mean_sum = {demand_sums.mean():.6f}')

    print(f'\nAbsolute (raw) values (multiplied back by scale):')
    abs_sums = demand_sums * args.scale
    print(f'  min_sum = {abs_sums.min():.4f}')
    print(f'  max_sum = {abs_sums.max():.4f}')
    print(f'  mean_sum = {abs_sums.mean():.4f}')


if __name__ == '__main__':
    main()
