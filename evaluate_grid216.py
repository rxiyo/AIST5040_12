"""Evaluate a 6x6x6 grid on the exact saved three-parameter test dataset."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from acu_benchmark import ACUSimulator, BOUNDS, BUDGET_GRID_SIDE, summarize
from physics import MASKS, score_grid, visible_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(__file__).parent/'outputs'/'acu_original_scale_seed42')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'outputs'/'acu_grid216_seed42')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.run/'dataset.npz')
    with (args.run/'run.json').open() as f:
        config = json.load(f)
    if not np.allclose(config['bounds'], BOUNDS):
        raise ValueError('Source run uses different parameter bounds')
    truth = data['test_parameters']
    sim = ACUSimulator()
    shape = [BUDGET_GRID_SIDE] * len(BOUNDS)
    axes = [np.linspace(lo, hi, side) for (lo, hi), side in zip(BOUNDS, shape)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        templates = sim.profiles(grid)
        build_seconds = time.perf_counter() - start
        metrics = []
        predictions = {}
        for name, mask in MASKS.items():
            observations = data[f'test_{name}']
            start = time.perf_counter()
            errors = score_grid(observations, visible_profiles(templates, mask), mask)
            order = np.argsort(errors, axis=1)[:, :5]
            elapsed = time.perf_counter() - start
            candidates = grid[order]
            best = np.take_along_axis(errors, order[:, :1], axis=1).mean()
            metrics.append(summarize(name, 'grid_216', candidates, truth, float(best), elapsed))
            predictions[name] = candidates
    with (args.output/'grid216_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    np.savez_compressed(args.output/'grid216_predictions.npz', **predictions)
    (args.output/'run.json').write_text(json.dumps({
        'source_run': str(args.run.resolve()), 'grid_shape': shape, 'candidate_count': len(grid),
        'test_cells': len(truth), 'grid_precompute_seconds': build_seconds,
        'notes': 'Same saved held-out spectra. Candidates ranked by visible profile MSE; no deduplication.'
    }, indent=2))
    print('Saved 216-point grid results to', args.output.resolve())


if __name__ == '__main__': main()
