import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from models import ConditionalMixture, SpectrumFeatures, normalized_targets
from physics import (
    ANGLES, BOUNDS, CENTER, HALF_RANGE, MASKS, PEAK_SIGMA, WAVELENGTH, WINDOWS,
    RutileDiffraction, diverse_indices, make_split, parameter_grid, rank_candidates,
    score_grid, visible_profiles,
)
from plots import case_figure, metrics_figure


TOP_K = 5
HIT_TOLERANCE = 0.02  # Each parameter must be within this many angstroms.
SEPARATION = 0.01  # Minimum L-infinity distance between two output candidates.


def pack_candidates(pool, orders):
    result = np.full((len(orders), TOP_K, 2), np.nan)
    for row, (parameters, order) in enumerate(zip(pool, orders)):
        result[row, :len(order)] = parameters[order]
    return result


def summarize(condition, method, candidates, truth, top1_mse, scores_per_sample, seconds):
    difference = np.abs(candidates - truth[:, None, :])
    hit = (difference <= HIT_TOLERANCE).all(axis=2)
    return {
        "condition": condition,
        "method": method,
        "test_size": len(truth),
        "mae_a_angstrom": float(difference[:, 0, 0].mean()),
        "mae_c_angstrom": float(difference[:, 0, 1].mean()),
        "top1_hit_rate": float(hit[:, 0].mean()),
        "top5_hit_rate": float(hit.any(axis=1).mean()) if candidates.shape[1] >= TOP_K else None,
        "mean_top1_observed_mse": float(top1_mse.mean()),
        "mean_returned_candidates": float(np.isfinite(candidates[:, :, 0]).sum(axis=1).mean()),
        "physics_scores_per_observation": scores_per_sample,
        "online_batch_ms_per_observation": 1e3 * seconds / len(truth),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    simulator = RutileDiffraction()
    seed_sequences = np.random.SeedSequence(args.seed).spawn(5)
    train_rng, validation_rng, test_rng, model_rng, proposal_rng = map(np.random.default_rng, seed_sequences)
    print("Simulating disjoint train, validation and test cells...", flush=True)
    train_y, train = make_split(simulator, train_rng, args.train, args.noise)
    validation_y, validation = make_split(simulator, validation_rng, args.validation, args.noise)
    test_y, test = make_split(simulator, test_rng, args.test, args.noise, args.test_peak_sigma)
    dataset = {"angles": ANGLES, **{f"mask_{name}": mask for name, mask in MASKS.items()}}
    for prefix, targets, spectra in [
        ("train", train_y, train), ("validation", validation_y, validation), ("test", test_y, test),
    ]:
        dataset[f"{prefix}_parameters"] = targets
        dataset.update({f"{prefix}_{name}": intensity for name, intensity in spectra.items()})
    np.savez_compressed(args.output / "dataset.npz", **dataset)

    print("Training point MLP and conditional mixture...", flush=True)
    features = SpectrumFeatures(train)
    train_x = features.transform(train)
    validation_x = features.transform(validation)
    training_targets = normalized_targets(train_y)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="tanh", max_iter=220,
        learning_rate_init=0.003, tol=1e-5, n_iter_no_change=18, random_state=args.seed,
    )
    start = perf_counter()
    mlp.fit(train_x, training_targets)
    mlp_training_seconds = perf_counter() - start
    mixture = ConditionalMixture(train_x.shape[1])
    start = perf_counter()
    training = mixture.fit(
        train_x, training_targets, validation_x, normalized_targets(validation_y), model_rng, args.iterations,
    )
    training["elapsed_seconds"] = perf_counter() - start

    grids = {"grid_dense": parameter_grid(args.grid_side), "grid_budget": parameter_grid(args.budget_side)}
    library, build_seconds = {}, {}
    for method, grid in grids.items():
        print(f"Building {method} bank ({len(grid)} templates)...", flush=True)
        start = perf_counter()
        library[method] = simulator.profiles(grid)
        build_seconds[method] = perf_counter() - start

    metrics, dense_losses, proposal_pools, reranked = [], {}, {}, {}
    predictions = {}
    budget = len(grids["grid_budget"])
    for name, observations in test.items():
        print(f"Evaluating {name} on {len(test_y)} spectra...", flush=True)
        mask = MASKS[name]
        for method, grid in grids.items():
            templates = visible_profiles(library[method], mask)
            start = perf_counter()
            errors = score_grid(observations, templates, mask)
            orders = [rank_candidates(grid, row, TOP_K, SEPARATION) for row in errors]
            candidates = pack_candidates(np.broadcast_to(grid, (len(test_y), *grid.shape)), orders)
            elapsed = perf_counter() - start
            best_mse = np.array([row[order[0]] for row, order in zip(errors, orders)])
            metrics.append(summarize(name, method, candidates, test_y, best_mse, len(grid), elapsed))
            predictions[f"{name}_{method}"] = candidates
            if method == "grid_dense":
                dense_losses[name] = errors

        spectrum = {name: observations}
        start = perf_counter()
        x = features.transform(spectrum)
        point = CENTER + HALF_RANGE * np.clip(mlp.predict(x), -1, 1)
        elapsed = perf_counter() - start
        point_visible = visible_profiles(simulator.profiles(point), mask)
        point_mse = np.mean((point_visible - observations[:, mask]) ** 2, axis=1)
        predictions[f"{name}_mlp"] = point[:, None, :]
        metrics.append(summarize(name, "mlp", point[:, None, :], test_y, point_mse, 0, elapsed))

        start = perf_counter()
        pool = mixture.sample(features.transform(spectrum), budget, proposal_rng)
        proposal_seconds = perf_counter() - start
        start = perf_counter()
        raw_orders = [diverse_indices(sample, range(budget), TOP_K, SEPARATION) for sample in pool]
        raw = pack_candidates(pool, raw_orders)
        raw_seconds = perf_counter() - start

        start = perf_counter()
        profiles = simulator.profiles(pool.reshape(-1, 2))
        visible = visible_profiles(profiles, mask).reshape(len(test_y), budget, -1)
        errors = np.mean((visible - observations[:, None, mask]) ** 2, axis=2)
        orders = [
            rank_candidates(candidates, mse, TOP_K, SEPARATION) for candidates, mse in zip(pool, errors)
        ]
        ranked = pack_candidates(pool, orders)
        physics_seconds = perf_counter() - start
        raw_mse = np.array([row[order[0]] for row, order in zip(errors, raw_orders)])
        ranked_mse = np.array([row[order[0]] for row, order in zip(errors, orders)])
        metrics.append(summarize(name, "mdn", raw, test_y, raw_mse, 0, proposal_seconds + raw_seconds))
        metrics.append(summarize(
            name, "mdn_physics", ranked, test_y, ranked_mse, budget, proposal_seconds + physics_seconds,
        ))
        predictions[f"{name}_mdn"] = raw
        predictions[f"{name}_mdn_physics"] = ranked
        predictions[f"{name}_mdn_pool"] = pool
        predictions[f"{name}_mdn_pool_mse"] = errors
        proposal_pools[name], reranked[name] = pool, ranked

    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    metadata = {
        "seed": args.seed,
        "base_cells_per_split": [args.train, args.validation, args.test],
        "noise_sd_relative_intensity": args.noise,
        "wavelength_angstrom": WAVELENGTH,
        "peak_gaussian_sigma_deg": PEAK_SIGMA,
        "test_peak_gaussian_sigma_deg": args.test_peak_sigma,
        "angle_step_deg": round(float(ANGLES[1] - ANGLES[0]), 8),
        "parameter_bounds_angstrom": BOUNDS.tolist(),
        "windows_deg": WINDOWS,
        "hit_tolerance_angstrom": HIT_TOLERANCE,
        "separation_angstrom": SEPARATION,
        "proposal_budget": budget,
        "grid_sides": {"dense": args.grid_side, "budget_matched": args.budget_side},
        "offline_template_seconds": build_seconds,
        "mlp_training_seconds": mlp_training_seconds,
        "mixture_training": training,
        "figure_test_index": 0,
        "dependencies": {
            name: version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "pymatgen", "pymatgen-core")
        },
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    case_figure(
        args.output / "example.png", test, test_y[0], 0, grids["grid_dense"],
        dense_losses, proposal_pools, reranked,
    )
    metrics_figure(args.output / "comparison.png", metrics)
    print(f"Saved benchmark to {args.output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark lattice inference on simulated rutile PXRD.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=1800)
    parser.add_argument("--validation", type=int, default=240)
    parser.add_argument("--test", type=int, default=160)
    parser.add_argument("--noise", type=float, default=0.01, help="Noise SD relative to strongest visible peak")
    parser.add_argument(
        "--test-peak-sigma", type=float, default=PEAK_SIGMA,
        help="Test-only Gaussian peak width (degrees); training and fitting use the nominal width",
    )
    parser.add_argument("--grid-side", type=int, default=41)
    parser.add_argument("--budget-side", type=int, default=6, help="Square root of the matched proposal budget")
    parser.add_argument("--iterations", type=int, default=260, help="Maximum mixture optimizer iterations")
    args = parser.parse_args()
    if min(args.train, args.validation, args.test, args.iterations) < 1 or args.seed < 0:
        parser.error("Split sizes and iterations must be positive and seed must be nonnegative.")
    if not (3 <= args.budget_side <= args.grid_side) or not np.isfinite(args.noise) or args.noise < 0:
        parser.error("Require 3 <= budget-side <= grid-side and finite, nonnegative noise.")
    if not np.isfinite(args.test_peak_sigma) or args.test_peak_sigma <= 0:
        parser.error("Test peak sigma must be finite and positive.")
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
