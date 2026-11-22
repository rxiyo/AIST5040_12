import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics import BOUNDS, CENTER, MASKS, order_by_fit
from run_benchmark import HIT_TOLERANCE, SEPARATION, TOP_K


SETTINGS = {"matched": 0.12, "broadened": 0.20}
METRICS = (
    "mae_a_angstrom", "mae_c_angstrom", "top1_hit_rate", "top5_hit_rate",
    "mean_top1_observed_mse", "online_batch_ms_per_observation",
)
LABELS = ["Full scan", "Low-information", "Equal-width control"]


def read_run(root, setting, seed):
    directory = root / setting / f"seed-{seed}"
    with (directory / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = {(row["condition"], row["method"]): row for row in csv.DictReader(file)}
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if metadata["seed"] != seed or metadata["test_peak_gaussian_sigma_deg"] != SETTINGS[setting]:
        raise ValueError(f"Unexpected seed or test width in {directory}")
    with np.load(directory / "dataset.npz") as data:
        truth = data["test_parameters"]
        # The same seed determines the cell draws and intensity noise in both runs.
        parameters = [data[f"{split}_parameters"] for split in ("train", "validation", "test")]
    return directory, rows, metadata, truth, parameters


def stats(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def write_results(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diverse_at_similar_fit(pool, costs, cutoff):
    """Keep the same best fit and spread remaining candidates inside a residual band."""
    by_fit = order_by_fit(pool, costs)
    chosen = [by_fit[0]]
    remaining = by_fit[1:]
    scales = np.diff(BOUNDS, axis=1).ravel()
    while remaining.size and len(chosen) < TOP_K:
        delta = np.abs(pool[remaining, None, :] - pool[chosen][None, :, :])
        separate = delta.max(axis=2).min(axis=1) >= SEPARATION
        remaining = remaining[separate]
        if not remaining.size:
            break
        plausible = costs[remaining] <= costs[by_fit[0]] + cutoff
        if plausible.any():
            distances = (delta[separate] / scales).max(axis=2).min(axis=1)
            index = np.argmax(np.where(plausible, distances, -np.inf))
        else:
            index = 0
        chosen.append(remaining[index])
        remaining = np.delete(remaining, index)
    selected = np.full((TOP_K, 2), np.nan)
    selected[:len(chosen)] = pool[chosen]
    return selected


def candidate_stats(candidates, truth):
    hit = (np.abs(candidates - truth[:, None, :]) <= HIT_TOLERANCE).all(axis=2).any(axis=1)
    c = candidates[:, :, 1]
    span = np.nanmax(c, axis=1) - np.nanmin(c, axis=1)
    return float(hit.mean()), float(span.mean())


def diagnostics(directory, setting, seed, metadata, truth):
    rows = []
    with np.load(directory / "predictions.npz") as predictions:
        for condition, mask in MASKS.items():
            # Excess sum of squared residuals is capped at six nominal noise variances.
            cutoff = 6 * metadata["noise_sd_relative_intensity"] ** 2 / mask.sum()
            raw = predictions[f"{condition}_mdn"]
            ranked = predictions[f"{condition}_mdn_physics"]
            pools = predictions[f"{condition}_mdn_pool"]
            costs = predictions[f"{condition}_mdn_pool_mse"]
            diverse = np.stack([diverse_at_similar_fit(pool, mse, cutoff) for pool, mse in zip(pools, costs)])
            if not np.array_equal(diverse[:, 0], ranked[:, 0]):
                raise ValueError("The diversity ablation must preserve the original top-ranked candidate.")
            raw_hit, raw_span = candidate_stats(raw, truth)
            physics_hit, physics_span = candidate_stats(ranked, truth)
            diverse_hit, diverse_span = candidate_stats(diverse, truth)
            pool_hit, _ = candidate_stats(pools, truth)
            rows.append({
                "setting": setting, "seed": seed, "condition": condition,
                "c_prior_center_mae": float(np.abs(truth[:, 1] - CENTER[1]).mean()),
                "pool_hit_at_36": pool_hit,
                "raw_top5": raw_hit, "physics_top5": physics_hit, "diverse_top5": diverse_hit,
                "raw_c_span": raw_span, "physics_c_span": physics_span, "diverse_c_span": diverse_span,
            })
    return rows


def make_figure(path, summary, diagnostic_rows):
    index = {(r["setting"], r["condition"], r["method"], r["metric"]): r for r in summary}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    dense_color, mixture_color, rescore_color, diversity_color = "#254764", "#9873aa", "#007d70", "#b86427"
    x = np.arange(len(MASKS))
    width = 0.23
    for i, setting in enumerate(SETTINGS):
        mae_axis, hit_axis = axes[i]
        for offset, (method, label, color) in enumerate([
            ("grid_dense", "Dense grid", dense_color), ("mdn", "MDN", mixture_color),
            ("mdn_physics", "MDN + physics", rescore_color),
        ]):
            rows = [index[setting, condition, method, "mae_c_angstrom"] for condition in MASKS]
            mae_axis.bar(x + (offset - 1) * width, [row["mean"] for row in rows],
                         yerr=[row["sd_across_seeds"] for row in rows], width=width,
                         capsize=3, color=color, label=label)
        budget = [index[setting, condition, "grid_budget", "top5_hit_rate"] for condition in MASKS]
        hit_axis.bar(x - 1.5 * width, [100 * row["mean"] for row in budget],
                     yerr=[100 * row["sd_across_seeds"] for row in budget],
                     width=width, capsize=3, color="#6793aa", label="Budget-matched grid")
        for offset, (column, label, color) in enumerate([
            ("raw_top5", "MDN", mixture_color), ("physics_top5", "MDN + physics", rescore_color),
            ("diverse_top5", "Physics + diversity", diversity_color),
        ]):
            aggregates = [
                stats([100 * row[column] for row in diagnostic_rows
                       if row["setting"] == setting and row["condition"] == condition])
                for condition in MASKS
            ]
            means, errors = zip(*aggregates)
            hit_axis.bar(x + (offset - 0.5) * width, means, yerr=errors, width=width,
                         capsize=3, color=color, label=label)
        condition_title = f"Test peak σ = {SETTINGS[setting]:.2f}°"
        mae_axis.set(ylabel="Top-1 c MAE (Å)", title=condition_title)
        hit_axis.set(ylabel="Top-5 hit rate (%)", ylim=(0, 110), title=condition_title)
        for axis in (mae_axis, hit_axis):
            axis.set(xticks=x, xticklabels=LABELS)
            axis.grid(axis="y", alpha=0.24)
            axis.set_axisbelow(True)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Window and peak-width effects · mean ± SD across seeds", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate paired synthetic PXRD experiments.")
    parser.add_argument("--study", type=Path, default=Path(__file__).parent / "outputs" / "study")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("Use at least two distinct seeds for between-seed standard deviations.")

    runs = {setting: [] for setting in SETTINGS}
    diagnostic_rows = []
    for seed in args.seeds:
        paired_splits = None
        for setting in SETTINGS:
            directory, rows, metadata, truth, parameters = read_run(args.study, setting, seed)
            if paired_splits is not None and any(
                not np.array_equal(original, changed)
                for original, changed in zip(paired_splits, parameters)
            ):
                raise ValueError(f"Peak-width runs do not share the same cells for seed {seed}.")
            paired_splits = parameters
            runs[setting].append(rows)
            diagnostic_rows.extend(diagnostics(directory, setting, seed, metadata, truth))

    summary = []
    for setting, repeats in runs.items():
        for condition, method in repeats[0]:
            for metric in METRICS:
                values = [row[condition, method][metric] for row in repeats]
                if all(value == "" for value in values):
                    continue
                mean, sd = stats(list(map(float, values)))
                summary.append({
                    "setting": setting, "condition": condition, "method": method, "metric": metric,
                    "mean": mean, "sd_across_seeds": sd, "repeats": len(repeats),
                })
    write_results(args.study / "summary.csv", summary)
    write_results(args.study / "candidate_diagnostics.csv", diagnostic_rows)
    paired = []
    for condition, method in runs["matched"][0]:
        for metric in ("mae_a_angstrom", "mae_c_angstrom", "top1_hit_rate", "top5_hit_rate"):
            deltas = [
                float(changed[condition, method][metric]) - float(original[condition, method][metric])
                for original, changed in zip(runs["matched"], runs["broadened"])
                if original[condition, method][metric] != ""
            ]
            if deltas:
                mean, sd = stats(deltas)
                paired.append({
                    "condition": condition, "method": method, "metric": metric,
                    "broadened_minus_matched_mean": mean, "sd_across_paired_seeds": sd,
                    "repeats": len(deltas),
                })
    write_results(args.study / "paired_peak_width_effect.csv", paired)
    make_figure(args.study / "robustness.png", summary, diagnostic_rows)
    print(f"Saved summaries to {args.study.resolve()}")


if __name__ == "__main__":
    main()
