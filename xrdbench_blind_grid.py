"""Approximate blind rutile posterior over a lattice grid and public nuisance prior."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from xrdbench.data import observation_mask
from xrdbench.physics import expected_counts, peaks
from xrdbench.protocol import BLIND_PRIOR, HARD_PRIOR, conditions_for


GRID_SIDE = 21
NUISANCE_LOG2_DRAWS = 5


def nuisance_draws(prior):
    unit = qmc.Sobol(d=len(prior), scramble=True, seed=42).random_base2(NUISANCE_LOG2_DRAWS)
    draws = []
    for values, spec in zip(unit.T, prior.values()):
        lo, hi = spec["range"]
        if spec["distribution"] == "log_uniform":
            draws.append(np.exp(np.log(lo) + values * (np.log(hi) - np.log(lo))))
        else:
            draws.append(lo + values * (hi - lo))
    return np.column_stack(draws)


class Adapter:
    """Marginalize likelihood over the same fixed nuisance draws at every grid cell."""

    def __init__(self, context, checkpoint=None):
        if context.task != "rutile" or context.track != "blind" or context.parameter_names != ("a", "c"):
            raise ValueError("This diagnostic expects the blind rutile (a, c) union task")
        fraction = (np.arange(GRID_SIDE) + 0.5) / GRID_SIDE
        axes = [lo + fraction * (hi - lo) for lo, hi in zip(context.lower, context.upper)]
        self.grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
        hard = conditions_for("rutile", "v0.2")
        # Identify the public measurement design from its mask, never from private condition labels.
        self.hard_masks = np.stack([observation_mask(name, 0, hard[name])
                                    for name in ("hard_mixed_e100", "hard_paired_e100")])
        peak_bank = [peaks(context.task, theta) for theta in self.grid]
        self.templates = []
        for prior in (BLIND_PRIOR, HARD_PRIOR):
            nuisance = nuisance_draws(prior)
            rates = np.empty((len(self.grid) * len(nuisance), len(context.two_theta_deg)), dtype=np.float32)
            for i, (theta, pattern) in enumerate(zip(self.grid, peak_bank)):
                for j, values in enumerate(nuisance):
                    rates[i * len(nuisance) + j] = expected_counts(
                        context.task, theta, values, exposure=1.0, peak_data=pattern
                    )
            self.templates.append((rates.T, np.log(rates).T))

    def synchronize(self):
        pass

    def predict(self, batch, n_samples, seeds):
        hard = (batch.mask[:, None, :] == self.hard_masks).all(axis=-1).any(axis=1)
        samples = np.empty((len(batch.ids), n_samples, 2))
        for is_hard, (rates, log_rates) in enumerate(self.templates):
            rows = np.flatnonzero(hard == bool(is_hard))
            if not len(rows):
                continue
            mask = batch.mask[rows].astype(np.float32)
            counts = batch.counts[rows].astype(np.float32) * mask
            exposure = batch.exposure[rows, None].astype(np.float32)
            with threadpool_limits(limits=1, user_api="blas"):
                log_likelihood = counts @ log_rates - exposure * (mask @ rates)
            # The omitted sum(counts * log(exposure)) is common to every hypothesis.
            log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
            log_weight = logsumexp(log_likelihood.reshape(len(rows), len(self.grid), -1), axis=-1)
            prob = np.exp(log_weight - logsumexp(log_weight, axis=1, keepdims=True)).astype(np.float64)
            prob /= prob.sum(axis=1, keepdims=True)
            for row, weights in zip(rows, prob):
                indexes = np.random.default_rng(int(seeds[row])).choice(
                    len(self.grid), size=n_samples, p=weights
                )
                samples[row] = self.grid[indexes]
        return {"samples": samples}
