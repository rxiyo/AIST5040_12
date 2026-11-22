import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, logsumexp, ndtr, ndtri

from physics import CENTER, HALF_RANGE, MASKS


class SpectrumFeatures:
    """Shared input transform for both learned methods, fitted on training spectra."""

    def __init__(self, spectra):
        raw = self.raw(spectra)
        self.mean = raw.mean(axis=0)
        self.scale = np.maximum(raw.std(axis=0), 0.04)

    @staticmethod
    def raw(spectra):
        features = []
        for name, observations in spectra.items():
            sampled_mask = MASKS[name][::25]
            mask_features = np.broadcast_to(sampled_mask, (len(observations), sampled_mask.size))
            features.append(np.column_stack((observations[:, ::2], mask_features)))
        return np.concatenate(features)

    def transform(self, spectra):
        return (self.raw(spectra) - self.mean) / self.scale


def normalized_targets(parameters):
    return np.tile((parameters - CENTER) / HALF_RANGE, (len(MASKS), 1))


class ConditionalMixture:
    """A small neural density estimator with diagonal Gaussian mixture components."""

    def __init__(self, inputs, hidden=32, components=3):
        self.inputs = inputs
        self.hidden = hidden
        self.components = components
        self.output_size = 5 * components
        self.shapes = [(inputs, hidden), (hidden,), (hidden, self.output_size), (self.output_size,)]
        self.splits = np.cumsum([np.prod(shape) for shape in self.shapes])[:-1]

    def parameters(self, weights):
        return [part.reshape(shape) for part, shape in zip(np.split(weights, self.splits), self.shapes)]

    def forward(self, x, weights):
        w1, b1, w2, b2 = self.parameters(weights)
        hidden = np.tanh(x @ w1 + b1)
        out = (hidden @ w2 + b2).reshape(-1, self.components, 5)
        log_prob = log_softmax(out[:, :, 0], axis=1)
        means = np.tanh(out[:, :, 1:3])
        scale_gate = expit(out[:, :, 3:5])
        log_sd = -4.6 + 5 * scale_gate
        return hidden, log_prob, means, scale_gate, log_sd

    def objective(self, weights, x, y, gradient=False):
        hidden, log_prob, means, scale_gate, log_sd = self.forward(x, weights)
        sd = np.exp(log_sd)
        z = (y[:, None, :] - means) / sd
        joint = log_prob - (log_sd + 0.5 * z**2).sum(axis=2) - np.log(2 * np.pi)
        marginal = logsumexp(joint, axis=1)
        loss = -marginal.mean()
        if not gradient:
            return loss

        responsibilities = np.exp(joint - marginal[:, None])
        r = responsibilities[:, :, None]
        d_logits = np.exp(log_prob) - responsibilities
        d_mean = -r * z / sd * (1 - means**2)
        d_scale = r * (1 - z**2) * 5 * scale_gate * (1 - scale_gate)
        d_out = np.concatenate((d_logits[:, :, None], d_mean, d_scale), axis=2)
        d_out = d_out.reshape(-1, self.output_size) / len(x)
        _, _, w2, _ = self.parameters(weights)
        d_hidden = (d_out @ w2.T) * (1 - hidden**2)
        grads = [x.T @ d_hidden, d_hidden.sum(axis=0), hidden.T @ d_out, d_out.sum(axis=0)]
        regularization = 1e-5
        return (
            loss + regularization * np.dot(weights, weights),
            np.concatenate([g.ravel() for g in grads]) + 2 * regularization * weights,
        )

    def fit(self, x, y, validation_x, validation_y, rng, iterations):
        w1 = rng.normal(0, 1 / np.sqrt(self.inputs), size=self.shapes[0])
        w2 = rng.normal(0, 0.035, size=self.shapes[2])
        b2 = np.zeros((self.components, 5))
        b2[:, 2] = np.linspace(-0.35, 0.35, self.components)
        b2[:, 3:5] = 1.2
        initial = np.concatenate((w1.ravel(), np.zeros(self.hidden), w2.ravel(), b2.ravel()))
        self.weights = initial
        self.validation_nll = self.objective(initial, validation_x, validation_y)

        def checkpoint(weights):
            loss = self.objective(weights, validation_x, validation_y)
            if loss < self.validation_nll:
                self.weights = weights.copy()
                self.validation_nll = loss

        result = minimize(
            self.objective, initial, args=(x, y, True), method="L-BFGS-B",
            jac=True, callback=checkpoint, options={"maxiter": iterations},
        )
        checkpoint(result.x)
        return {
            "iterations": int(result.nit),
            "termination": str(result.message),
            "validation_nll_normalized": float(self.validation_nll),
        }

    def sample(self, x, count, rng):
        _, log_prob, means, _, log_sd = self.forward(x, self.weights)
        sd = np.exp(log_sd)
        lower_cdf = ndtr((-1 - means) / sd)
        upper_cdf = ndtr((1 - means) / sd)
        component_mass = np.prod(upper_cdf - lower_cdf, axis=2)
        # Condition the predictive mixture on the same parameter box as the grid.
        weights = np.exp(log_prob) * component_mass
        weights /= weights.sum(axis=1, keepdims=True)
        components = (
            rng.random((len(x), count, 1)) > weights.cumsum(axis=1)[:, None, :-1]
        ).sum(axis=2)
        rows = np.arange(len(x))[:, None]
        lower = lower_cdf[rows, components]
        upper = upper_cdf[rows, components]
        quantile = lower + (upper - lower) * rng.random(lower.shape)
        z = ndtri(np.clip(quantile, 1e-15, 1 - 1e-15))
        samples = means[rows, components] + sd[rows, components] * z
        return CENTER + HALF_RANGE * np.clip(samples, -1, 1)
