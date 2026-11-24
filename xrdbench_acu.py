"""Use the project MDN/CVAE as raw-posterior adapters for xrdbench rutile."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from xrdbench.api import make_batch, make_context
from xrdbench.io import file_hash, iter_observations, load_targets, manifest, write_json
from xrdbench.protocol import SAMPLES, WAVELENGTH

from acu_benchmark import CVAE, train_cvae


DEFAULT_SPECTRUM_BINS = 256
VALIDATION_SAMPLES = 32
VALIDATION_EVERY = 5


class ConvSpectrumModel(nn.Module):
    """Encode intensity and visibility jointly, retaining coarse angular position."""

    def __init__(self):
        super().__init__()
        self.spectrum = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv1d(16, 24, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv1d(24, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32), nn.SiLU(),
            nn.AdaptiveAvgPool1d(32), nn.Flatten(),
            nn.Linear(32 * 32, 128), nn.SiLU(),
        )
    def encode_spectrum(self, x):
        spectrum = self.spectrum(x[:, :-1].reshape(len(x), 2, -1))
        return torch.cat((spectrum, x[:, -1:]), dim=1)


class ConvSpectrumCVAE(ConvSpectrumModel):
    def __init__(self, inputs, hidden=64, latent=4, target_dim=2):
        super().__init__()
        self.cvae = CVAE(129, hidden=hidden, latent=latent, target_dim=target_dim)

    def loss(self, x, y, beta, generator):
        return self.cvae.loss(self.encode_spectrum(x), y, beta, generator)

    def sample(self, x, count, generator):
        return self.cvae.sample(self.encode_spectrum(x), count, generator)


class ConvSpectrumMDN(ConvSpectrumModel):
    """Train a diagonal Gaussian mixture head jointly with the spectrum encoder."""

    def __init__(self, inputs, hidden=64, latent=4, target_dim=2):
        super().__init__()
        self.components = 3
        self.dim = target_dim
        self.head = nn.Sequential(nn.Linear(129, hidden), nn.Tanh(),
                                  nn.Linear(hidden, self.components * (1 + 2 * self.dim)))

    def mixture(self, x):
        out = self.head(self.encode_spectrum(x)).reshape(-1, self.components, 1 + 2 * self.dim)
        log_weights = torch.log_softmax(out[..., 0], dim=-1)
        means = torch.tanh(out[..., 1:1 + self.dim])
        scales = (-4.6 + 5 * torch.sigmoid(out[..., 1 + self.dim:])).exp()
        return log_weights, means, scales

    def loss(self, x, y, beta, generator):
        log_weights, means, scales = self.mixture(x)
        component_logp = torch.distributions.Normal(means, scales).log_prob(y[:, None]).sum(-1)
        nll = -(log_weights + component_logp).logsumexp(dim=1).mean()
        return nll, nll, nll.new_zeros(())

    def sample(self, x, count, generator):
        log_weights, means, scales = (v.double() for v in self.mixture(x))
        lower = torch.special.ndtr((-1 - means) / scales)
        upper = torch.special.ndtr((1 - means) / scales)
        # Condition the entire mixture on the parameter bounds, including component weights.
        weights = log_weights.exp() * (upper - lower).prod(dim=-1)
        components = torch.multinomial(weights, count, replacement=True, generator=generator)
        rows = torch.arange(len(x), device=x.device)[:, None]
        lo, hi = lower[rows, components], upper[rows, components]
        quantile = lo + (hi - lo) * torch.rand(lo.shape, dtype=lo.dtype, device=x.device, generator=generator)
        samples = means[rows, components] + scales[rows, components] * torch.special.ndtri(
            quantile.clamp(1e-15, 1 - 1e-15))
        return samples.clamp(-1, 1)


POSTERIOR_MODELS = {"mdn": ConvSpectrumMDN, "cvae": ConvSpectrumCVAE}


def check_context(context):
    if context.task != "rutile" or context.parameter_names != ("a", "c"):
        raise ValueError("This adapter expects the xrdbench rutile (a, c) task")
    if not np.isclose(context.wavelength_angstrom, WAVELENGTH):
        raise ValueError("Unexpected XRD wavelength")


def bench_features(batch, bins):
    n = len(batch.ids)
    mask = batch.mask.reshape(n, bins, -1)
    counts = batch.counts.reshape(mask.shape) * mask
    exposure = np.log1p(batch.exposure[:, None])
    intensity = np.log1p(counts.sum(axis=-1)) / exposure
    return np.column_stack((intensity, mask.mean(axis=-1), exposure))


def load_split(public, split, bins):
    blocks = [(arrays["ids"], bench_features(make_batch(arrays), bins))
              for arrays in iter_observations(public, split)]
    labels = load_targets(public, split)
    ids = np.concatenate([ids for ids, _ in blocks])
    if not np.array_equal(ids, labels["ids"]):
        raise ValueError(f"{split} observations and targets are misaligned")
    return np.concatenate([x for _, x in blocks]), labels["theta"]


def energy(samples, truth):
    """Inputs are in [-1, 1]; xrdbench measures energy in unit prior widths."""
    count = samples.shape[1] // 2
    error = np.linalg.norm(samples - truth[:, None, :], axis=-1).mean()
    spread = np.linalg.norm(samples[:, :count] - samples[:, count:], axis=-1).mean()
    return (error - spread / 2) / 2


def posterior_validation(model, x, y):
    generator = torch.Generator().manual_seed(251)
    samples = np.concatenate([model.sample(xb, VALIDATION_SAMPLES, generator).numpy()
                              for xb in x.split(64)])
    return energy(samples, y.numpy())


def train(args):
    meta = manifest(args.public)
    context = make_context(meta)
    check_context(context)
    if context.track != "blind":
        raise ValueError("This training recipe uses the blind measurement inputs")
    args.output.mkdir(parents=True, exist_ok=False)
    if len(context.two_theta_deg) % args.spectrum_bins:
        raise ValueError("Spectrum bin count must divide the input angle count")
    x, y = load_split(args.public, "train", args.spectrum_bins)
    vx, vy = load_split(args.public, "val", args.spectrum_bins)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.01)
    x, vx = (x - mean) / scale, (vx - mean) / scale
    center = (context.upper + context.lower) / 2
    half = (context.upper - context.lower) / 2
    y, vy = (y - center) / half, (vy - center) / half
    start = perf_counter()
    model, history, best_epoch = train_cvae(
        x, y, vx, vy, args, validation_score=posterior_validation, validation_every=VALIDATION_EVERY,
        model_class=POSTERIOR_MODELS[args.method],
    )
    with (args.output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    parameters = {"state_dict": model.state_dict(), "hidden": args.hidden,
                  "latent": args.latent, "encoder": "conv"}
    info = {"best_epoch": best_epoch, "validation": history[best_epoch - 1],
            "trainable_parameters": sum(p.numel() for p in model.parameters())}
    info["training_wall_seconds"] = perf_counter() - start
    torch.save({"dataset_id": context.dataset_id, "method": args.method, "mean": mean, "scale": scale,
                "spectrum_bins": args.spectrum_bins,
                "center": center, "half": half, **parameters}, args.output / "model.pt")
    name = "AIST MDN" if args.method == "mdn" else (
        f"AIST CVAE KL warmup {args.kl_warmup_epochs}" if args.kl_warmup_epochs else "AIST CVAE fixed KL"
    )
    name += " + Conv1D"
    fields = ("epochs", "batch_size", "hidden", "latent", "lr", "beta",
              "kl_warmup_epochs", "spectrum_bins")
    write_json(args.output / "metadata.json", {
        "method_name": name, "implementation": "AIST5040_12 architecture retrained for rutile (a,c)",
        "training_dataset_id": context.dataset_id, "training_suite": meta["config"]["suite"],
        "training_seed": args.seed, "unique_train_structures": meta["splits"]["train"]["structure_count"],
        "train_observations": len(x), "extra_simulations": 0, "external_pretraining": False,
        "postprocessing": "none", "hardware": "local CPU, one training thread",
        "checkpoint_selection": "validation Energy, 32 fixed-seed draws per observation, evaluated every 5 steps",
        "hyperparameters": {"encoder": "conv", **{key: getattr(args, key) for key in fields}},
        "source_sha256": {filename: file_hash(Path(__file__).with_name(filename))
                          for filename in ("xrdbench_acu.py", "models.py", "acu_benchmark.py")},
        **info,
    })
    print(f"{name}: {info}", flush=True)


class Adapter:
    """Posterior adapter for jointly trained, two-parameter checkpoints."""

    def __init__(self, context, checkpoint):
        check_context(context)
        self.state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if self.state["dataset_id"] != context.dataset_id:
            raise ValueError("Checkpoint belongs to a different xrdbench release")
        self.center, self.half = self.state["center"], self.state["half"]
        self.model = load_model(self.state, len(self.center))
        self.bins = self.state["spectrum_bins"]

    def synchronize(self):
        pass

    def predict(self, batch, n_samples, seeds):
        x = (bench_features(batch, self.bins) - self.state["mean"]) / self.state["scale"]
        return {"samples": self.center + self.half * draw_rows(self.model, x, n_samples, seeds)}


class PointAdapter(Adapter):
    """Submit the same 256-draw marginal median used by posterior scoring."""

    def predict(self, batch, n_samples, seeds):
        samples = super().predict(batch, SAMPLES, seeds)["samples"]
        return {"point": np.median(samples, axis=1)}


class CandidateAdapter(Adapter):
    """Order the requested samples by normalized distance to their marginal median."""

    def predict(self, batch, n_samples, seeds):
        samples = super().predict(batch, n_samples, seeds)["samples"]
        median = np.median(samples, axis=1, keepdims=True)
        distance = np.abs(samples - median) / self.half
        order = np.argsort(distance.sum(axis=-1), axis=1, kind="stable")
        candidates = np.take_along_axis(samples, order[:, :, None], axis=1)
        return {"point": candidates[:, 0], "candidates": candidates}


def load_model(state, dim):
    if state.get("encoder") != "conv":
        raise ValueError("This adapter requires a Conv1D checkpoint. Retrain with xrdbench_acu.")
    model = POSTERIOR_MODELS[state["method"]](len(state["mean"]), state["hidden"],
                                              state["latent"], target_dim=dim)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def draw_rows(model, x, count, seeds):
    with torch.inference_mode():
        rows = torch.as_tensor(x[:, None], dtype=torch.float32)
        return np.concatenate([model.sample(row, count, torch.Generator().manual_seed(int(seed))).numpy()
                               for row, seed in zip(rows, seeds)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("mdn", "cvae"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--kl-warmup-epochs", type=int, default=0)
    parser.add_argument("--spectrum-bins", type=int, default=DEFAULT_SPECTRUM_BINS,
                        help="Spectrum bins (default: %(default)s)")
    parser.set_defaults(hidden=64, latent=4, batch_size=64, lr=0.001, beta=1.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("Use positive epoch counts and a nonnegative seed")
    if not 0 <= args.kl_warmup_epochs <= args.epochs:
        parser.error("KL warm-up must be between 0 and the number of epochs")
    if args.method == "mdn" and args.kl_warmup_epochs:
        parser.error("KL warm-up only applies to CVAE")
    if args.method == "mdn":
        args.beta = 0.0
    if args.spectrum_bins < 256:
        parser.error("Use at least 256 spectrum bins with the convolutional encoder")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        train(args)


if __name__ == "__main__":
    main()
