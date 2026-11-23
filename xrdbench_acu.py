"""Use the project MDN/CVAE as raw-posterior adapters for xrdbench rutile."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from xrdbench.api import make_batch, make_context
from xrdbench.io import file_hash, iter_observations, load_targets, manifest, write_json
from xrdbench.protocol import SAMPLES, WAVELENGTH

from acu_benchmark import CENTER, HALF, CVAE, train_cvae
from models import ConditionalMixture
from physics import ANGLES


POOL_BINS = 256
VALIDATION_SAMPLES = 32
VALIDATION_EVERY = 5


def check_context(context):
    if context.task != "rutile" or context.parameter_names != ("a", "c"):
        raise ValueError("This adapter expects the xrdbench rutile (a, c) task")
    if not np.isclose(context.wavelength_angstrom, WAVELENGTH):
        raise ValueError("Unexpected XRD wavelength")


def bench_features(batch):
    n = len(batch.ids)
    mask = batch.mask.reshape(n, POOL_BINS, -1)
    counts = batch.counts.reshape(mask.shape) * mask
    exposure = np.log1p(batch.exposure[:, None])
    intensity = np.log1p(counts.sum(axis=-1)) / exposure
    return np.column_stack((intensity, mask.mean(axis=-1), exposure))


def load_split(public, split):
    blocks = [(arrays["ids"], bench_features(make_batch(arrays)))
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


def cvae_validation(model, x, y):
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
    x, y = load_split(args.public, "train")
    vx, vy = load_split(args.public, "val")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.01)
    x, vx = (x - mean) / scale, (vx - mean) / scale
    center = (context.upper + context.lower) / 2
    half = (context.upper - context.lower) / 2
    y, vy = (y - center) / half, (vy - center) / half
    start = perf_counter()
    if args.method == "mdn":
        model = ConditionalMixture(x.shape[1], center=np.zeros(len(center)), half_range=np.ones(len(center)))

        def mdn_validation(weights):
            samples = model.sample(vx, VALIDATION_SAMPLES, np.random.default_rng(251), weights=weights)
            return energy(samples, vy)

        info = model.fit(x, y, vx, vy, np.random.default_rng(args.seed), args.mdn_iterations,
                         validation_score=mdn_validation, validation_every=VALIDATION_EVERY)
        parameters = {"weights": model.weights}
        info["trainable_parameters"] = len(model.weights)
    else:
        model, history, best_epoch = train_cvae(
            x, y, vx, vy, args, validation_score=cvae_validation, validation_every=VALIDATION_EVERY
        )
        with (args.output / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        parameters = {"state_dict": model.state_dict(), "hidden": args.hidden, "latent": args.latent}
        info = {"best_epoch": best_epoch, "validation": history[best_epoch - 1],
                "trainable_parameters": sum(p.numel() for p in model.parameters())}
    info["training_wall_seconds"] = perf_counter() - start
    torch.save({"dataset_id": context.dataset_id, "method": args.method, "mean": mean, "scale": scale,
                "center": center, "half": half, **parameters}, args.output / "model.pt")
    name = "AIST MDN" if args.method == "mdn" else (
        f"AIST CVAE KL warmup {args.kl_warmup_epochs}" if args.kl_warmup_epochs else "AIST CVAE fixed KL"
    )
    fields = ("epochs", "mdn_iterations", "batch_size", "hidden", "latent", "lr", "beta", "kl_warmup_epochs")
    write_json(args.output / "metadata.json", {
        "method_name": name, "implementation": "AIST5040_12 architecture retrained for rutile (a,c)",
        "training_dataset_id": context.dataset_id, "training_suite": meta["config"]["suite"],
        "training_seed": args.seed, "unique_train_structures": meta["splits"]["train"]["structure_count"],
        "train_observations": len(x), "extra_simulations": 0, "external_pretraining": False,
        "postprocessing": "none", "hardware": "local CPU, one training thread",
        "checkpoint_selection": "validation Energy, 32 fixed-seed draws per observation, evaluated every 5 steps",
        "hyperparameters": {key: getattr(args, key) for key in fields},
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
        self.method = self.state["method"]
        self.model = load_model(self.state, len(self.state["mean"]), self.method, len(self.center))

    def synchronize(self):
        pass

    def predict(self, batch, n_samples, seeds):
        x = (bench_features(batch) - self.state["mean"]) / self.state["scale"]
        return {"samples": self.center + self.half * draw_rows(self.model, x, self.method, n_samples, seeds)}


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


def load_model(state, inputs, method, dim):
    if method == "mdn":
        model = ConditionalMixture(inputs, center=np.zeros(dim), half_range=np.ones(dim))
        model.weights = state["weights"]
    else:
        model = CVAE(inputs, state["hidden"], state["latent"], target_dim=dim)
        model.load_state_dict(state["state_dict"])
        model.eval()
    return model


def draw_rows(model, x, method, count, seeds):
    if method == "mdn":
        return np.concatenate([model.sample(row, count, np.random.default_rng(int(seed)))
                               for row, seed in zip(x[:, None], seeds)])
    with torch.inference_mode():
        rows = torch.as_tensor(x[:, None], dtype=torch.float32)
        return np.concatenate([model.sample(row, count, torch.Generator().manual_seed(int(seed))).numpy()
                               for row, seed in zip(rows, seeds)])


class FrozenAdapter:
    """Project the old (a, c, u) posterior onto (a, c) without training."""

    def __init__(self, context, checkpoint):
        check_context(context)
        checkpoint = Path(checkpoint)
        with np.load(checkpoint.parent / "feature_stats.npz") as stats:
            self.mean, self.scale = stats["mean"], stats["scale"]
        self.method = "mdn" if checkpoint.suffix == ".npz" else "cvae"
        if self.method == "mdn":
            with np.load(checkpoint) as state:
                self.model = load_model(state, len(self.mean), self.method, 3)
            self.center, self.half = CENTER, HALF
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.model = load_model(state, len(self.mean), self.method, 3)
            self.center, self.half = state["center"], state["half_range"]
        axis = context.two_theta_deg
        self.left = np.searchsorted(axis, ANGLES, side="right") - 1
        self.right = self.left + 1
        self.fraction = (ANGLES - axis[self.left]) / (axis[self.right] - axis[self.left])

    def synchronize(self):
        pass

    def predict(self, batch, n_samples, seeds):
        visible = batch.mask[:, self.left] & batch.mask[:, self.right]
        profile = batch.counts[:, self.left] * (1 - self.fraction) + batch.counts[:, self.right] * self.fraction
        profile *= visible
        profile /= np.maximum(profile.max(axis=1, keepdims=True), 1)
        x = (np.column_stack((profile[:, ::2], visible[:, ::25])) - self.mean) / self.scale
        samples = draw_rows(self.model, x, self.method, n_samples, seeds)
        return {"samples": (self.center + self.half * samples)[:, :, :2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("mdn", "cvae"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--mdn-iterations", type=int, default=260)
    parser.add_argument("--kl-warmup-epochs", type=int, default=0)
    parser.set_defaults(hidden=64, latent=4, batch_size=64, lr=0.001, beta=1.0)
    args = parser.parse_args()
    if min(args.epochs, args.mdn_iterations) < 1 or args.seed < 0:
        parser.error("Use positive iteration counts and a nonnegative seed")
    if not 0 <= args.kl_warmup_epochs <= args.epochs:
        parser.error("KL warm-up must be between 0 and the number of epochs")
    if args.method == "mdn" and args.kl_warmup_epochs:
        parser.error("KL warm-up only applies to CVAE")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        train(args)


if __name__ == "__main__":
    main()
