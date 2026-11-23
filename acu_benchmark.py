"""CPU-first, simulated rutile (a,c,u) benchmark; separate from the original study."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pymatgen.core import Lattice, Structure
from threadpoolctl import threadpool_limits

from models import ConditionalMixture, SpectrumFeatures
from physics import ANGLES, MASKS, WINDOWS, RutileDiffraction, score_grid, visible_profiles

# Illustrative synthetic prior, not a claim that every sampled structure is stable.
BOUNDS = np.array([[4.50, 4.69], [2.90, 3.02], [0.2953, 0.3153]])
CENTER = BOUNDS.mean(axis=1)
HALF = np.diff(BOUNDS, axis=1).ravel() / 2
TOL = np.array([0.02, 0.02, 0.002])  # Angstrom, Angstrom, dimensionless


class ACUSimulator(RutileDiffraction):
    def pattern(self, a, c, u):
        coords = [[0, 0, 0], [.5, .5, .5], [u, u, 0], [1-u, 1-u, 0],
                  [.5+u, .5-u, .5], [.5-u, .5+u, .5]]
        structure = Structure(Lattice.tetragonal(a, c), ['Ti', 'Ti', 'O', 'O', 'O', 'O'], coords)
        return self.calculator.get_pattern(structure, scaled=False, two_theta_range=(19, 71))

    def profiles(self, parameters, peak_sigma=.12):
        profiles = np.empty((len(parameters), len(ANGLES)))
        for i, (a, c, u) in enumerate(parameters):
            pattern = self.pattern(a, c, u)
            distance = (ANGLES[None, :] - pattern.x[:, None]) / peak_sigma
            profiles[i] = (pattern.y[:, None] * np.exp(-.5 * distance**2)).sum(axis=0)
        return profiles


def split(sim, rng, size, noise):
    y = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (size, 3))
    clean = sim.profiles(y)
    paired_noise = rng.normal(0, noise, clean.shape)
    spectra = {}
    for name, mask in MASKS.items():
        obs = np.zeros_like(clean)
        obs[:, mask] = visible_profiles(clean, mask) + paired_noise[:, mask]
        spectra[name] = obs
    return y, spectra


class CVAE(nn.Module):
    def __init__(self, inputs, hidden=64, latent=4):
        super().__init__()
        self.latent = latent
        self.encoder = nn.Sequential(nn.Linear(inputs + 3, hidden), nn.ReLU(), nn.Linear(hidden, 2 * latent))
        self.decoder = nn.Sequential(nn.Linear(inputs + latent, hidden), nn.ReLU(), nn.Linear(hidden, 3), nn.Tanh())

    def loss(self, x, y, beta, generator):
        mu, logvar = self.encoder(torch.cat([x, y], dim=-1)).chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 6)
        z = mu + torch.exp(.5 * logvar) * torch.randn(mu.shape, generator=generator, device=x.device)
        recon = self.decoder(torch.cat([x, z], dim=-1))
        # Fixed Gaussian reconstruction scale in normalized parameter coordinates.
        reconstruction = (((recon-y) / .2)**2 / 2).sum(dim=1).mean()
        kl = (-.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)).mean()
        return reconstruction + beta * kl, reconstruction, kl

    def sample(self, x, count, generator):
        repeated = x[:, None, :].expand(-1, count, -1)
        z = torch.randn((len(x), count, self.latent), generator=generator, device=x.device)
        # The fixed reconstruction distribution matches the training objective.
        means = self.decoder(torch.cat([repeated, z], dim=-1))
        eps = torch.randn(means.shape, generator=generator, device=x.device)
        return (means + .2 * eps).clamp(-1, 1)


def train_cvae(x, y, vx, vy, args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = CVAE(x.shape[1], hidden=args.hidden, latent=args.latent)
    train_generator = torch.Generator().manual_seed(args.seed + 500)
    val_generator = torch.Generator().manual_seed(args.seed + 600)
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
                        batch_size=args.batch_size, shuffle=True, generator=train_generator)
    vx, vy = torch.tensor(vx, dtype=torch.float32), torch.tensor(vy, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss, _, _ = model.loss(xb, yb, args.beta, train_generator)
            loss.backward()
            optimizer.step()
        model.eval()
        # Reusing the same random draws makes epoch comparisons less noisy.
        val_generator.manual_seed(args.seed + 600)
        with torch.no_grad():
            val, recon, kl = model.loss(vx, vy, args.beta, val_generator)
        history.append({'epoch': epoch, 'validation_loss': float(val), 'reconstruction': float(recon), 'kl': float(kl)})
        if float(val) < best:
            best, best_epoch = float(val), epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history, best_epoch


def summarize(condition, method, candidates, truth, scores, seconds):
    diff = np.abs(candidates - truth[:, None, :])
    hits = (diff <= TOL).all(axis=-1)
    return {'condition': condition, 'method': method, 'n': len(truth),
            'mae_a': diff[:, 0, 0].mean(), 'mae_c': diff[:, 0, 1].mean(),
            'mae_u': diff[:, 0, 2].mean(), 'top1_hit_acu': hits[:, 0].mean(),
            'top5_hit_acu': hits[:, :5].any(axis=1).mean(),
            'top5_hit_ac': (diff[:, :5, :2] <= TOL[:2]).all(axis=-1).any(axis=1).mean(),
            'top5_hit_u': (diff[:, :5, 2] <= TOL[2]).any(axis=1).mean(),
            'top1_observed_mse': scores, 'online_seconds_per_sample': seconds / len(truth)}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    sim = ACUSimulator()
    streams = np.random.SeedSequence(args.seed).spawn(6)
    trng, vrng, terng, mrng, prng, grng = map(np.random.default_rng, streams)
    t0 = time.perf_counter()
    print('Generating train, validation, test spectra...', flush=True)
    train_y, train = split(sim, trng, args.train, args.noise)
    val_y, val = split(sim, vrng, args.validation, args.noise)
    test_y, test = split(sim, terng, args.test, args.noise)
    data_time = time.perf_counter() - t0
    data = {'angles': ANGLES}
    for label, y, spectra in [('train', train_y, train), ('validation', val_y, val), ('test', test_y, test)]:
        data[f'{label}_parameters'] = y
        data.update({f'{label}_{name}': arr for name, arr in spectra.items()})
    np.savez_compressed(args.output / 'dataset.npz', **data)
    features = SpectrumFeatures(train)
    x, vx = features.transform(train), features.transform(val)
    y = np.tile((train_y-CENTER)/HALF, (len(MASKS), 1))
    vy = np.tile((val_y-CENTER)/HALF, (len(MASKS), 1))
    np.savez_compressed(args.output/'feature_stats.npz', mean=features.mean, scale=features.scale)
    print('Training MDN...', flush=True)
    mdn = ConditionalMixture(x.shape[1], center=CENTER, half_range=HALF)
    t0 = time.perf_counter()
    mdn_info = mdn.fit(x, y, vx, vy, mrng, args.mdn_iterations)
    mdn_info['seconds'] = time.perf_counter()-t0
    np.savez_compressed(args.output/'mdn_weights.npz', weights=mdn.weights)
    print('Training cVAE on CPU...', flush=True)
    t0 = time.perf_counter()
    cvae, history, best_epoch = train_cvae(x, y, vx, vy, args)
    cvae_seconds = time.perf_counter()-t0
    torch.save({'state_dict': cvae.state_dict(), 'inputs': x.shape[1], 'hidden': args.hidden,
                'latent': args.latent, 'center': CENTER, 'half_range': HALF}, args.output/'cvae.pt')
    with (args.output/'cvae_history.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    grids, libraries, grid_seconds = {}, {}, {}
    budget_shape = args.grid_shape if args.grid_shape else [args.grid_side] * 3
    for label, shape in [('grid_budget', budget_shape), ('grid_dense', [args.dense_side] * 3)]:
        axes = [np.linspace(lo, hi, side) for (lo, hi), side in zip(BOUNDS, shape)]
        grids[label] = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        t0 = time.perf_counter(); libraries[label] = sim.profiles(grids[label]); grid_seconds[label] = time.perf_counter()-t0
    print(f'Evaluating {len(test_y)} held-out cells across three windows...', flush=True)
    metrics, predictions = [], {}
    sample_gen = torch.Generator().manual_seed(args.seed + 700)
    for name, obs in test.items():
        mask = MASKS[name]
        xt = features.transform({name: obs})
        for label, grid in grids.items():
            t0 = time.perf_counter()
            grid_scores = score_grid(obs, visible_profiles(libraries[label], mask), mask)
            order = np.argsort(grid_scores, axis=1)[:, :5]
            grid_out = grid[order]
            elapsed = time.perf_counter()-t0
            metrics.append(summarize(name, label, grid_out, test_y, float(np.take_along_axis(grid_scores,order[:,:1],axis=1).mean()), elapsed))
            predictions[f'{name}_{label}'] = grid_out
        for method in ('mdn', 'cvae'):
            t0 = time.perf_counter()
            if method == 'mdn':
                pool = mdn.sample(xt, args.proposals, prng)
            else:
                with torch.no_grad():
                    pool = CENTER + HALF*cvae.sample(torch.tensor(xt, dtype=torch.float32), args.proposals, sample_gen).numpy()
            sampling_seconds = time.perf_counter()-t0
            t0=time.perf_counter()
            profiles = sim.profiles(pool.reshape(-1,3))
            profiles = visible_profiles(profiles, mask).reshape(len(test_y), args.proposals, -1)
            errors = np.mean((profiles-obs[:,None,mask])**2, axis=2)
            ranked_idx = np.argsort(errors, axis=1)[:, :5]
            ranked = np.take_along_axis(pool, ranked_idx[:,:,None], axis=1)
            physics_seconds = time.perf_counter()-t0
            raw=pool[:,:5]
            metrics.append(summarize(name, method, raw, test_y, float(errors[:,0].mean()), sampling_seconds))
            metrics.append(summarize(name, method+'_physics', ranked, test_y,
                                     float(np.take_along_axis(errors, ranked_idx[:,:1], axis=1).mean()),
                                     sampling_seconds+physics_seconds))
            predictions[f'{name}_{method}_pool'] = pool
            predictions[f'{name}_{method}_pool_mse'] = errors
            predictions[f'{name}_{method}_physics'] = ranked
    with (args.output/'metrics.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    lookup = {(m['condition'], m['method']): m for m in metrics}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    names = list(MASKS)
    methods = ['grid_budget', 'grid_dense', 'mdn', 'mdn_physics', 'cvae', 'cvae_physics']
    loc = np.arange(len(names)); width = .13
    for i, method in enumerate(methods):
        pos = loc + (i - (len(methods)-1)/2)*width
        axes[0].bar(pos, [lookup[(n, method)]['mae_u'] for n in names], width, label=method)
        axes[1].bar(pos, [100*lookup[(n, method)]['top5_hit_acu'] for n in names], width, label=method)
    axes[0].set(ylabel='Top-1 MAE for u (dimensionless)', title='Oxygen position error')
    axes[1].set(ylabel='Top-5 joint hit rate (%)', title='Joint a, c, u coverage', ylim=(0, 105))
    for ax in axes: ax.set_xticks(loc, names, rotation=10); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=3)
    fig.savefig(args.output/'comparison.png', dpi=160); plt.close(fig)
    np.savez_compressed(args.output/'predictions.npz', **predictions)
    config = vars(args).copy(); config['output'] = str(config['output'])
    config.update({'bounds': BOUNDS.tolist(), 'tolerances': TOL.tolist(), 'windows_deg': WINDOWS,
                   'grid_candidates': {k: len(v) for k, v in grids.items()}, 'data_seconds': data_time, 'grid_precompute_seconds': grid_seconds,
                   'mdn_training': mdn_info, 'cvae_training_seconds': cvae_seconds, 'cvae_best_epoch': best_epoch,
                   'cvae_sampling': 'decoder mean plus fixed Gaussian noise, clipped to bounds',
                   'notes': 'Simulated rutile, independently uniform parameters, idealized simulator. CPU timing is hardware dependent.'})
    (args.output/'run.json').write_text(json.dumps(config, indent=2))
    print('Saved results to', args.output.resolve(), flush=True)
    print('Times: data %.2fs, MDN %.2fs, cVAE %.2fs, grid banks %.2fs' % (data_time, mdn_info['seconds'], cvae_seconds, sum(grid_seconds.values())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs'/'acu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train',type=int,default=600); p.add_argument('--validation',type=int,default=80)
    p.add_argument('--test',type=int,default=40); p.add_argument('--noise',type=float,default=.01)
    p.add_argument('--epochs',type=int,default=80); p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--hidden',type=int,default=64); p.add_argument('--latent',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001); p.add_argument('--beta',type=float,default=1.)
    p.add_argument('--mdn-iterations',type=int,default=120); p.add_argument('--proposals',type=int,default=27)
    p.add_argument('--grid-side',type=int,default=3)
    p.add_argument('--grid-shape',type=int,nargs=3,metavar=('NA','NC','NU'),help='Budget grid points per parameter; product must equal proposals')
    p.add_argument('--dense-side',type=int,default=9)
    args=p.parse_args()
    if min(args.train,args.validation,args.test,args.epochs,args.batch_size,args.hidden,args.latent,args.mdn_iterations) < 1 or args.seed < 0:
        p.error('Counts must be positive and seed must be nonnegative')
    shape = args.grid_shape if args.grid_shape else [args.grid_side] * 3
    if args.proposals < 5 or min(shape) < 2 or args.dense_side < max(shape) or not np.isfinite(args.noise) or args.noise < 0 or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.beta) or args.beta < 0:
        p.error('Invalid proposals, grid, noise, learning rate, or beta')
    if np.prod(shape) != args.proposals:
        p.error('For matched candidate budgets, proposals must equal the grid shape product')
    with threadpool_limits(limits=1): run(args)


if __name__ == '__main__': main()
