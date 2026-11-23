"""Predict candidates for a CSV spectrum using saved simulated-rutile models."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from models import ConditionalMixture, SpectrumFeatures
from physics import ANGLES, MASKS, visible_profiles
from acu_benchmark import ACUSimulator, CANDIDATE_COUNT, CENTER, HALF, CVAE, diverse_u_indices


def main():
    p=argparse.ArgumentParser(description=f'{__doc__} Generates {CANDIDATE_COUNT} candidates.')
    p.add_argument('spectrum_csv', type=Path, help='CSV with angle_deg,intensity headers and the required 0.1 degree grid')
    p.add_argument('--model-dir', type=Path, default=Path(__file__).parent/'outputs'/'acu')
    p.add_argument('--window', choices=list(MASKS), required=True)
    p.add_argument('--method', choices=['mdn', 'cvae'], default='cvae')
    p.add_argument('--ranking', choices=['physics', 'diverse-u'], default='physics',
                   help='Rank by spectral error or diversify u among the first five candidates')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=Path('acu_candidates.csv'))
    args=p.parse_args()
    if args.seed < 0: p.error('seed must be nonnegative')
    with args.spectrum_csv.open(newline='') as f:
        reader=csv.DictReader(f)
        if not {'angle_deg','intensity'}.issubset(reader.fieldnames or []): p.error('CSV needs angle_deg,intensity headers')
        rows=list(reader)
    angles=np.array([float(r['angle_deg']) for r in rows]); intensities=np.array([float(r['intensity']) for r in rows])
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(intensities)): p.error('Input must be finite')
    indices=np.rint((angles-ANGLES[0])/.1).astype(int)
    if np.any(indices<0) or np.any(indices>=len(ANGLES)) or np.any(np.abs(angles-ANGLES[np.clip(indices,0,len(ANGLES)-1)])>1e-5) or len(np.unique(indices)) != len(indices):
        p.error('Angles must be unique and on the 20–70 degree grid at 0.1 degree steps')
    mask=MASKS[args.window]
    if not np.isin(np.flatnonzero(mask),indices).all(): p.error('CSV must contain every angle in the selected window')
    obs=np.zeros((1,len(ANGLES))); obs[0,indices]=intensities
    maximum=obs[0,mask].max()
    if maximum<=0: p.error('Selected window maximum must be positive')
    obs[0,mask] /= maximum
    obs[0,~mask]=0
    stats=np.load(args.model_dir/'feature_stats.npz')
    features=SpectrumFeatures.__new__(SpectrumFeatures); features.mean=stats['mean']; features.scale=stats['scale']
    x=features.transform({args.window:obs})
    if args.method=='mdn':
        model=ConditionalMixture(x.shape[1],center=CENTER,half_range=HALF)
        model.weights=np.load(args.model_dir/'mdn_weights.npz')['weights']
        pool=model.sample(x,CANDIDATE_COUNT,np.random.default_rng(args.seed))[0]
    else:
        # Checkpoints can execute code when loaded: only use a trusted local model directory.
        ckpt = torch.load(args.model_dir / 'cvae.pt', map_location='cpu', weights_only=False)
        model = CVAE(ckpt['inputs'], ckpt['hidden'], ckpt['latent'])
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        with torch.no_grad():
            samples = model.sample(torch.tensor(x, dtype=torch.float32), CANDIDATE_COUNT,
                                   torch.Generator().manual_seed(args.seed))
            pool = (ckpt['center'] + ckpt['half_range'] * samples.numpy())[0]
    sim=ACUSimulator(); profiles=visible_profiles(sim.profiles(pool),mask)
    mse=((profiles-obs[0,mask])**2).mean(axis=1)
    by_fit = np.argsort(mse)
    order = by_fit
    if args.ranking == 'diverse-u':
        top = diverse_u_indices(pool, by_fit)
        order = np.concatenate((top, by_fit[~np.isin(by_fit, top)]))
    physics_rank = np.empty_like(by_fit)
    physics_rank[by_fit] = np.arange(1, len(by_fit) + 1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['physics_rank','a_angstrom','c_angstrom','u_dimensionless','observed_mse'])
        for idx in order: writer.writerow([physics_rank[idx],*pool[idx],mse[idx]])
    print(f'Saved {len(pool)} candidates ({args.ranking}) to {args.output.resolve()}')
    print('Best:', pool[order[0]], 'observed MSE:', mse[order[0]])
    print('Caution: assumes ideal single-phase rutile, a fixed prior range and the simulator in README_ACU.md.')


if __name__=='__main__': main()
