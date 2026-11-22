from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure
from scipy.spatial.distance import cdist


ANGLES = np.linspace(20.0, 70.0, 501)
WINDOWS = {"full": (20, 70), "low_information": (24, 33), "contrast": (33, 42)}
MASKS = {name: (ANGLES >= start) & (ANGLES <= end) for name, (start, end) in WINDOWS.items()}
BOUNDS = np.array([[4.50, 4.69], [2.90, 3.02]])
CENTER = BOUNDS.mean(axis=1)
HALF_RANGE = np.diff(BOUNDS, axis=1).ravel() / 2
WAVELENGTH = 1.5406
PEAK_SIGMA = 0.12  # Gaussian standard deviation in degrees of 2 theta


class RutileDiffraction:
    def __init__(self):
        cif = Path(__file__).parent / "data" / "rutile.cif"
        self.reference = Structure.from_file(cif)
        self.calculator = XRDCalculator(wavelength=WAVELENGTH)

    def pattern(self, a, c):
        structure = Structure(
            Lattice.tetragonal(a, c), self.reference.species, self.reference.frac_coords
        )
        return self.calculator.get_pattern(
            structure, scaled=False, two_theta_range=(ANGLES[0] - 1, ANGLES[-1] + 1)
        )

    def profiles(self, parameters, peak_sigma=PEAK_SIGMA):
        profiles = np.empty((len(parameters), ANGLES.size))
        for row, (a, c) in enumerate(parameters):
            pattern = self.pattern(a, c)
            distance = (ANGLES[None, :] - pattern.x[:, None]) / peak_sigma
            profiles[row] = (pattern.y[:, None] * np.exp(-0.5 * distance**2)).sum(axis=0)
        return profiles


def visible_profiles(profiles, mask):
    visible = profiles[..., mask]
    return visible / visible.max(axis=-1, keepdims=True)


def make_split(simulator, rng, size, noise, peak_sigma=PEAK_SIGMA):
    parameters = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(size, 2))
    clean = simulator.profiles(parameters, peak_sigma)
    paired_noise = rng.normal(0, noise, size=clean.shape)
    spectra = {}
    for name, mask in MASKS.items():
        observation = np.zeros_like(clean)
        observation[:, mask] = visible_profiles(clean, mask) + paired_noise[:, mask]
        spectra[name] = observation
    return parameters, spectra


def parameter_grid(side):
    axes = [np.linspace(low, high, side) for low, high in BOUNDS]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)


def score_grid(observations, templates, mask):
    return cdist(observations[:, mask], templates, metric="sqeuclidean") / mask.sum()


def order_by_fit(parameters, errors):
    # For a score tie at numerical precision, rank closest to the prior center first.
    prior_distance = np.sum(((parameters - CENTER) / HALF_RANGE) ** 2, axis=1)
    return np.lexsort((prior_distance, np.round(errors, 12)))


def rank_candidates(parameters, errors, k, separation):
    return diverse_indices(parameters, order_by_fit(parameters, errors), k, separation)


def diverse_indices(parameters, order, k, separation):
    selected = []
    for index in order:
        if all(np.max(np.abs(parameters[index] - parameters[j])) >= separation for j in selected):
            selected.append(index)
            if len(selected) == k:
                break
    return np.array(selected, dtype=int)
