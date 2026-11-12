import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from eispy2d.api import api
from eispy2d.discretization import richmond

from bayesnetwork import BayesNetwork



RESOLUTION = (30, 30)
MODE = "random"
N_SAMPLES = 1000
SEED = 0

N_IF_BINS = 100
N_SF_BINS = 100
N_CHI_BINS = 100

N_CALIB = 100
CALIB_SEED = 12345

FORWARD_KWARGS = {"tolerance": 0.001, "maximum_iterations": 5000}

BASE_PARAMS = {
    "wavelength": 1.0,
    "image_size": (2.0, 2.0),
    "observation_radius": 6.0,
    "background_permittivity": 1.0,
    "resolution": RESOLUTION,
    "noise_level": 1.0,
    "disp": False,
    "contrast": 0.25,
    "object_size": 1.0,
}

SOURCES = [5, 10, 15, 20]
MEASUREMENTS = [5, 10, 15, 20]
SHAPES = ["triangle", "square", "circle", "cross", "ellipse",
          "parallelogram", "polygon", "star4", "star5", "star6",
          "rhombus", "trapezoid", "ring", "random"]

WAVELENGTH_RANGE = (0.8, 1.2)
OBJECT_SIZE_RANGE = (0.5, 1.5)
CONTRAST_RANGE = (0.1, 0.5)

# Grid mode parameters
CENTER_RANGE = 1.0
CENTERS = [[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]
ROTATIONS = [0.0, 30.0, 60.0, 90.0] 

WAVELENGTHS = [0.8, 1.0, 1.2]
OBJECT_SIZES = [0.5, 1.0, 1.5]
CONTRASTS = [0.1, 0.3, 0.5]

def grid_params():
    for (wavelength, object_size, contrast,
         ns, nm, shape, center, rot) in itertools.product(
            WAVELENGTHS,
            OBJECT_SIZES,
            CONTRASTS,
            SOURCES,
            MEASUREMENTS,
            SHAPES,
            CENTERS,
            ROTATIONS
    ):
        p = BASE_PARAMS.copy()
        p.update(
            wavelength=wavelength,
            object_size=object_size,
            contrast=contrast,
            number_sources=ns,
            number_measurements=nm,
            shape=shape,
            center=list(center),
            rotate=rot
        )
        yield p


def random_params(n, seed=SEED):
    rng = np.random.default_rng(seed)

    for _ in range(n):
        p = BASE_PARAMS.copy()

        p["wavelength"] = float(rng.uniform(*WAVELENGTH_RANGE))
        p["object_size"] = float(rng.uniform(*OBJECT_SIZE_RANGE))
        p["contrast"] = float(rng.uniform(*CONTRAST_RANGE))

        p["number_sources"] = int(rng.choice(SOURCES))
        p["number_measurements"] = int(rng.choice(MEASUREMENTS))
        p["shape"] = str(rng.choice(SHAPES))
        p["center"] = rng.uniform(
            -CENTER_RANGE, CENTER_RANGE, 2
        ).tolist()
        p["rotate"] = float(rng.uniform(0.0, 180.0))

        yield p


def make_instance(params):
    config, inputdata, solver, _ = api.prepare_problem(params, FORWARD_KWARGS)
    res = tuple(params["resolution"])

    scf = inputdata.scattered_field                      # (nm, ns)
    idf = solver.incident_field(res, config)             # (N, ns)
    gs = richmond.richmond_data(config, res)
    gd = richmond.richmond_state(config, res)

    chi = np.asarray(inputdata.rel_permittivity) / config.epsilon_rb - 1
    assert chi.shape == res, f"chi {chi.shape} != resolution {res}"
    return scf, idf, gs, gd, res, chi


def calibrate(net, n=N_CALIB, seed=CALIB_SEED):
    ifs, sfs, chis, used = [], [], [], 0
    for p in random_params(n, seed=seed):
        try:
            scf, idf, gs, gd, res, chi = make_instance(p)
            f_if, f_sf = net.pixel_features(scf, idf, gs)
        except Exception:
            continue
        ifs.append(f_if)
        sfs.append(f_sf)
        chis.append(np.real(chi).ravel())
        used += 1

    if not used:
        raise RuntimeError("Calibração falhou: nenhuma instância válida.")

    net.calibrate(np.concatenate(ifs), np.concatenate(sfs),
                  np.concatenate(chis))
    print(f"Calibração ({used} instâncias): IF_RANGE={net.IF_RANGE} "
          f"SF_RANGE={net.SF_RANGE} CHI_RANGE={net.CHI_RANGE}")


def run_training(trainer, params_iter):
    trained, failed = 0, []

    for i, p in enumerate(params_iter):
        label = (
            f"#{i:04d} "
            f"wavelength={p['wavelength']:.3f} "
            f"object_size={p['object_size']:.3f} "
            f"contrast={p['contrast']:.3f} "
            f"shape={p['shape']} "
            f"NS={p['number_sources']} "
            f"NM={p['number_measurements']} "
            f"center={p['center']} "
            f"rot={p['rotate']:.0f}"
        )
        try:
            scf, idf, gs, gd, res, chi = make_instance(p)
            trainer.train(scf, idf, gs, gd, chi)
        except Exception as e:  # ex.: forma que não cabe no domínio
            failed.append((label, repr(e)))
            print(f"[FALHOU] {label}: {e}")
            continue

        trained += 1
        print(f"[ok] {label} | instâncias: {trainer.n_instances} "
              f"| células (IF,SF) ocupadas: {trainer.occupied_cells()}"
              f"/{trainer.N_IF_BINS * trainer.N_SF_BINS}")

    return trained, failed


if __name__ == "__main__":
    trainer = BayesNetwork(n_if_bins=N_IF_BINS, n_sf_bins=N_SF_BINS,
                       n_chi_bins=N_CHI_BINS)

    calibrate(trainer)

    params_iter = grid_params() if MODE == "grid" else random_params(N_SAMPLES)
    if MODE == "grid":
        total = (len(SOURCES) * len(MEASUREMENTS) * len(SHAPES)
                 * len(CENTERS) * len(ROTATIONS))
        print(f"Modo grid: {total} instâncias")

    trained, failed = run_training(trainer, params_iter)

    print(f"\nTreinadas: {trained} | Falhas: {len(failed)}")
    print(f"Células (IF,SF) ocupadas: {trainer.occupied_cells()}"
          f"/{trainer.N_IF_BINS * trainer.N_SF_BINS}")

    trainer.save("./data/bayes_data_4.pkl")

    test_params = next(random_params(1, seed=999))

    test_params.update(
        number_sources=10,
        number_measurements=10,
        shape="star4",
        center=[0.2, -0.3],
        rotate=15.0,
        disp=True
    )

    result = api.evaluate(trainer.solve, test_params)