import itertools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from eispy2d.api import api
from eispy2d.discretization import richmond

from bayesnetwork import BayesNetwork



RESOLUTION = (30, 30)
MODE = "random"
N_SAMPLES = 100000
SEED = 0

N_IF_BINS = 900
N_SF_BINS = 900
N_CHI_BINS = 900

N_CALIB = 1000
CALIB_SEED = 12345

N_WORKERS = min(4, os.cpu_count() or 1)
CHUNKSIZE = 4

FORWARD_KWARGS = {"tolerance": 0.001, "maximum_iterations": 5000}

BASE_PARAMS = {
    "wavelength": 1.0,
    "image_size": (4.0, 4.0),
    "observation_radius": 6.0,
    "background_permittivity": 1.0,
    "resolution": RESOLUTION,
    "noise_level": 0.0,
    "disp": False,
    "contrast": 0.25,
    "object_size": 1.0,
}

SOURCES = [5, 10, 15, 20]
MEASUREMENTS = [5, 10, 15, 20]
SHAPES = ["triangle", "square", "circle", "cross", "ellipse",
          "parallelogram", "polygon", "star4", "star5", "star6",
          "rhombus", "trapezoid", "ring", "random"]

WAVELENGTH_RANGE = (0.5, 1.5)
OBJECT_SIZE_RANGE = (0.4, 1.6)
CONTRAST_RANGE = (0.1, 1.5)
BACKGROUND_PERMITTIVITY_RANGE = (0.7, 3.0)

CENTER_RANGE = 1.0
CENTERS = [[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]
ROTATIONS = [0.0, 30.0, 60.0, 90.0]

WAVELENGTHS = [0.8, 1.0, 1.2]
OBJECT_SIZES = [0.5, 1.0, 1.5]
CONTRASTS = [0.1, 0.3, 0.5]
BACKGROUND_PERMITTIVITIES = [1.0, 1.5, 2.0, 2.5, 3.0]

def grid_params():
    for (wavelength, object_size, contrast, background_permittivity,
         ns, nm, shape, center, rot) in itertools.product(
            WAVELENGTHS,
            OBJECT_SIZES,
            CONTRASTS,
            BACKGROUND_PERMITTIVITIES,
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
            background_permittivity=background_permittivity,
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
        p["background_permittivity"] = float(
            rng.uniform(*BACKGROUND_PERMITTIVITY_RANGE))

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


def _format_label(i, p):
    return (
        f"#{i:04d} "
        f"wavelength={p['wavelength']:.3f} "
        f"object_size={p['object_size']:.3f} "
        f"contrast={p['contrast']:.3f} "
        f"background_permittivity={p['background_permittivity']:.3f} "
        f"shape={p['shape']} "
        f"NS={p['number_sources']} "
        f"NM={p['number_measurements']} "
        f"center={p['center']} "
        f"rot={p['rotate']:.0f}"
    )


def _calib_worker(p):
    try:
        scf, idf, gs, gd, res, chi = make_instance(p)
        return scf, idf, gs, chi
    except Exception:
        return None


def _train_worker(p):
    try:
        scf, idf, gs, gd, res, chi = make_instance(p)
        return scf, idf, gs, chi
    except Exception as e:
        return ("__error__", repr(e))


def calibrate(net, n=N_CALIB, seed=CALIB_SEED, workers=N_WORKERS,
              chunksize=CHUNKSIZE):
    if workers and workers > 1:
        with Pool(workers) as pool:
            results = pool.map(_calib_worker,
                               list(random_params(n, seed=seed)),
                               chunksize=chunksize)
    else:
        results = [_calib_worker(p) for p in random_params(n, seed=seed)]

    ifs, sfs, chis, used = [], [], [], 0
    for r in results:
        if r is None:
            continue
        scf, idf, gs, chi = r
        try:
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


def _indexed_params(params_iter):
    for i, p in enumerate(params_iter):
        yield i, p


def run_training(trainer, params_iter, workers=N_WORKERS,
                 chunksize=CHUNKSIZE):
    trained, failed = 0, []
    indexed = list(_indexed_params(params_iter))

    def handle(i, p, result):
        nonlocal trained
        label = _format_label(i, p)
        if isinstance(result, tuple) and len(result) == 2 \
                and result[0] == "__error__":
            failed.append((label, result[1]))
            print(f"[FALHOU] {label}: {result[1]}")
            return
        scf, idf, gs, chi = result
        trainer.train(scf, idf, gs, None, chi)
        trained += 1
        print(f"[ok] {label} | instâncias: {trainer.n_instances} "
              f"| células (IF,SF) ocupadas: {trainer.occupied_cells()}"
              f"/{trainer.N_IF_BINS * trainer.N_SF_BINS}")

    if workers and workers > 1:
        with Pool(workers) as pool:
            for (i, p), result in zip(
                    indexed,
                    pool.imap(_train_worker,
                              (p for _, p in indexed),
                              chunksize=chunksize)):
                handle(i, p, result)
    else:
        for i, p in indexed:
            handle(i, p, _train_worker(p))

    return trained, failed


if __name__ == "__main__":
    trainer = BayesNetwork(n_if_bins=N_IF_BINS, n_sf_bins=N_SF_BINS,
                       n_chi_bins=N_CHI_BINS)

    calibrate(trainer)

    params_iter = grid_params() if MODE == "grid" else random_params(N_SAMPLES)
    if MODE == "grid":
        total = (len(SOURCES) * len(MEASUREMENTS) * len(SHAPES)
                 * len(CENTERS) * len(ROTATIONS)
                 * len(BACKGROUND_PERMITTIVITIES))
        print(f"Modo grid: {total} instâncias")

    trained, failed = run_training(trainer, params_iter)

    print(f"\nTreinadas: {trained} | Falhas: {len(failed)}")
    print(f"Células (IF,SF) ocupadas: {trainer.occupied_cells()}"
          f"/{trainer.N_IF_BINS * trainer.N_SF_BINS}")

    trainer.save("bayes_data.pkl")

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
