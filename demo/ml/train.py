
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from eispy2d.api import api
from eispy2d.discretization import richmond

from chain import Train  


# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------
RESOLUTION = (30, 30)
MODE = "random"          # "random" (amostragem) ou "grid" (produto cartesiano)
N_SAMPLES = 1000          # usado só no modo "random"
SEED = 0

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

# Objeto de lado ~1 em domínio 4x4: mantenha o centro em [-1, 1]
CENTER_RANGE = 1.0
CENTERS = [[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]
ROTATIONS = [0.0, 30.0, 60.0, 90.0]   # assumindo graus


# ----------------------------------------------------------------------
# Geração dos parâmetros
# ----------------------------------------------------------------------
def grid_params():
    for ns, nm, shape, center, rot in itertools.product(
            SOURCES, MEASUREMENTS, SHAPES, CENTERS, ROTATIONS):
        p = BASE_PARAMS.copy()
        p.update(number_sources=ns, number_measurements=nm, shape=shape,
                 center=list(center), rotate=rot)
        yield p


def random_params(n, seed=SEED):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        p = BASE_PARAMS.copy()
        p["number_sources"] = int(rng.choice(SOURCES))
        p["number_measurements"] = int(rng.choice(MEASUREMENTS))
        p["shape"] = str(rng.choice(SHAPES))
        p["center"] = rng.uniform(-CENTER_RANGE, CENTER_RANGE, 2).tolist()
        p["rotate"] = float(rng.uniform(0.0, 180.0))
        yield p


# ----------------------------------------------------------------------
# Instância no formato esperado pelo Train.train
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------
def run_training(trainer, params_iter):
    trained, failed = 0, []

    for i, p in enumerate(params_iter):
        label = (f"#{i:04d} shape={p['shape']} NS={p['number_sources']} "
                 f"NM={p['number_measurements']} center={p['center']} "
                 f"rot={p['rotate']:.0f}")
        try:
            scf, idf, gs, gd, res, chi = make_instance(p)
            trainer.train(scf, idf, gs, gd, chi)
        except Exception as e:  # ex.: forma que não cabe no domínio
            failed.append((label, repr(e)))
            print(f"[FALHOU] {label}: {e}")
            continue

        trained += 1
        print(f"[ok] {label} | chaves no chain: {len(trainer.chain)}")

    return trained, failed


if __name__ == "__main__":
    trainer = Train()

    params_iter = grid_params() if MODE == "grid" else random_params(N_SAMPLES)
    if MODE == "grid":
        total = (len(SOURCES) * len(MEASUREMENTS) * len(SHAPES)
                 * len(CENTERS) * len(ROTATIONS))
        print(f"Modo grid: {total} instâncias")

    trained, failed = run_training(trainer, params_iter)

    print(f"\nTreinadas: {trained} | Falhas: {len(failed)}")
    print(f"Total de chaves no chain: {len(trainer.chain)}")

    trainer.save("chain_data.pkl")

    # Teste em uma instância nova (fora do treino)
    test_params = BASE_PARAMS.copy()
    test_params.update(number_sources=10, number_measurements=10,
                       shape="star4", center=[0.2, -0.3], rotate=15.0, disp=True)
    result = api.evaluate(trainer.solve, test_params)
