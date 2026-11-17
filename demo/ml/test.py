import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eispy2d.api import api
from eispy2d.experiments import benchmark as bmk
from eispy2d.experiments import experiment as exp
from eispy2d.experiments import testset as tst
from eispy2d.core import result as rst

from eispy2d.core import inputdata as ipt
from eispy2d.core import configuration as cfg
from eispy2d.solvers.inverse import bornapprox as ba
from eispy2d.solvers.inverse import bim
from eispy2d.solvers.inverse import csi
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.utils import stopcriteria as stp
from eispy2d.discretization import richmond as ric

import numpy as np

from bayesnetwork import BayesNetwork

params = {
    "wavelength": 1.0,
    "image_size": (2.0, 2.0),
    "observation_radius": 6.0,
    "background_permittivity": 3.0,
    "resolution": (30, 30),
    "noise_level": 1.0,    
    "disp": True,
    "contrast": 0.25,
    "object_size": 1.0,
    "number_sources": 10,
    "number_measurements": 10,
    "shape": "star5",
    "center": [0.0, 0.0],
}

WAVELENGTH = params["wavelength"]
Lx, Ly = params["image_size"]
OBSERVATION_RADIUS = params["observation_radius"]
RESOLUTION = params["resolution"]
NOISE_LEVEL = params["noise_level"]
BACKGROUND_PERMITTIVITY = params["background_permittivity"]

def manual_born_approximation(scattered_field, incident_field, GS, GD, recover_resolution):
    NM, NS = scattered_field.shape
    N_pixels = incident_field.shape[0]
    A = np.zeros((NM * NS, N_pixels), dtype=complex)
    b = scattered_field.reshape(-1, 1, order='F')
    for s in range(NS):
        E_inc_s = incident_field[:, s:s+1]
        A_s = GS * E_inc_s.T
        A[s * NM:(s + 1) * NM, :] = A_s
    gamma = 1e-1
    A_reg = A.conj().T @ A + (gamma ** 2) * np.eye(N_pixels)
    b_reg = A.conj().T @ b
    chi_flat = np.linalg.solve(A_reg, b_reg)
    E_recover = (A @ chi_flat).reshape(NM, NS, order='F')

    return E_recover, chi_flat.reshape(recover_resolution)

def born_approximation(scattered_field, incident_field, GS, GD, recover_resolution):
    NM, NS = scattered_field.shape
    config = cfg.Configuration(
        name='temp',
        wavelength=1.0,
        number_measurements=NM,
        number_sources=NS,
        image_size=[4.0, 4.0],
        observation_radius=6.0,
        background_permittivity=1.0,
        perfect_dielectric=True
    )
    discretization = ric.Richmond(config, recover_resolution, state=False)
    inputdata = ipt.InputData(
        name='temp',
        configuration=config,
        resolution=recover_resolution,
        scattered_field=scattered_field,
        incident_field=incident_field,
        indicators=[]
    )
    solver = ba.FirstOrderBornApproximation(reg.Tikhonov(1e-1))
    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi

def contrast_source_inversion(scattered_field, incident_field, GS, GD, recover_resolution):
    NM, NS = scattered_field.shape
    config = cfg.Configuration(
        name='temp',
        wavelength=WAVELENGTH,
        number_measurements=NM,
        number_sources=NS,
        image_size=[Lx, Ly],
        observation_radius=OBSERVATION_RADIUS,
        background_permittivity=BACKGROUND_PERMITTIVITY,
        perfect_dielectric=True
    )
    discretization = ric.Richmond(config, recover_resolution, state=False)
    inputdata = ipt.InputData(
        name='temp',
        configuration=config,
        resolution=recover_resolution,
        scattered_field=scattered_field,
        incident_field=incident_field,
        indicators=[]
    )
    solver = csi.ContrastSourceInversion(
        stp.StopCriteria(max_iterations=100)
    )
    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi


#net = BayesNetwork.load("data/bayes_data_6.pkl")
net = BayesNetwork.load("bayes_data.pkl")


import numpy as np
import matplotlib.pyplot as plt

BASE = {
    "wavelength": 1.0,
    "image_size": (2.0, 2.0),
    "observation_radius": 6.0,
    "background_permittivity": 3.0,
    "resolution": (30, 30),
    "noise_level": 1.0,
    "disp": False,
    "contrast": 0.25,
    "object_size": 1.0,
    "number_sources": 10,
    "number_measurements": 10,
    "shape": "star5",
    "center": [0.0, 0.0],
}

METHODS = {
    "bayes": net.solve,
    "born": born_approximation,
    "csi": contrast_source_inversion,
}

resolutions = [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30),
               (35, 35), (40, 40), (45, 45), (50, 50)]

sources_measurements = [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25),
                        (30, 30), (35, 35), (40, 40), (45, 45), (50, 50)]

N_REPEATS = 5


def run_mean(method, params, n=N_REPEATS):
    ts = []
    for _ in range(n):
        r = api.evaluate(method, params)
        ts.append(r.execution_time)
    return float(np.mean(ts))


times_res = {name: [] for name in METHODS}
for res in resolutions:
    p = dict(BASE)
    p["resolution"] = res
    for name, method in METHODS.items():
        try:
            t = run_mean(method, p)
        except Exception as e:
            print(f"[FALHOU] {name} res={res}: {e}")
            t = float("nan")
        times_res[name].append(t)
        print(f"{name:6s} res={res} -> {t:.4f} s (média de {N_REPEATS})")

times_sm = {name: [] for name in METHODS}
for ns, nm in sources_measurements:
    p = dict(BASE)
    p["resolution"] = (30, 30)
    p["number_sources"] = ns
    p["number_measurements"] = nm
    for name, method in METHODS.items():
        try:
            t = run_mean(method, p)
        except Exception as e:
            print(f"[FALHOU] {name} ns={ns} nm={nm}: {e}")
            t = float("nan")
        times_sm[name].append(t)
        print(f"{name:6s} ns={ns} nm={nm} -> {t:.4f} s (média de {N_REPEATS})")

x_res = [r[0] for r in resolutions]
x_sm = [f"{ns}x{nm}" for ns, nm in sources_measurements]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for name, ts in times_res.items():
    axes[0].plot(x_res, ts, marker="o", label=name)
axes[0].set_xlabel("resolução (lado)")
axes[0].set_ylabel("tempo médio de execução (s)")
axes[0].set_title(f"Tempo vs resolução (média de {N_REPEATS})")
axes[0].legend()
axes[0].grid(True)

for name, ts in times_sm.items():
    axes[1].plot(x_sm, ts, marker="o", label=name)
axes[1].set_xlabel("(sources, measurements)")
axes[1].set_ylabel("tempo médio de execução (s)")
axes[1].set_title(f"Tempo vs (sources, measurements) (média de {N_REPEATS})")
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("timing_comparison_mean.png", dpi=150)
plt.show()

