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
    "background_permittivity": 1.0,
    "resolution": (30, 30),
    "noise_level": 1.0,    
    "disp": True,
    "contrast": 0.25,
    "object_size": 1.0,
    "number_sources": 25,
    "number_measurements": 25,
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


net = BayesNetwork.load("data/bayes_data_6.pkl")
#net = BayesNetwork.load("bayes_data.pkl")

api.evaluate(net.solve, params)

#api.evaluate(manual_born_approximation, params)

