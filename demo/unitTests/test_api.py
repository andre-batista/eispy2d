import sys
import os
import numpy as np
from numpy.linalg import inv
from scipy import sparse as sps
from scipy.linalg import solve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eispy2d.api import api
from eispy2d.discretization import richmond as ric
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.core import configuration as cfg
from eispy2d.core import inputdata as ipt
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.utils import stopcriteria as stp
from eispy2d.solvers.inverse import bim
from eispy2d.solvers.inverse import bornapprox as ba
from eispy2d.solvers.inverse import csi


BACKGROUND_PERMITTIVITY = 4.0

def born_approximation(scattered_field, incident_field, GS, GD, recover_resolution):
    NM, NS = scattered_field.shape
    config = cfg.Configuration(
        name='temp',
        wavelength=1.0,
        number_measurements=NM,
        number_sources=NS,
        image_size=[0.8, 0.8],
        observation_radius=1.0,
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
    solver = ba.FirstOrderBornApproximation(reg.Tikhonov(1e-3))
    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi


def born_iterative_method(scattered_field, incident_field, GS, GD, recover_resolution):
    NM, NS = scattered_field.shape
    config = cfg.Configuration(
        name='temp',
        wavelength=1.0,
        number_measurements=NM,
        number_sources=NS,
        image_size=[0.8, 0.8],
        observation_radius=1.0,
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
    solver = bim.BornIterativeMethod(
        mom.MoM_CG_FFT(),
        reg.Tikhonov(1e-3),
        stp.StopCriteria(max_iterations=5)
    )
    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi


def contrast_source_inversion(scattered_field, incident_field, GS, GD, recover_resolution):
    NM, NS = scattered_field.shape
    config = cfg.Configuration(
        name='temp',
        wavelength=1.0,
        number_measurements=NM,
        number_sources=NS,
        image_size=[0.8, 0.8],
        observation_radius=1.0,
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


def alg(scattered_field, incident_field, GS, GD, resolution):

    chi = np.zeros(resolution, dtype=complex)
    N = resolution[0] * resolution[1]
    C = sps.spdiags(chi.reshape(-1), 0, N, N)
    I = np.eye(N, dtype=complex)
    L = inv(I - GD@C)

    recon_scattered_field = GS @ C @ L @ incident_field

    return recon_scattered_field, chi

def solver(scattered_field, incident_field, GS, GD, recover_resolution):
  NM, NS = scattered_field.shape
  N_pixels = incident_field.shape[0]


  A = np.zeros((NM * NS, N_pixels), dtype=complex)

  b = scattered_field.reshape(-1, 1, order='F')

  for s in range(NS):
      E_inc_s = incident_field[:, s:s+1]  


      A_s = GS * E_inc_s.T

      A[s * NM : (s + 1) * NM, :] = A_s

  gamma = 1e-3  
  A_reg = A.conj().T @ A + (gamma ** 2) * np.eye(N_pixels)
  b_reg = A.conj().T @ b

  chi_flat = np.linalg.solve(A_reg, b_reg)

  E_recover = (A @ chi_flat).reshape(NM, NS, order='F')

  return E_recover, chi_flat.reshape(recover_resolution)

def solver2(scattered_field, incident_field, GS, GD, recover_resolution, max_iter=20, gamma=5e-2):
    NM, NS = scattered_field.shape
    N_pixels = incident_field.shape[0]
    b = scattered_field.reshape(-1, 1, order='F')

    E_tot = incident_field.copy()
    chi_flat = np.zeros((N_pixels, 1), dtype=complex)

    for it in range(max_iter):
        A = np.zeros((NM * NS, N_pixels), dtype=complex)
        for s in range(NS):
            E_tot_s = E_tot[:, s:s+1]
            A[s * NM : (s + 1) * NM, :] = GS * E_tot_s.T

        A_reg = A.conj().T @ A + (gamma ** 2) * np.eye(N_pixels)
        b_reg = A.conj().T @ b
        chi_flat = np.linalg.solve(A_reg, b_reg)

        

        C = np.diag(chi_flat.reshape(-1))
        I = np.eye(N_pixels, dtype=complex)

        A_int = I - GD @ C
        for s in range(NS):
            E_tot[:, s:s+1] = solve(A_int, incident_field[:, s:s+1])

    E_recover = (A @ chi_flat).reshape(NM, NS, order='F')

    return E_recover, chi_flat.reshape(recover_resolution)

def media(scattered_field, incident_field, GS, GD, recover_resolution):

    e1 = born_approximation(scattered_field, incident_field, GS, GD, recover_resolution)[0]
    e2 = born_iterative_method(scattered_field, incident_field, GS, GD, recover_resolution)[0]
    e3 = contrast_source_inversion(scattered_field, incident_field, GS, GD, recover_resolution)[0]
    e4 = solver(scattered_field, incident_field, GS, GD, recover_resolution)[0]
    e5 = solver2(scattered_field, incident_field, GS, GD, recover_resolution)[0]

    x1 = born_approximation(scattered_field, incident_field, GS, GD, recover_resolution)[1]
    x2 = born_iterative_method(scattered_field, incident_field, GS, GD, recover_resolution)[1]
    x3 = contrast_source_inversion(scattered_field, incident_field, GS, GD, recover_resolution)[1]
    x4 = solver(scattered_field, incident_field, GS, GD, recover_resolution)[1]
    x5 = solver2(scattered_field, incident_field, GS, GD, recover_resolution)[1]

    media_x = np.stack([x1, x2, x3, x4, x5], axis=0)

    media_x = media_x.mean(axis=0)

    media_e = np.stack([e1, e2, e3, e4, e5], axis=0)
    media_e = media_e.mean(axis=0)

    return media_e, media_x


params = {"shape":"star4", "disp":True, 'BACKGROUND_PERMITTIVITY':BACKGROUND_PERMITTIVITY}
api.evaluate(media, params)