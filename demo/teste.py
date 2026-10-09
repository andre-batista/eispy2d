import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eispy2d.api import api

from eispy2d.api import testset_api as ts
from eispy2d.api import benchmark_api as bmk
from eispy2d.core import configuration as cfg
from eispy2d.core import inputdata as ipt
from eispy2d.discretization import richmond as ric
from eispy2d.solvers.inverse import bornapprox as ba
from eispy2d.solvers.inverse import bim
from eispy2d.solvers.inverse import csi
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.utils import stopcriteria as stp

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

def born_iterative_method(scattered_field, incident_field, GS, GD, recover_resolution):
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
    solver = bim.BornIterativeMethod(
        mom.MoM_CG_FFT(tolerance=0.01, maximum_iterations=2500),
        reg.Tikhonov(reg.TIK_FIXED, parameter=0.1),
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
    solver = csi.ContrastSourceInversion(
        stp.StopCriteria(max_iterations=100)
    )
    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi


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

def best_error_method(scattered, incident, GS, GD, resolucao, cand_n=80, quant_rep=40):
    N = GD.shape[0]
    NM, NS = scattered.shape
    theta = cfg.get_angles(NM)
    phi   = cfg.get_angles(NS)

    QUANT_REP = quant_rep

    # es, chi_init, A = manual_born_approimation(scattered, incident, GS, GD, resolucao)
    # chi = chi_init.reshape(-1, 1).copy()

    chi = np.zeros((N, 1), dtype=complex)
    A = np.zeros(
        (NM * NS, N),
        dtype=complex
    )

    for pos in range(N):

        contribution = (
            GS[:, pos, None]
            * incident[pos, :][None, :]
        )

        A[:, pos] = contribution.reshape(
            -1,
            order='F'
        )

    b_full = (A @ chi[:, 0]) 

    def erro_linear(chi_vec):
        b_hat = A @ chi_vec[:, 0]
        pred_mat = b_hat.reshape(NM, NS, order='F')
        diff_mat = scattered - pred_mat
        y_mat = np.real(diff_mat * np.conj(diff_mat))
        ip = np.trapezoid(y_mat, x=phi, axis=1)
        it = np.trapezoid(ip, x=theta)
        return np.real(np.sqrt(it))

    erro_atual = erro_linear(chi)



    for rep in range(QUANT_REP):


        print(f"\n--- PASSADA {rep+1}/{QUANT_REP} --- "
              f"(Erro: {erro_atual:.6e}")

        re_min, re_max = chi.real.min(), chi.real.max()
        im_min, im_max = chi.imag.min(), chi.imag.max()
        folga_re = max(0.1, (re_max - re_min) * 0.1)
        folga_im = max(0.1, (im_max - im_min) * 0.1)
        if rep == 0:
            folga_re = 1
            folga_im = 1

        re = np.linspace(re_min - folga_re, re_max + folga_re, cand_n)
        im = np.linspace(im_min - folga_im, im_max + folga_im, cand_n)
        cand_flat = (re[:, None] + 1j * im[None, :]).ravel()

        #cand_n += 1

        for pos in range(N):
            # if not mask[pos]:
            #     continue

            val_atual = chi[pos, 0]
            candidates = np.append(cand_flat, val_atual)
            deltas = candidates - val_atual              # (K,)

            A_col = A[:, pos]                            # (NM*NS,)

            B_hat = b_full[None, :] + deltas[:, None] * A_col[None, :]

            pred = B_hat.reshape(-1, NM, NS, order='F')

            diff = scattered[None, :, :] - pred
            y = np.real(diff * np.conj(diff))
            ip = np.trapezoid(y, x=phi, axis=2)
            it = np.trapezoid(ip, x=theta, axis=1)
            erros_rn = np.real(np.sqrt(it))

            best_idx = np.argmin(erros_rn)
            candidate_val = candidates[best_idx]

            if erros_rn[best_idx] < erro_atual:
                chi[pos, 0] = candidate_val
                erro_atual = erros_rn[best_idx]
                b_full = b_full + deltas[best_idx] * A_col   # atualiza base

    residual_error = erro_atual
    print(f"\nResidual norm error Final: {residual_error:.6e}")

    E_sct = (A @ chi[:, 0]).reshape(NM, NS, order='F')



    return E_sct, chi.reshape(resolucao)

params = {"disp":True, "resolution":(60, 60),
          "wavelength":1.0, "number_measurements":25,
          "number_sources":25, "image_size":(4.0, 4.0),
          "observation_radius":6.0, "background_permittivity":1.0,
          "contrast":0.25}

api.evaluate(best_error_method, params)