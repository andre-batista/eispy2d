import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eispy2d.api import casestudy_api as cst
from eispy2d.core import configuration as cfg
from eispy2d.core import inputdata as ipt
from eispy2d.discretization import richmond as ric
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.solvers.inverse import bim
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.utils import stopcriteria as stp


WAVELENGTH = 1.0
Lx, Ly = 0.8, 0.8
OBSERVATION_RADIUS = 1.0
RESOLUTION = (30, 30)
NOISE_LEVEL = 1.0
NUMBER_MEASUREMENTS = 10
NUMBER_SOURCES = 10
BACKGROUND_PERMITTIVITY = 4.0
SHAPE = "triangle"
STOCHASTIC_RUNS = 1

WAVELENGTH_VALUES = [i * 0.2 for i in range(1, 16)]
NOISE_VALUES = [i * 0.5 for i in range(1, 21)]
NS_VALUES = [i for i in range(4, 65, 4)]
NM_VALUES = [i for i in range(4, 65, 4)]

N_CANDS = [i for i in range(10, 70, 10)]
REPS = [i for i in range(1, 30, 4)]
REG_TIK_VALUES = [
    1e-4, 5e-4, 1e-3, 5e-3,
    1e-2, 5e-2, 1e-1, 5e-1
]

CASE_STUDY_NAME = "api_casestudy2"

def best_error_method(scattered, incident, GS, GD, resolucao, cand_n=60, quant_rep=30):
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


def fixed_params():
    return {
        "wavelength": WAVELENGTH,
        "image_size": (Lx, Ly),
        "observation_radius": OBSERVATION_RADIUS,
        "resolution": RESOLUTION,
        "noise_level": NOISE_LEVEL,
        "number_measurements": NUMBER_MEASUREMENTS,
        "number_sources": NUMBER_SOURCES,
        "shape": SHAPE,
        "background_permittivity": BACKGROUND_PERMITTIVITY
    }


def build_input_parameter_tests(variable_param, values, study_name):
    tests = []

    for value in values:
        params = fixed_params()
        params[variable_param] = value
        params["_study"] = study_name
        tests.append(params)

    return tests


def build_algorithm_parameter_tests(algorithm_param, values, study_name):
    tests = []
    algorithm_params = []

    for value in values:
        tests.append({
            **fixed_params(),
            "_study": study_name
        })
        algorithm_params.append({
            algorithm_param: value
        })

    return tests, algorithm_params


def build_case_study():
    print('\n[START] Building case study...')

    tests = []
    algorithm_params = []

    input_studies = [
        ("wavelength", WAVELENGTH_VALUES, "wavelength"),
        ("noise_level", NOISE_VALUES, "noise"),
        ("number_sources", NS_VALUES, "sources"),
        ("number_measurements", NM_VALUES, "measurements")
    ]

    print('[INFO] Adding input parameter studies...')
    for variable_param, values, study_name in input_studies:
        tests.extend(
            build_input_parameter_tests(
                variable_param,
                values,
                study_name
            )
        )
        algorithm_params.extend(
            [None] * len(values)
        )
        print(f'[INFO]   - Study "{study_name}": {len(values)} test(s) added.')

    algorithm_studies = [
        ("cand_n", N_CANDS, "cand_n"),
        ("quant_rep", REPS, "quant_rep")
    ]

    print('[INFO] Adding algorithm parameter studies...')
    for algorithm_param, values, study_name in algorithm_studies:
        current_tests, current_algorithm_params = (
            build_algorithm_parameter_tests(
                algorithm_param,
                values,
                study_name
            )
        )
        tests.extend(current_tests)
        algorithm_params.extend(current_algorithm_params)
        print(f'[INFO]   - Study "{study_name}": {len(values)} test(s) added.')

    case_study = cst.CaseStudy(
        name=CASE_STUDY_NAME,
        algorithm=best_error_method,
        test=tests,
        stochastic_runs=STOCHASTIC_RUNS,
        save_stochastic_runs=True,
        algorithm_params=algorithm_params
    )

    print(f'[OK] Case study built: {case_study.name}')
    print(f'[INFO] Total tests: {len(tests)}')
    print(f'[INFO] Stochastic runs per test: {STOCHASTIC_RUNS}')

    return case_study


def run_case_study():
    print('=' * 70)
    print('CASE STUDY GENERATOR - API EVALUATE')
    print('=' * 70)

    study = build_case_study()

    print('\n[START] Executing case study...')
    print('[INFO] This may take a while. Please wait...')

    try:
        study.run(
            parallelization=cst.PARALLELIZE_EXECUTIONS
        )
        print('[OK] Case study completed successfully!')
    except Exception as e:
        print(f'[ERROR] Error during case study execution: {e}')
        print('[WARN] Saving partial results...')

    print('\n[START] Saving results...')
    study.save(save_test=True)
    print(f'[OK] Results saved to: {study.name}')

    print('\n' + '=' * 70)
    print('[DONE] Case study execution finished!')
    print('=' * 70)

    return study


case_study = run_case_study()