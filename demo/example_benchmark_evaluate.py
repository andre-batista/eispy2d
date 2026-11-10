import sys

sys.path.insert(1, '../eispy2d')

import os
from pathlib import Path

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

WAVELENGTH = 1.0
Lx, Ly = 0.8, 0.8
OBSERVATION_RADIUS = 1.0
RESOLUTION = (60, 60)
NOISE_LEVEL = 1.0
BACKGROUND_PERMITTIVITY = 4.0

N_REPETITIONS = 30

BASE_PARAMS = {
    "wavelength": 1.0,
    "image_size": (4.0, 4.0),
    "number_measurements": 10,
    "number_sources": 10,
    "observation_radius": 6.0,
    "background_permittivity": 1.0,
    "resolution": (60, 60),
    "noise_level": 1.0,
    "disp": False,
    "shape": "triangle",
    "center": [0.0, 0.0],
    "rotate": 0.0,
    "contrast": 0.5,
    "object_size": 1.0,
}

SHAPES = [
    "triangle",
    "square",
    "circle",
    "cross",
    "ellipse",
    "parallelogram",
    "polygon",
    "random",
    "random_gaussians",
    "star4",
    "star5",
    "star6",
    "rhombus",
    "trapezoid",
    "ring",
]

SOURCE_VALUES = [5, 10, 15, 20, 30 ]
MEASUREMENT_VALUES = [5, 10, 15, 20, 30 ]
SOURCE_MEASUREMENT_PAIRS = [
    (5, 5),
    (5, 10),
    (5, 15),
    (10, 5),
    (10, 10),
    (10, 15),
    (15, 5),
    (15, 10),
    (15, 15),
]
FORWARD_KWARGS = {
    "tolerance": 0.001,
    "maximum_iterations": 1000,
}

OUTPUT_DIRECTORY = Path("api_benchmark_results")
BENCHMARK_NAME = "api_evaluate_benchmark"


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

ALGORITHMS = [
    born_iterative_method,
    born_approximation,
    contrast_source_inversion
]

def make_testset(name, params, repetitions=N_REPETITIONS):
    samples = []
    configuration = None

    for repetition in range(repetitions):
        config, inputdata, _, _ = api.prepare_problem(
            params=params,
            forward_kwargs=FORWARD_KWARGS,
        )

        if configuration is None:
            configuration = config

        inputdata.name = f"{name}_{repetition:03d}"
        samples.append(inputdata)

    testset = tst.TestSet(
        name=name,
        configuration=configuration,
        contrast=params["contrast"],
        object_size=params["object_size"],
        resolution=params["resolution"],
        density=None,
        map_pattern=exp.REGULAR_POLYGONS_PATTERN,
        sample_size=repetitions,
        noise=params["noise_level"],
        indicators=[
            rst.REL_PERMITTIVITY_PAD_ERROR,
            rst.RESIDUAL_NORM_ERROR,
            rst.EXECUTION_TIME,
        ],
        contrast_mode=exp.FIXED_CONTRAST,
        object_size_mode=exp.FIXED_SIZE,
        density_mode=exp.SINGLE_OBJECT,
        allow_rotation=False,
        random_position=False,
    )

    testset.test = samples
    return testset


def build_testsets(repetitions=N_REPETITIONS):
    testsets = []

    # Phase 1: vary only the object shape.
    for shape in SHAPES:
        params = BASE_PARAMS.copy()
        params["shape"] = shape
        name = f"shape_{shape}"
        testsets.append(make_testset(name, params, repetitions))

    # Phase 2: vary only the number of sources.
    for number_sources in SOURCE_VALUES:
        params = BASE_PARAMS.copy()
        params["number_sources"] = number_sources
        name = f"sources_{number_sources}"
        testsets.append(make_testset(name, params, repetitions))

    # Phase 3: vary only the number of measurements.
    for number_measurements in MEASUREMENT_VALUES:
        params = BASE_PARAMS.copy()
        params["number_measurements"] = number_measurements
        name = f"measurements_{number_measurements}"
        testsets.append(make_testset(name, params, repetitions))

    # Phase 4: vary source/measurement pairs for every shape.
    for shape in SHAPES:
        for number_sources, number_measurements in SOURCE_MEASUREMENT_PAIRS:
            params = BASE_PARAMS.copy()
            params["shape"] = shape
            params["number_sources"] = number_sources
            params["number_measurements"] = number_measurements

            name = (
                f"pair_{shape}_sources_{number_sources}"
                f"_measurements_{number_measurements}"
            )
            testsets.append(make_testset(name, params, repetitions))

    return testsets


def run_benchmark(repetitions=N_REPETITIONS):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Generating API evaluate test sets...")
    testsets = build_testsets(repetitions)

    if not ALGORITHMS:
        raise ValueError("Set ALGORITHM to the inverse algorithm function.")

    evaluate_methods = [
        api.EvaluateS(
            algorithm=algorithm,
            forward_kwargs=FORWARD_KWARGS,
        )
        for algorithm in ALGORITHMS
    ]

    benchmark = bmk.Benchmark(
        name=BENCHMARK_NAME,
        method=evaluate_methods,
        discretization=None,
        testset=testsets,
    )

    print(f"Test sets: {len(testsets)}")
    print(f"Repetitions per test set: {repetitions}")
    print(f"Total evaluations: {len(testsets) * repetitions}")
    print("Running benchmark...")

    benchmark.run(parallelization=None)

    save_path = str(OUTPUT_DIRECTORY) + os.sep
    benchmark.save(file_path=save_path, save_testset=True)

    print(f"Benchmark saved to: {save_path}{BENCHMARK_NAME}")
    return benchmark


if __name__ == "__main__":
    run_benchmark()