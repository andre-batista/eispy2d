import sys
import os
import numpy as np
import scipy.sparse as sps
from numpy.linalg import inv

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


WAVELENGTH = 1.0
Lx, Ly = 0.8, 0.8
OBSERVATION_RADIUS = 1.0
RESOLUTION = (30, 30)
NOISE_LEVEL = 1.0
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
        mom.MoM_CG_FFT(tolerance=0.01, maximum_iterations=2500),
        reg.Tikhonov(reg.TIK_FIXED, parameter=0.1),
        stp.StopCriteria(max_iterations=5))

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


def sum_approximation(scattered_field, incident_field, GS, GD, recover_resolution):
    NM, NS = scattered_field.shape
    N_pixels = incident_field.shape[0]


    A = np.zeros((NM * NS, N_pixels), dtype=complex)

    b = scattered_field.reshape(-1, 1, order='F')

    for s in range(NS):
        E_inc_s = incident_field[:, s:s+1]


        A_s = GS * E_inc_s.T

        A[s * NM : (s + 1) * NM, :] = A_s

    gamma = 1e-2
    A_reg = A.conj().T @ A + (gamma ** 2) * np.eye(N_pixels)
    b_reg = A.conj().T @ b

    chi_flat = np.linalg.solve(A_reg, b_reg)

    E_recover = (A @ chi_flat).reshape(NM, NS, order='F')

    return E_recover, chi_flat.reshape(recover_resolution)


algorithms = [
    born_approximation,
    born_iterative_method,
    contrast_source_inversion,
    sum_approximation,
]

algorithm_names = [
    'Born Approximation',
    'Born Iterative Method',
    'Contrast Source Inversion',
    'Sum Approximation'
]


# ============================================================================
# CONFIGURAÇÃO DOS TESTES
# ============================================================================
# A variação dos cenários é feita pelo próprio TestSet.
# Não são criadas configurações manuais de shape, noise, fontes, medições etc.
SAMPLE_SIZE = 60


print('=' * 70)
print('BENCHMARK GENERATOR - API EVALUATE')
print('=' * 70)

print('\nCreating test set...')

mytestset = ts.TestSet(
    name="benchmark_tests",
    wavelength=WAVELENGTH,
    image_size=(Lx, Ly),
    observation_radius=OBSERVATION_RADIUS,
    resolution=RESOLUTION,
    noise_level=NOISE_LEVEL,
    sample_size=SAMPLE_SIZE
)

mytestset.randomize_tests(
    parallelization=True,
    vary_noise=True
)

print(f'Test set created: {mytestset.sample_size} test cases.')
print(f'Condition: {mytestset._testset_condition}')

print('\nCreating benchmark...')

mybenchmark = bmk.Benchmark(
    name="api_benchmark",
    algorithm=algorithms,
    testset=mytestset
)

print(f'Benchmark: {mybenchmark.name}')
print(f'Algorithms: {len(algorithms)}')
print(f'Test set: {mytestset.name}')

print('\nExecuting benchmark...')
print('This may take a while. Please wait...')

try:
    mybenchmark.run(parallelization=bmk.PARALLELIZE_TESTS)
    print('Benchmark completed successfully!')
except Exception as e:
    print(f'Error during benchmark execution: {e}')
    print('Saving partial results...')

print('\nResults:')
if hasattr(mybenchmark.results, 'shape'):
    print(f'Results shape: {mybenchmark.results.shape}')
else:
    print('Results shape: N/A')

if mybenchmark.results is not None:
    if isinstance(mybenchmark.results, np.ndarray):
        print(f'Format: {mybenchmark.results.shape}')
        if mybenchmark.results.size > 0:
            first_result = mybenchmark.results.flat[0]
            if hasattr(first_result, 'indicators'):
                print(f'Available indicators: {list(first_result.indicators.keys())}')
    else:
        print(f'Type: {type(mybenchmark.results)}')

print('\nSaving results...')
mybenchmark.save(save_testset=True)
print(f'Results saved to: {mybenchmark.name}')

print('\n' + '=' * 70)
print('Done!')
print('=' * 70)