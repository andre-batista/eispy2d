import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eispy2d.api import testset_api as ts
from eispy2d.api import casestudy_api as cst
from eispy2d.core import configuration as cfg
from eispy2d.core import inputdata as ipt
from eispy2d.discretization import richmond as ric
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.solvers.inverse import bornapprox as ba
from eispy2d.solvers.inverse import bim
from eispy2d.solvers.inverse import csi
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.utils import stopcriteria as stp


WAVELENGTH = 1.0
Lx, Ly = 0.8, 0.8
OBSERVATION_RADIUS = 1.0
RESOLUTION = (60, 60)
NOISE_LEVEL = 1.0
BACKGROUND_PERMITTIVITY = 4.0
CONTRAST_LEVEL = 1.0
OBJECT_SIZE = 0.16
STOCHASTIC_RUNS = 30
MAX_ITERATIONS = 100


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
        mom.MoM_CG_FFT(),
        reg.Tikhonov(1e-1),
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
        stp.StopCriteria(max_iterations=MAX_ITERATIONS)
    )
    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi


algorithms = [
    born_approximation,
    born_iterative_method,
    contrast_source_inversion
]


print('Creating test case...')

test_params = {
    "wavelength": WAVELENGTH,
    "image_size": (Lx, Ly),
    "observation_radius": OBSERVATION_RADIUS,
    "resolution": RESOLUTION,
    "noise_level": NOISE_LEVEL,
    "shape": "triangle",
    "background_permittivity": BACKGROUND_PERMITTIVITY
}

print(f'Test parameters: {test_params}')

print('\nCreating case study...')

mycasestudy = cst.CaseStudy(
    name="api_casestudy",
    algorithm=algorithms,
    test=test_params,
    stochastic_runs=STOCHASTIC_RUNS,
    save_stochastic_runs=True
)

print(f'Case study created: {mycasestudy.name}')
print(f'Algorithms: {len(algorithms)}')
print(f'Stochastic runs: {STOCHASTIC_RUNS}')

print('\nExecuting case study...')
mycasestudy.run(parallelization=True)

print('Case study completed!')


print('\nResults:')
if mycasestudy.results is not None:
    if isinstance(mycasestudy.results, list):
        print(f'Number of algorithm results: {len(mycasestudy.results)}')
        for i, algo_result in enumerate(mycasestudy.results):
            if isinstance(algo_result, list):
                print(f'  Algorithm {i+1}: {len(algo_result)} stochastic executions')
                if len(algo_result) > 0:
                    first = algo_result[0]
                    if hasattr(first, 'indicators'):
                        print(f'    Available indicators: {list(first.indicators.keys())}')
            else:
                print(f'  Algorithm {i+1}: single execution')
    else:
        print(f'Type of results: {type(mycasestudy.results)}')

print('\nSaving results...')
mycasestudy.save(save_test=True)
print(f'Results saved to: {mycasestudy.name}')