import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eispy2d.api.api import EvaluateS
from eispy2d.experiments import testset as ts
from eispy2d.experiments import benchmark as bmk

from eispy2d.solvers.inverse import bornapprox as ba
from eispy2d.solvers.inverse import bim
from eispy2d.solvers.inverse import csi
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.utils import stopcriteria as stp


WAVELENGTH = 1.0
Lx, Ly = 0.8, 0.8
OBSERVATION_RADIUS = 1.0
RESOLUTION = (60, 60)
NOISE_LEVEL = 1.0
BACKGROUND_PERMITTIVITY = 4.0
SAMPLE_SIZE = 30

SHAPES = ["triangle","square","circle","ellipse","cross","star4","star5",
          "star6","rhombus","trapezoid","polygon","random","random_gaussians",
          "ring","parallelogram"]
BACKGROUND_PERMITTIVITIES = [1.,2.,4.,6.,8.,10.,12.,14.]
NOISE_LEVELS = [0.,0.5,1.,1.5,2.,2.5,3.,4.,5.]
NUMBER_VALUES = [4,8,16,24,32,40,48]
CENTERS = [(0.,0.),(-0.1,0.),(0.1,0.),(0.,-0.1),(0.,0.1)]
ROTATIONS = [0.,30.,60.,90.]
OBSERVATION_RADII = [0.8,1.,1.2,1.5,1.8]
WAVELENGTHS = [0.75,1.,1.25,1.5,1.75,2.]

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

raw_algorithms = [born_approximation, born_iterative_method,
                  contrast_source_inversion, manual_born_approximation]

algorithm_names = ['Born Approximation', 'Born Iterative Method',
                   'Contrast Source Inversion', 'Manual Born Approximation']

algorithms = [EvaluateS(fn) for fn in raw_algorithms]


def build_configurations():
    configs = []
    configs.extend({"shape": v} for v in SHAPES)
    configs.extend({"background_permittivity": v} for v in BACKGROUND_PERMITTIVITIES)
    configs.extend({"noise_level": v} for v in NOISE_LEVELS)
    configs.extend(
        {"number_measurements": nm, "number_sources": ns}
        for nm in NUMBER_VALUES for ns in NUMBER_VALUES
    )
    configs.extend({"center": list(v)} for v in CENTERS)
    configs.extend({"rotate": v} for v in ROTATIONS)
    configs.extend({"observation_radius": v} for v in OBSERVATION_RADII)
    configs.extend({"wavelength": v} for v in WAVELENGTHS)
    return configs


print('=' * 70)
print('BENCHMARK GENERATOR - GENERIC BENCHMARK')
print('=' * 70)

mytestset = ts.TestSet(
    name="benchmark_tests",
    wavelength=WAVELENGTH,
    image_size=(Lx, Ly),
    observation_radius=OBSERVATION_RADIUS,
    resolution=RESOLUTION,
    noise_level=NOISE_LEVEL,
    shape="triangle",
    center=[0.0, 0.0],
    rotate=0.0,
    sample_size=SAMPLE_SIZE,
)
mytestset.randomize_tests(parallelization=True, vary_noise=False)

configurations = build_configurations()

mybenchmark = bmk.Benchmark(
    name="api_benchmark",
    algorithm=algorithms,
    testset=mytestset,
    configurations=configurations,
)

print(f'Benchmark: {mybenchmark.name}')
print(f'Algorithms: {len(algorithms)}')
print(f'Test set: {mytestset.name}')
print(f'Configurations: {len(configurations)}')

try:
    mybenchmark.run(parallelization=bmk.PARALLELIZE_TESTS)
    print('Benchmark completed successfully!')
except Exception as e:
    print(f'Error during benchmark execution: {e}')
    print('Saving partial results...')

print('\nResults:')
results = np.asarray(mybenchmark.results, dtype=object)
print(f'Results shape: {results.shape}')
mybenchmark.save(save_testset=True)
print(f'Results saved to: {mybenchmark.name}')
