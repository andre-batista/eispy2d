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

from demo.ml.bayesnetwork import BayesNetwork as bn

WAVELENGTH = 1.0
Lx, Ly = 4.0, 4.0
OBSERVATION_RADIUS = 1.0
RESOLUTION = (30, 30)
NOISE_LEVEL = 1.0
BACKGROUND_PERMITTIVITY = 4.0

N_REPETITIONS = 30

BASE_PARAMS = {
    "wavelength": WAVELENGTH,
    "image_size": (Lx, Ly),
    "number_measurements": 10,
    "number_sources": 10,
    "observation_radius": OBSERVATION_RADIUS,
    "background_permittivity": BACKGROUND_PERMITTIVITY,
    "resolution": RESOLUTION,
    "noise_level": 1.0,
    "disp": False,
    "shape": "triangle",
    "center": [0.0, 0.0],
    "rotate": 0.0,
    "contrast": 0.25,
    "object_size": 1.0,
    "calculate_dnl": True
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
    (5, 20),
    (10, 5),
    (10, 10),
    (10, 15),
    (10, 20),
    (15, 5),
    (15, 10),
    (15, 15),
    (15, 20),
    (20, 5),
    (20, 10),
    (20, 15),
    (20, 20)
]

CONTRAST_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

RESOLUTION_VALUES = [(15, 15), (30, 30), (45, 45), (60, 60), (100, 100)]

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
    solver = ba.FirstOrderBornApproximation(reg.Tikhonov(1e-1))
    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi


def born_iterative_method(scattered_field, incident_field, GS, GD, recover_resolution):
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

def bayes(scattered_field, incident_field, GS, GD, recover_resolution):
    net = bn.load("demo/ml/data/bayes_data.pkl")
    return net.solve(scattered_field, incident_field, GS, GD, recover_resolution)

ALGORITHMS = [
    born_iterative_method,
    born_approximation,
    contrast_source_inversion,
    bayes
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
    
    # Phase 5: vary only the contrast.
    for contrast in CONTRAST_VALUES:
        params = BASE_PARAMS.copy()
        params["contrast"] = contrast
        name = f"contrast_{contrast}"
        testsets.append(make_testset(name, params, repetitions))

    # Phase 6: vary only the resolution.
    for resolution in RESOLUTION_VALUES:
        params = BASE_PARAMS.copy()
        params["resolution"] = resolution
        name = f"resolution_{resolution[0]}x{resolution[1]}"
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

    benchmark.run(parallelization=bmk.PARALLELIZE_TESTS)

    save_path = str(OUTPUT_DIRECTORY) + os.sep
    benchmark.save(file_path=save_path, save_testset=True)

    print(f"Benchmark saved to: {save_path}{BENCHMARK_NAME}")
    return benchmark


if __name__ == "__main__":
    run_benchmark()
