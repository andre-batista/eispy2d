import sys
import os

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
STOCHASTIC_RUNS = 30

WAVELENGTH_VALUES = [i * 0.2 for i in range(1, 16)]
NOISE_VALUES = [i * 0.5 for i in range(1, 21)]
NS_VALUES = [i for i in range(4, 65, 4)]
NM_VALUES = [i for i in range(4, 65, 4)]

MOM_MAX_ITER_VALUES = [i for i in range(100, 10001, 100)]
STOP_MAX_ITER_VALUES = [i for i in range(1, 10, 1)]
REG_TIK_VALUES = [
    1e-4, 5e-4, 1e-3, 5e-3,
    1e-2, 5e-2, 1e-1, 5e-1
]

CASE_STUDY_NAME = "api_casestudy"


def born_iterative_method(
    scattered_field,
    incident_field,
    GS,
    GD,
    recover_resolution,
    mom_max_iter=2500,
    stop_max_iter=5,
    reg_tik=None
):
    if reg_tik is None:
        reg_tik = reg.TIK_FIXED

    NM, NS = scattered_field.shape

    config = cfg.Configuration(
        name="temp",
        wavelength=1.0,
        number_measurements=NM,
        number_sources=NS,
        image_size=[Lx, Ly],
        observation_radius=OBSERVATION_RADIUS,
        background_permittivity=BACKGROUND_PERMITTIVITY,
        perfect_dielectric=True
    )

    discretization = ric.Richmond(
        config,
        recover_resolution,
        state=False
    )

    inputdata = ipt.InputData(
        name="temp",
        configuration=config,
        resolution=recover_resolution,
        scattered_field=scattered_field,
        incident_field=incident_field,
        indicators=[]
    )

    solver = bim.BornIterativeMethod(
        mom.MoM_CG_FFT(tolerance=0.01, maximum_iterations=mom_max_iter),
        reg.Tikhonov(reg_tik, parameter=0.1),
        stp.StopCriteria(max_iterations=stop_max_iter)
    )

    result = solver.solve(
        inputdata,
        discretization,
        print_info=False
    )

    chi = (result.rel_permittivity / config.epsilon_rb) - 1

    return result.scattered_field, chi


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
        ("mom_max_iter", MOM_MAX_ITER_VALUES, "mom_max_iter"),
        ("stop_max_iter", STOP_MAX_ITER_VALUES, "stop_max_iter"),
        ("reg_tik", REG_TIK_VALUES, "reg_tik")
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
        algorithm=born_iterative_method,
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