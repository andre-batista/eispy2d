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

WAVELENGTH_VALUES = [i*0.2 for i in range(1, 16)]
NOISE_VALUES = [i*0.5 for i in range(1, 21)]
NS_VALUES = [ i for i in range(4, 65, 4)]
NM_VALUES = [ i for i in range(4, 65, 4)]

MAX_ITER_VALUES = [i for i in range(100, 10001, 100)]
REG_TIK_VALUES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]


def born_iterative_method(
    scattered_field,
    incident_field,
    GS,
    GD,
    recover_resolution,
    max_iter=2500,
    reg_tik=None
):
    if reg_tik is None:
        reg_tik = reg.TIK_FIXED

    NM, NS = scattered_field.shape

    config = cfg.Configuration(
        name='temp',
        wavelength=1.0,
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
        mom.MoM_CG_FFT(),
        reg.Tikhonov(reg_tik),
        stp.StopCriteria(max_iterations=max_iter)
    )

    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1

    return result.scattered_field, chi


def build_case_study(name, variable_param, variable_values, algorithm_params=None):
    fixed_params = {
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

    study = cst.CaseStudy(
        name=name,
        algorithm=born_iterative_method,
        fixed_params=fixed_params,
        variable_param=variable_param,
        variable_values=variable_values,
        stochastic_runs=STOCHASTIC_RUNS,
        save_stochastic_runs=True,
        algorithm_params=algorithm_params
    )

    study.run(parallelization=True)
    study.save(save_test=True)

    return study


def run_input_parameter_studies():
    studies = []

    studies.append(
        build_case_study(
            name="api_casestudy_wavelength",
            variable_param="wavelength",
            variable_values=WAVELENGTH_VALUES
        )
    )

    studies.append(
        build_case_study(
            name="api_casestudy_noise",
            variable_param="noise_level",
            variable_values=NOISE_VALUES
        )
    )

    studies.append(
        build_case_study(
            name="api_casestudy_sources",
            variable_param="number_sources",
            variable_values=NS_VALUES
        )
    )

    studies.append(
        build_case_study(
            name="api_casestudy_measurements",
            variable_param="number_measurements",
            variable_values=NM_VALUES
        )
    )

    return studies


def build_algorithm_case_study(name, algorithm_param, values):
    fixed_params = {
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

    tests = [fixed_params.copy() for _ in values]
    algorithm_params = [
        {algorithm_param: value}
        for value in values
    ]

    study = cst.CaseStudy(
        name=name,
        algorithm=born_iterative_method,
        test=tests,
        stochastic_runs=STOCHASTIC_RUNS,
        save_stochastic_runs=True,
        algorithm_params=algorithm_params
    )

    study.run(parallelization=True)
    study.save(save_test=True)

    return study


def run_algorithm_parameter_studies():
    studies = []

    studies.append(
        build_algorithm_case_study(
            name="api_casestudy_max_iter",
            algorithm_param="max_iter",
            values=MAX_ITER_VALUES
        )
    )

    studies.append(
        build_algorithm_case_study(
            name="api_casestudy_reg_tik",
            algorithm_param="reg_tik",
            values=REG_TIK_VALUES
        )
    )

    return studies


input_parameter_studies = run_input_parameter_studies()
algorithm_parameter_studies = run_algorithm_parameter_studies()