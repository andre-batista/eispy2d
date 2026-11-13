import sys
import os
import time
import pickle
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from eispy2d import api
from eispy2d.core import configuration as cfg, inputdata as ipt
from eispy2d.core import result as rst
from eispy2d.discretization import richmond as ric
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.solvers.inverse import bim
from eispy2d.regularization import regularization as reg
from eispy2d.utils import stopcriteria as stp

WAVELENGTH = 1.0
IMAGE_SIZE = (0.8, 0.8)
OBSERVATION_RADIUS = 1.0
RESOLUTION = (30, 30)
NOISE_LEVEL = 1.0
NUMBER_MEASUREMENTS = 10
NUMBER_SOURCES = 10
BACKGROUND_PERMITTIVITY = 4.0
SHAPE = 'triangle'
CONTRAST = 1.0
OBJECT_SIZE = 0.16

Lx, Ly = IMAGE_SIZE

MOM_MAX_ITERATIONS = 2500
BIM_MAX_ITERATIONS = 5

STOCHASTIC_RUNS = 30
SAVE_RESULTS = True
OUTPUT_NAME = 'cst_bim_sensitivity'
OUTPUT_PATH = ''

PROBLEM = 'problem'
ALGORITHM = 'algorithm'

WAVELENGTH_VALUES = np.round(np.arange(0.2, 3.0 + 1e-9, 0.2), 1)
NOISE_VALUES = np.round(np.arange(0.5, 10.0 + 1e-9, 0.5), 1)
SOURCE_VALUES = np.arange(4, 64 + 1, 4)
MEASUREMENT_VALUES = np.arange(4, 64 + 1, 4)
MOM_MAX_ITER_VALUES = np.arange(100, 10000 + 1, 100)
STOP_MAX_ITER_VALUES = np.arange(1, 9 + 1, 1)
TIKHONOV_VALUES = np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1])

GROUPS = [
    ('wavelength', PROBLEM, 'wavelength', WAVELENGTH_VALUES),
    ('noise', PROBLEM, 'noise_level', NOISE_VALUES),
    ('sources', PROBLEM, 'number_sources', SOURCE_VALUES),
    ('measurements', PROBLEM, 'number_measurements', MEASUREMENT_VALUES),
    ('mom_max_iterations', ALGORITHM, 'mom_max_iter', MOM_MAX_ITER_VALUES),
    ('bim_max_iterations', ALGORITHM, 'stop_max_iter', STOP_MAX_ITER_VALUES),
    ('tikhonov', ALGORITHM, 'reg_tik', TIKHONOV_VALUES),
]

EPAD = 'epad'
RESIDUAL_NORM = 'residual_norm'
TIME = 'time'
RESULTS = 'results'
ERRORS = 'errors'


def born_iterative_method(
    scattered_field, incident_field, GS, GD, recover_resolution,
    mom_max_iter=2500, stop_max_iter=5, reg_tik=None
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
        config, recover_resolution, state=False
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
        inputdata, discretization, print_info=False
    )
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi


class TimedAlgorithm:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.elapsed = None
        self.__name__ = 'born_iterative_method'

    def __call__(self, scattered_field, incident_field, GS, GD,
                 recover_resolution):
        start = time.perf_counter()
        output = born_iterative_method(
            scattered_field, incident_field, GS, GD, recover_resolution,
            **self.kwargs
        )
        self.elapsed = time.perf_counter() - start
        return output


def build_params(overrides=None):
    params = dict(
        wavelength=WAVELENGTH,
        image_size=IMAGE_SIZE,
        number_measurements=NUMBER_MEASUREMENTS,
        number_sources=NUMBER_SOURCES,
        observation_radius=OBSERVATION_RADIUS,
        background_permittivity=BACKGROUND_PERMITTIVITY,
        resolution=RESOLUTION,
        noise_level=NOISE_LEVEL,
        shape=SHAPE,
        contrast=CONTRAST,
        object_size=OBJECT_SIZE,
        center=[0.0, 0.0],
        rotate=0.0,
        disp=False,
    )
    params.update(overrides or {})
    return params


def build_algorithm_kwargs(overrides=None):
    kwargs = dict(
        mom_max_iter=MOM_MAX_ITERATIONS,
        stop_max_iter=BIM_MAX_ITERATIONS,
    )
    kwargs.update(overrides or {})
    return kwargs


def run_single(params, algorithm_kwargs):
    algorithm = TimedAlgorithm(**algorithm_kwargs)
    try:
        result = api.evaluate(algorithm, params)
    except Exception as exception:
        return dict(epad=np.nan, residual_norm=np.nan,
                    time=np.nan, result=None, error=repr(exception))
    return dict(
        epad=result.final_value(rst.REL_PERMITTIVITY_PAD_ERROR),
        residual_norm=result.final_value(rst.RESIDUAL_NORM_ERROR),
        time=algorithm.elapsed,
        result=result if SAVE_RESULTS else None,
        error=None,
    )


def run_test(params, algorithm_kwargs, runs, number_jobs):
    outputs = Parallel(n_jobs=number_jobs)(
        delayed(run_single)(params, algorithm_kwargs) for _ in range(runs)
    )
    return {
        EPAD: np.array([o['epad'] for o in outputs], dtype=float),
        RESIDUAL_NORM: np.array([o['residual_norm'] for o in outputs],
                                dtype=float),
        TIME: np.array([o['time'] for o in outputs], dtype=float),
        RESULTS: [o['result'] for o in outputs],
        ERRORS: [o['error'] for o in outputs],
    }


def run_group(name, target, key, values, runs, number_jobs):
    group_output = {}
    for index, value in enumerate(values):
        value = value.item() if hasattr(value, 'item') else value
        if target == PROBLEM:
            params = build_params({key: value})
            algorithm_kwargs = build_algorithm_kwargs()
        else:
            params = build_params()
            algorithm_kwargs = build_algorithm_kwargs({key: value})
        group_output[value] = run_test(params, algorithm_kwargs, runs,
                                       number_jobs)
        failures = sum(e is not None for e in group_output[value][ERRORS])
        print('%s: test %d/%d (%s=%s) | mean EPAD: %.3f | mean time: %.2fs'
              ' | failures: %d'
              % (name, index + 1, len(values), key, value,
                 np.nanmean(group_output[value][EPAD]),
                 np.nanmean(group_output[value][TIME]), failures),
              flush=True)
    return group_output


def main():
    number_jobs = multiprocessing.cpu_count()
    number_tests = sum(len(values) for _, _, _, values in GROUPS)
    print('Tests: %d | Runs per test: %d | Total runs: %d'
          % (number_tests, STOCHASTIC_RUNS, number_tests * STOCHASTIC_RUNS))

    study = {
        'name': OUTPUT_NAME,
        'reference_params': build_params(),
        'reference_algorithm': build_algorithm_kwargs(),
        'stochastic_runs': STOCHASTIC_RUNS,
        'groups': {},
    }

    for name, target, key, values in GROUPS:
        study['groups'][name] = dict(
            target=target,
            key=key,
            values=[v.item() if hasattr(v, 'item') else v for v in values],
            output=run_group(name, target, key, values, STOCHASTIC_RUNS,
                             number_jobs),
        )

    with open(OUTPUT_PATH + OUTPUT_NAME, 'wb') as datafile:
        pickle.dump(study, datafile)


if __name__ == '__main__':
    main()