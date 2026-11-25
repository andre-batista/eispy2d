import sys

sys.path.insert(1, '../eispy2d')

import copy
import json
import os
from pathlib import Path

from eispy2d.api import api
from eispy2d.experiments import casestudy as cst

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

N_REPETITIONS = 30

PARAMS = {
    "wavelength": 1.0,
    "image_size": (2.0, 2.0),
    "observation_radius": 6.0,
    "background_permittivity": 4.0,
    "resolution": (100, 100),
    "noise_level": 1.0,
    "number_measurements": 25,
    "number_sources": 25,
    "shape": "star5",
    "contrast": 1.0,
    "object_size": 1.0,
    "center": [0.0, 0.0],
    "rotate": 0.0,
    "disp": False,
    "calculate_dnl": True
}

FORWARD_KWARGS = {
    "tolerance": 0.001,
    "maximum_iterations": 1000,
}

BAYES_DATA_DIRECTORY = "demo/ml/data/"
BAYES_DATA_NAMES = [
    "bayes_data_1000_80",
    "bayes_data_1000_800",
    "bayes_data_100000_80",
    "bayes_data_100000_800",
]

DECISIONS = ["mean", "map", "sample"]

OUTPUT_DIRECTORY = Path("api_case_study_results")
CASE_STUDY_NAME = "api_bayes_case_study"


def build_configuration(scattered_field):
    NM, NS = scattered_field.shape
    Lx, Ly = PARAMS["image_size"]
    return cfg.Configuration(
        name='temp',
        wavelength=PARAMS["wavelength"],
        number_measurements=NM,
        number_sources=NS,
        image_size=[Lx, Ly],
        observation_radius=PARAMS["observation_radius"],
        background_permittivity=PARAMS["background_permittivity"],
        perfect_dielectric=True
    )


def build_inputdata(config, scattered_field, incident_field, recover_resolution):
    return ipt.InputData(
        name='temp',
        configuration=config,
        resolution=recover_resolution,
        scattered_field=scattered_field,
        incident_field=incident_field,
        indicators=[]
    )


def born_approximation(scattered_field, incident_field, GS, GD, recover_resolution):
    config = build_configuration(scattered_field)
    discretization = ric.Richmond(config, recover_resolution, state=False)
    inputdata = build_inputdata(
        config, scattered_field, incident_field, recover_resolution
    )
    solver = ba.FirstOrderBornApproximation(reg.Tikhonov(1e-1))
    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi


def born_iterative_method(scattered_field, incident_field, GS, GD, recover_resolution):
    config = build_configuration(scattered_field)
    discretization = ric.Richmond(config, recover_resolution, state=False)
    inputdata = build_inputdata(
        config, scattered_field, incident_field, recover_resolution
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
    config = build_configuration(scattered_field)
    discretization = ric.Richmond(config, recover_resolution, state=False)
    inputdata = build_inputdata(
        config, scattered_field, incident_field, recover_resolution
    )
    solver = csi.ContrastSourceInversion(
        stp.StopCriteria(max_iterations=100)
    )
    result = solver.solve(inputdata, discretization, print_info=False)
    chi = (result.rel_permittivity / config.epsilon_rb) - 1
    return result.scattered_field, chi


class BayesAlgorithm:

    def __init__(self, net, data_name, decision):
        self.net = net
        self.data_name = data_name
        self.decision = decision
        self.__name__ = f"{data_name}_{decision}"

    def __call__(self, scattered_field, incident_field, GS, GD, recover_resolution):
        return self.net.solve(
            scattered_field, incident_field, GS, GD, recover_resolution
        )

    def stripped(self):
        return BayesAlgorithm(None, self.data_name, self.decision)


def build_bayes_algorithms():
    algorithms = []
    for data_name in BAYES_DATA_NAMES:
        base_net = bn.load(BAYES_DATA_DIRECTORY + data_name + ".pkl")
        for decision in DECISIONS:
            net = copy.deepcopy(base_net)
            net.DECISION = decision
            algorithms.append(BayesAlgorithm(net, data_name, decision))
        del base_net
    return algorithms


def build_algorithms():
    algorithms = [
        born_approximation,
        born_iterative_method,
        contrast_source_inversion,
    ]
    algorithms.extend(build_bayes_algorithms())
    return algorithms


def strip_algorithm(algorithm):
    if isinstance(algorithm, BayesAlgorithm):
        return algorithm.stripped()
    return algorithm


def build_algorithm_info(algorithms):
    info = {}
    for algorithm in algorithms:
        if isinstance(algorithm, BayesAlgorithm):
            info[algorithm.__name__] = {
                "type": "bayes",
                "data_file": algorithm.data_name,
                "decision": algorithm.decision,
            }
        else:
            info[algorithm.__name__] = {
                "type": "classic",
                "data_file": None,
                "decision": None,
            }
    return info


def build_methods(algorithms):
    return [
        api.EvaluateS(
            algorithm=algorithm,
            forward_kwargs=FORWARD_KWARGS,
        )
        for algorithm in algorithms
    ]


def make_inputdata(repetition):
    _, inputdata, _, _ = api.prepare_problem(
        params=PARAMS,
        forward_kwargs=FORWARD_KWARGS,
    )
    inputdata.name = f"{CASE_STUDY_NAME}_{repetition:03d}"
    return inputdata


def run_case_study(repetitions=N_REPETITIONS):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    save_path = str(OUTPUT_DIRECTORY) + os.sep

    print("Loading networks...")
    algorithms = build_algorithms()
    algorithm_info = build_algorithm_info(algorithms)

    with open(OUTPUT_DIRECTORY / "algorithm_info.json", "w") as info_file:
        json.dump(algorithm_info, info_file, indent=4)

    print(f"Algorithms: {[algorithm.__name__ for algorithm in algorithms]}")
    print(f"Repetitions: {repetitions}")
    print("Running case study...")

    studies = []
    for repetition in range(repetitions):
        inputdata = make_inputdata(repetition)

        study = cst.CaseStudy(
            name=f"{CASE_STUDY_NAME}_{repetition:03d}",
            method=build_methods(algorithms),
            discretization=None,
            test=inputdata,
        )

        study.run(parallelization=cst.PARALLELIZE_METHOD)
        study.method = build_methods(
            [strip_algorithm(algorithm) for algorithm in algorithms]
        )
        study.save(file_path=save_path, save_test=True)
        studies.append(study)

        print(f"Repetition {repetition + 1}/{repetitions} saved to: {save_path}")

    return studies


if __name__ == "__main__":
    run_case_study()