import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

import numpy as np

from bayesnetwork import BayesNetwork

params = {
    "wavelength": 1.0,
    "image_size": (2.0, 2.0),
    "observation_radius": 6.0,
    "background_permittivity": 1.0,
    "resolution": (60, 60),
    "noise_level": 1.0,    
    "disp": True,
    "contrast": 0.25,
    "object_size": 1.0,
    "number_sources": 30,
    "number_measurements": 30,
    "shape": "random",
    "center": [0.0, 0.0],
}



#net = BayesNetwork.load("data/bayes_data_6.pkl")
net = BayesNetwork.load("data/bayes_data.pkl")
net.DECISION = "mean"

api.evaluate(net.solve, params)