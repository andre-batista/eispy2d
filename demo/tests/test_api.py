import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eispy2d.api import api
from eispy2d.discretization import richmond as ric
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.core import inputdata as ipt
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.utils import stopcriteria as stp, draw
from eispy2d.solvers.inverse import bim
from eispy2d.solvers.inverse import backprop


def alg(scattered_field, incident_field, GS, GD):
    NM, NS = scattered_field.shape
    N_pixels = GS.shape[1]
    resolution = (int(np.sqrt(N_pixels)), int(np.sqrt(N_pixels)))
    
    config = cfg.Configuration(
        name='temp_config',
        frequency=3e8,
        number_measurements=NM,
        number_sources=NS,
        image_size=[0.8, 0.8],
        background_permittivity=4.0,
        perfect_dielectric=True
    )
    
    discretization = ric.Richmond(config, resolution, state=False)
    
    inputdata = ipt.InputData(
        name='temp_input',
        configuration=config,
        resolution=resolution,
        noise=1.,
        scattered_field=scattered_field,
        incident_field=incident_field,
        indicators=[rst.REL_PERMITTIVITY_PAD_ERROR, rst.OBJECTIVE_FUNCTION]
    )

    inputdata.rel_permittivity, _ = draw.triangle(
        .16*np.sqrt(3),
        center=[-.14, .09],
        axis_length_x=config.Lx,
        axis_length_y=config.Ly,
        resolution=resolution,
        background_rel_permittivity=4.0,
        object_rel_permittivity=(1.0+1)*4.0
    )
    
    method = bim.BornIterativeMethod(
        forward_solver=mom.MoM_CG_FFT(tolerance=0.01, maximum_iterations=2500),
        regularization=reg.Tikhonov(reg.TIK_FIXED, parameter=0.1),
        stop_criteria=stp.StopCriteria(max_iterations=5)
    )
    
    result = method.solve(inputdata, discretization)
    
    chi = result.rel_permittivity
    if chi.ndim == 2:
        chi = chi.flatten()

    print("CONTRAST", chi)
    print("EPAD", result.zeta_epad)

    scattered_field = result.scattered_field
    
    return scattered_field, chi

def alg2(scattered_field, incident_field, GS, GD, max_iter=10, tol=1e-6, reg_param=0.1):
    NM, NS = scattered_field.shape
    Npix = incident_field.shape[0]

    A_blocks = []
    b_blocks = []

    for s in range(NS):
        Es = incident_field[:, s]          
        As = GS @ np.diag(Es)              

        A_blocks.append(As)
        b_blocks.append(scattered_field[:, s])

    A = np.vstack(A_blocks)                 
    b = np.concatenate(b_blocks)            

    # Estima o contraste
    chi = np.linalg.pinv(A) @ b

    # Reconstrói o campo espalhado
    scattered_est = np.zeros_like(scattered_field, dtype=complex)

    for s in range(NS):
        scattered_est[:, s] = GS @ (chi * incident_field[:, s])

    # EPAD
    epad_error = (
        np.linalg.norm(scattered_field - scattered_est)
        / np.linalg.norm(scattered_field)
        * 100
    )

    print(f"EPAD: {epad_error}%")

    return scattered_est, chi


def alg3(scattered_field, incident_field, GS, GD):
    NM, NS = scattered_field.shape
    N_pixels = GS.shape[1]
    resolution = (int(np.sqrt(N_pixels)), int(np.sqrt(N_pixels)))
    
    config = cfg.Configuration(
        name='temp_config',
        frequency=3e8,
        number_measurements=NM,
        number_sources=NS,
        image_size=[0.8, 0.8],
        background_permittivity=4.0,
        perfect_dielectric=True
    )
    
    discretization = ric.Richmond(config, resolution, state=False)
    
    inputdata = ipt.InputData(
        name='temp_input',
        configuration=config,
        resolution=resolution,
        noise=1.,
        scattered_field=scattered_field,
        incident_field=incident_field,
        indicators=[rst.REL_PERMITTIVITY_PAD_ERROR]
    )

    inputdata.rel_permittivity, _ = draw.triangle(
        .16*np.sqrt(3),
        center=[-.14, .09],
        axis_length_x=config.Lx,
        axis_length_y=config.Ly,
        resolution=resolution,
        background_rel_permittivity=4.0,
        object_rel_permittivity=(1.0+1)*4.0
    )

    solver = backprop.BackPropagation()
    result = solver.solve(inputdata, discretization)
    chi = result.rel_permittivity.flatten()

    print(result.zeta_epad)

    return scattered_field, chi




api.evaluate(alg)