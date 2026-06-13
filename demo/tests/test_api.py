import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eispy2d.api import api
from eispy2d.solvers.inverse import backprop
from eispy2d.discretization import richmond
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.core import inputdata as ipt
from eispy2d.solvers.inverse import regularization as reg_lib

def alg(scattered_field, incident_field, GS, GD, reg=1e-3):
    Es = np.asarray(scattered_field) 
    Ei = np.asarray(incident_field)  
    NM, NS = Es.shape
    Nd, N = GD.shape
    
    A = np.zeros((NM * NS, N), dtype=complex)
    for s in range(NS):
        A[s * NM:(s + 1) * NM, :] = GS @ np.diag(Ei[:, s])
    
    Es_flat = Es.flatten()
    AH = A.conj().T
    I = np.eye(N, dtype=complex)
    chi = np.linalg.solve(AH @ A + reg * I, AH @ Es_flat)
    
    
    return scattered_field, chi


def alg2(scattered_field, incident_field, GS, GD, reg=1e-3):
    
    config = cfg.Configuration(
        name='alg2_config',
        wavelength=1.0,
        number_measurements=scattered_field.shape[0],
        number_sources=scattered_field.shape[1],
        image_size=[2.0, 2.0],
        background_permittivity=1.0
    )

    resolution = (int(np.sqrt(GD.shape[0])), int(np.sqrt(GD.shape[0])))
    discretization = richmond.Richmond(config, resolution)

    
    reg_solver = reg_lib.Tikhonov(choice='fixed', parameter=reg)
    
    chi = discretization.solve(
        scattered_field=scattered_field,
        total_field=incident_field,
        linear_solver=reg_solver
    )
    
    return scattered_field, chi

import numpy as np

def alg3(scattered_field, incident_field, GS, GD):
    
    n_dominio = GS.shape[1]
    
    A_operador = np.einsum('sd, df -> sfd', GS, incident_field)
    
    A_flat = A_operador.reshape(-1, n_dominio)
    
    b_flat = scattered_field.flatten()
    

    lambda_reg = 1e-2 
    
    A_H = A_flat.conj().T
    I = np.eye(n_dominio)
    
    chi_flat = np.linalg.solve(A_H @ A_flat + lambda_reg * I, A_H @ b_flat)
    
    chi = chi_flat 
    

    scattered_calc_flat = A_flat @ chi
    scattered_field_calc = scattered_calc_flat.reshape(scattered_field.shape)
    
    return scattered_field_calc, chi

def alg4(scattered_field, incident_field, GS, GD, reg_param=1e-3, max_iter=10):
    NM, NS = scattered_field.shape
    N_pixels = GS.shape[1]
    resolution = (int(np.sqrt(N_pixels)), int(np.sqrt(N_pixels)))
    
    config = cfg.Configuration(
        name='alg4_config',
        wavelength=1.0,
        number_measurements=NM,
        number_sources=NS,
        image_size=[2.0, 2.0],
        background_permittivity=1.0,
        perfect_dielectric=True
    )
    
    discretization = richmond.Richmond(config, resolution)
    
    inputdata = ipt.InputData(
        name='alg4_input',
        configuration=config,
        resolution=resolution,
        scattered_field=scattered_field,
        incident_field=incident_field
    )
    
    forward_solver = mom.MoM_CG_FFT(tolerance=0.001, maximum_iterations=5000)
    regularizacao = reg.Tikhonov(choice='fixed', parameter=reg_param)
    stop = stp.StopCriteria(max_iterations=max_iter)
    
    solver_bim = bim.BornIterativeMethod(
        forward_solver=forward_solver,
        regularization=regularizacao,
        stop_criteria=stop
    )
    
    result = solver_bim.solve(inputdata, discretization)
    
    chi = result.rel_permittivity
    if chi.ndim == 2:
        chi = chi.flatten()
    
    return scattered_field, chi

api.evaluate(alg)
api.evaluate(alg2)
api.evaluate(alg3)
api.evaluate(alg4)
