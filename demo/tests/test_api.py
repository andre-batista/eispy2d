import sys
import os
import unittest
import numpy as np
from numpy.linalg import inv
from scipy import sparse as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eispy2d.api import api
from eispy2d.discretization import richmond as ric
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.core import configuration as cfg, result
from eispy2d.core import result as rst
from eispy2d.core import inputdata as ipt
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.utils import stopcriteria as stp, draw
from eispy2d.solvers.inverse import bim
from eispy2d.solvers.inverse import backprop


def test_evaluate(scattered_field, incident_field, GS, GD, recover_resolution):
    contrast = 0
    recon_scattered_field = 0

    NS = scattered_field.shape[1] #sources
    NM = scattered_field.shape[0] #measurements    resolution = (int(np.sqrt(N)), int(np.sqrt(N)))

    contrast = np.zeros(recover_resolution, dtype=complex) 
    recon_scattered_field = scattered_field.copy() 

    # for it in range(2):
    #     E_tot = np.zeros((N, NM), dtype=complex)
    #     for m in range(NM):
    #         A_total = np.eye(N) - GD @ np.diag(contrast)
    #         E_tot[:, m] = np.linalg.solve(A_total, incident_field[:, m])
        
    #     for m in range(NM):
    #         E_tot_col = E_tot[:, m]  # (3600,)
    #         fonte = contrast * E_tot_col  # (3600,)
            
    #         recon_scattered_field[:, m] = GS @ fonte  # (9, 3600) @ (3600,) = (9,) 
        
    #     erro = np.linalg.norm(scattered_field - recon_scattered_field) / np.linalg.norm(recon_scattered_field)
    #     print(f"Iteração {it}: Erro = {erro:.6f}")
        
        
        
    #     delta_contrast = np.zeros(N, dtype=complex)
    #     for m in range(NM):
    #         A = GS @ np.diag(E_tot[:, m])  # (9, 3600)
            
    #         res = scattered_field[:, m] - recon_scattered_field[:, m]  # (9,)
            
    #         lambda_reg = 0.01  
    #         AHA = A.conj().T @ A + lambda_reg * np.eye(N)
    #         rhs = A.conj().T @ res
            
    #         delta_contrast += np.linalg.solve(AHA, rhs)
        
    #     contrast = contrast + delta_contrast / NM

    print("TEST contrast first 5:", contrast[:5])
    print("TEST mean contrast:", np.mean(np.real(contrast)).item())

    print("TEST recon scattered first 5:", recon_scattered_field[:5, 0])


    return recon_scattered_field, contrast






def alg(scattered_field, incident_field, GS, GD, resolution):
    # NM, NS = scattered_field.shape
    # Lx, Ly = resolution[0] / 75 , resolution[1] / 75 
    # E0 = np.max(np.abs(incident_field)) 
    
    # config = cfg.Configuration(
    #     name='temp_config',
    #     frequency=3e8,
    #     number_measurements=NM,
    #     number_sources=NS,
    #     image_size=[Ly, Lx],
    #     observation_radius=1.0,
    #     background_permittivity=4.0,
    #     magnitude=E0,
    #     perfect_dielectric=True
    # )
    
    # discretization = ric.Richmond(config, resolution, state=False)
    
    # inputdata = ipt.InputData(
    #     name='temp_input',
    #     configuration=config,
    #     resolution=resolution,
    #     noise=1.,
    #     scattered_field=scattered_field,
    #     incident_field=incident_field,
    #     indicators=[rst.REL_PERMITTIVITY_PAD_ERROR, rst.OBJECTIVE_FUNCTION]
    # )

    # inputdata.rel_permittivity, _ = draw.triangle(
    #     .16*np.sqrt(3),
    #     center=[-.14, .09],
    #     axis_length_x=config.Lx,
    #     axis_length_y=config.Ly,
    #     resolution=resolution,
    #     background_rel_permittivity=4.0,
    #     object_rel_permittivity=(1.0+1)*4.0
    # )
    
    # method = bim.BornIterativeMethod(
    #     forward_solver=mom.MoM_CG_FFT(tolerance=0.01, maximum_iterations=2500),
    #     regularization=reg.Tikhonov(reg.TIK_FIXED, parameter=0.1),
    #     stop_criteria=stp.StopCriteria(max_iterations=5)
    # )
    
    # result = method.solve(inputdata, discretization)
    
    # chi = result.rel_permittivity
    # if chi.ndim == 2:
    #     chi = chi.flatten()

    # chi = (result.rel_permittivity / config.epsilon_rb) - 1

    # print("CONTRAST", chi)
    # print("EPAD", result.zeta_epad)

    # #scattered_field = result.scattered_field

    # print("Scattered field shape: ", scattered_field.shape)
    # print("Incident field shape: ", incident_field.shape)
    # print("GS shape: ", GS.shape)
    # print("GD shape: ", GD.shape)

    # print("Contrast shape: ", chi.shape)
    # print("Recon scattered field shape: ", result.scattered_field.shape)

    # print("ALG contrast first 5:", chi[:5])
    # print("ALG mean contrast:", np.mean(np.real(chi)).item())

    # print("ALG recon scattered first 5:", result.scattered_field[:5, 0])

    chi = np.zeros(resolution, dtype=complex)
    N = resolution[0] * resolution[1]
    C = sps.spdiags(chi.reshape(-1), 0, N, N)
    I = np.eye(N, dtype=complex)
    L = inv(I - GD@C)

    recon_scattered_field = GS @ C @ L @ incident_field

    return recon_scattered_field, chi

params = {"shape":"random", "disp":True}
api.evaluate(alg, params)