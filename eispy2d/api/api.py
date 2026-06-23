import numpy as np
from skimage import data

from eispy2d.discretization import richmond
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.core import inputdata as ipt
from eispy2d.utils import draw
from scipy.linalg import norm

import numpy as np

import numpy as np


def evaluate(algorithim, data=None):
    if data is not None:
        if "wavelength" in data:
            wavelength = data["wavelength"]
        if "number_measurements" in data:
            number_measurements = data["number_measurements"]
        if "number_sources" in data:
            number_sources = data["number_sources"]
        if "image_size" in data:
            image_size = data["image_size"]
        if "observation_radius" in data:
            observation_radius = data["observation_radius"]
        if "background_permittivity" in data:
            background_permittivity = data["background_permittivity"]
        if "resolution" in data:
            resolution = data["resolution"]
        if "rel_permittivity" in data:
            epslon_r = data["rel_permittivity"]

    f0 = 3e8 # linear frequency [m]
    Lx, Ly = .8, .8 # D domain size [m]
    NS, NM = 10, 9 # number of sources and measurements
    RO = 1. # observation radius [m]
    epsilon_rb = 4. # background relative permittivity
    E0 = 1 # incident wave magnitude [V/m]
    resolution = (60, 60) # ground-truth image resolution [pixels]
    noise_level = 1. # [%/sample]
    indicators = [rst.REL_PERMITTIVITY_PAD_ERROR, rst.OBJECTIVE_FUNCTION]
    contrast_level = 1.
    object_size = .16 # [m]

    # Define domain and source parameters
    config = cfg.Configuration(name='cfg_test',
                            frequency=f0,
                            wavelength_unit=False,
                            number_measurements=NM,
                            number_sources=NS,
                            image_size=[Ly, Lx],
                            observation_radius=RO,
                            background_permittivity=epsilon_rb,
                            magnitude=E0,
                            perfect_dielectric=True)

    # Build test object
    inputdata = ipt.InputData(name='iptTest',
                            configuration=config,
                            resolution=resolution,
                            noise=noise_level,
                            indicators=indicators)

    # Draw figure
    inputdata.rel_permittivity, _ = draw.triangle(
        object_size*np.sqrt(3),
        center=[-.14, .09],
        axis_length_x=config.Lx,
        axis_length_y=config.Ly,
        resolution=resolution,
        background_rel_permittivity=epsilon_rb,
        object_rel_permittivity=(contrast_level+1)*epsilon_rb
    )


    # Build forward solver object
    solver = mom.MoM_CG_FFT(tolerance=.001,
                            maximum_iterations=5000)

    # Solve forward problem
    _ = solver.solve(inputdata,
                    PRINT_INFO=True,
                    COMPUTE_SCATTERED_FIELD=True,
                    SAVE_INTERN_FIELD=True)
    # Number of elements (pixels)

    GS = richmond.richmond_data(config, resolution)
    GD = richmond.richmond_state(config, resolution)

    result = rst.Result(
        name='evaluated_result',
        method_name=algorithim.__name__,
        configuration=config
    )

    incident_field = inputdata.ei
    scattered_field = inputdata.scattered_field
    ground_truth_epsilon = inputdata.rel_permittivity
    
    recon_scattered, chi = algorithim(scattered_field, incident_field, GS, GD)

    epsilon_r_recon = config.epsilon_rb * (np.real(chi) + 1)
    
    if epsilon_r_recon.ndim == 1:
        epsilon_r_recon = epsilon_r_recon.reshape(resolution)

    epad = rst.compute_zeta_epad(ground_truth_epsilon, epsilon_r_recon)
    
    
    print(f"Avarage Contrast shape: {np.mean(np.real(chi)):.2f}")

    result = rst.Result(
        name='evaluated_result',
        method_name=algorithim.__name__,
        configuration=config,
        rel_permittivity=epsilon_r_recon,
    )

    objective_function = norm(inputdata.scattered_field - recon_scattered)**2

    result.update_error(inputdata=inputdata,
                        scattered_field=recon_scattered,
                        rel_permittivity=epsilon_r_recon,
                        contrast=chi,
                        objective_function=objective_function)
    


    print(f"Permittivity error: {epad}%")

    return result