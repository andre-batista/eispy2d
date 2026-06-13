import numpy as np

from eispy2d.discretization import richmond
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.core import inputdata as ipt

import numpy as np

import numpy as np


def evaluate(algorithim, data=None):
    
    if data == None:
        wavelength = 1.0
        number_measurements = 16
        number_sources = 16
        image_size = [2.0, 2.0]
        observation_radius = 3.0
        background_permittivity = 1.0
        resolution = (60, 60)
        epslon_r = np.ones(resolution) * 2.0
    else:
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

    config = cfg.Configuration(
        name='test',
        wavelength=wavelength,
        number_measurements=number_measurements,
        number_sources=number_sources,
        image_size=image_size,
        observation_radius=observation_radius,
        background_permittivity=background_permittivity)

    inputdata = ipt.InputData(name='ipt_test',
                          configuration=config,
                          resolution=resolution,
                          noise=1.,
                          indicators=[rst.RESIDUAL_PAD_ERROR,
                                      rst.RESIDUAL_NORM_ERROR,
                                      rst.REL_PERMITTIVITY_PAD_ERROR,
                                      rst.REL_PERMITTIVITY_OBJECT_ERROR,
                                      rst.REL_PERMITTIVITY_BACKGROUND_ERROR,
                                      rst.POSITION_ERROR,
                                      rst.SHAPE_ERROR,
                                      rst.EXECUTION_TIME],
                          rel_permittivity=epslon_r)


    GS = richmond.richmond_data(config, resolution)
    GD = richmond.richmond_state(config, resolution)

    result = rst.Result(
        name='evaluated_result',
        method_name=algorithim.__name__,
        configuration=config
    )

    discretization = richmond.Richmond(config, resolution)
    
    solver = mom.MoM_CG_FFT(tolerance=.001, maximum_iterations=5000)

    solver.solve(inputdata,
                 PRINT_INFO=True,
                 COMPUTE_SCATTERED_FIELD=True,
                 SAVE_INTERN_FIELD=True)

    incident_field = inputdata.ei
    scattered_field = inputdata.scattered_field
    
    scattered, chi = algorithim(scattered_field, incident_field, GS, GD)

    epsilon_r_recon = config.epsilon_rb * (np.real(chi) + 1)

    epad = rst.compute_zeta_epad(epslon_r, epsilon_r_recon)
    result.zeta_epad = [epad]

    print(f"Contrast shape: {chi}")
    print(f"Permissivity error: {result.zeta_epad[-1]:.2f}%")

    result.chi = chi
    result.epsilon_r_recon = epsilon_r_recon

    return result
