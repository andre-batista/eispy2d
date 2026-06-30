import numpy as np
from skimage import data

from eispy2d.discretization import richmond
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.core import inputdata as ipt
from eispy2d.utils import draw
from scipy.linalg import norm

def evaluate(algorithm, data=None):
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

    if data is not None and "wavelength" in data:
        wavelength = data["wavelength"]
    else:
        wavelength = 1. # [m]
    
    if data is not None and "image_size" in data:
        image_size = data["image_size"]
        Lx, Ly = image_size
    else:
        Lx, Ly = .8, .8 # D domain size [m]

    if data is not None and "number_measurements" in data:
        NM = data["number_measurements"]
    else:
        NM = 10 # number of measurements

    if data is not None and "number_sources" in data:
        NS = data["number_sources"]
    else:
        NS = 10 # number of sources

    if data is not None and "observation_radius" in data:
        RO = data["observation_radius"]
    else:
        RO = 1. # observation radius [m]

    if data is not None and "background_permittivity" in data:
        epsilon_rb = data["background_permittivity"]
    else:
        epsilon_rb = 1. # background relative permittivity

    if data is not None and "resolution" in data:
        resolution = data["resolution"]
    else:
        resolution = (60, 60) # ground-truth image resolution [pixels]
    
    if data is not None and "noise_level" in data:
        noise_level = data["noise_level"]
    else:
        noise_level = 1. # [%/sample]


    E0 = 1.0 # incident wave magnitude [V/m]
    indicators = [rst.REL_PERMITTIVITY_PAD_ERROR, rst.RESIDUAL_NORM_ERROR]
    contrast_level = 1.
    object_size = .2 # [m]

    # Define domain and source parameters
    config = cfg.Configuration(name='cfg_test',
                               wavelength_unit=True,
                               number_measurements=number_measurements,
                               number_sources=number_sources,
                               image_size=[Ly, Lx],
                               observation_radius=observation_radius,
                               background_permittivity=background_permittivity,
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
        object_size,
        center=[0, 0],
        axis_length_x=config.Lx,
        axis_length_y=config.Ly,
        resolution=resolution,
        background_rel_permittivity=background_permittivity,
        object_rel_permittivity=(contrast_level+1)*background_permittivity
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
        method_name=algorithm.__name__,
        configuration=config
    )

    incident_field = inputdata.ei
    scattered_field = inputdata.scattered_field
    ground_truth_epsilon = inputdata.rel_permittivity
    
    recon_scattered, chi = algorithm(scattered_field, incident_field, GS, GD)

    epsilon_r_recon = config.epsilon_rb * (np.real(chi) + 1)
    
    if epsilon_r_recon.ndim == 1:
        epsilon_r_recon = epsilon_r_recon.reshape(resolution)

    epad = rst.compute_zeta_epad(ground_truth_epsilon, epsilon_r_recon)
    
    
    print(f"Avarage Contrast shape: {np.mean(np.real(chi)):.2f}")

    result = rst.Result(
        name='evaluated_result',
        method_name=algorithm.__name__,
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