import numpy as np

from eispy2d.discretization import richmond
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.core import inputdata as ipt
from eispy2d.utils import draw
from scipy.linalg import norm

def evaluate(algorithm, params=None):

    if params is not None and "wavelength" in params:
        wavelength = params["wavelength"]
    else:
        wavelength = 1. # [m]
    
    if params is not None and "image_size" in params:
        image_size = params["image_size"]
        Lx, Ly = image_size
    else:
        Lx, Ly = .8, .8 # D domain size [m]

    if params is not None and "number_measurements" in params:
        NM = params["number_measurements"]
    else:
        NM = 10 # number of measurements

    if params is not None and "number_sources" in params:
        NS = params["number_sources"]
    else:
        NS = 10 # number of sources

    if params is not None and "observation_radius" in params:
        RO = params["observation_radius"]
    else:
        RO = 1. # observation radius [m]

    if params is not None and "background_permittivity" in params:
        epsilon_rb = params["background_permittivity"]
    else:
        epsilon_rb = 1. # background relative permittivity

    if params is not None and "resolution" in params:
        resolution = params["resolution"]
    else:
        resolution = (60, 60) # ground-truth image resolution [pixels]
    
    if params is not None and "noise_level" in params:
        noise_level = params["noise_level"]
    else:
        noise_level = 1. # [%/sample]

    if params is not None and "disp" in params:
        disp = params["disp"]
    else:
        disp = False

    if params is not None and "shape" in params:
        shape = params["shape"]
    else:
        shape = "triangle"

    E0 = 1.0 # incident wave magnitude [V/m]
    indicators = [rst.REL_PERMITTIVITY_PAD_ERROR, rst.RESIDUAL_NORM_ERROR]
    contrast_level = 1.
    object_size = .2 # [m]

    # Define domain and source parameters
    config = cfg.Configuration(name='cfg_test',
                               wavelength_unit=True,
                               wavelength=wavelength,
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
    if shape == "triangle":
        inputdata.rel_permittivity, _ = draw.triangle(
            object_size,  
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb
        )
    elif shape == "square":
        inputdata.rel_permittivity, _ = draw.square(
            side_length=np.sqrt(object_size),  
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb
        )
    elif shape == "circle":
        inputdata.rel_permittivity, _ = draw.circle(
            radius=np.sqrt(object_size / np.pi),
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb
        )
    elif shape == "cross":
        cross_size = np.sqrt(object_size)
        inputdata.rel_permittivity, _ = draw.cross(
            height=cross_size,
            width=cross_size,
            thickness=cross_size * 0.2,  
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb
        )
    elif shape == "ellipse":
        radius = np.sqrt(object_size / np.pi)
        inputdata.rel_permittivity, _ = draw.ellipse(
            x_radius=radius * 1.2, 
            y_radius=radius * 0.8,  
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb
        )
    elif shape == "parallelogram":
        size = np.sqrt(object_size)
        inputdata.rel_permittivity, _ = draw.parallelogram(
            length=size,
            height=size * 0.7,
            inclination=30.0,
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb
        )
    elif shape == "polygon":
        radius = np.sqrt(object_size / (5 * np.sin(2*np.pi/5) / 2)) 
        inputdata.rel_permittivity, _ = draw.polygon(
            number_sides=5,
            radius=radius,
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb
        )
    elif shape == "random":
        
        radius = np.sqrt(object_size / np.pi)  
        inputdata.rel_permittivity, _ = draw.random(
            number_sides=8, 
            maximum_radius=radius,
            minimum_radius=radius * 0.5, 
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb
        )
    elif shape == "random_gaussians":
        inputdata.rel_permittivity, _ = draw.random_gaussians(
            number_distributions=5,
            maximum_spread=0.5,
            minimum_spread=0.2,
            distance_from_border=0.1,
            resolution=resolution,
            surface_area=(1.0, 1.0),
            rel_permittivity_amplitude=(contrast_level)*epsilon_rb,  # variation amplitude
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            background_rel_permittivity=epsilon_rb,
            background_conductivity=0.0,
            center=[0, 0],
            rotate=0.0,
            edge_smoothing=0.03
        )
    elif shape == "star4":
        radius = np.sqrt(object_size / np.pi) * 1.2 
        inputdata.rel_permittivity, _ = draw.star4(
            radius=radius,
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb,
            rotate=0.0
        )
    elif shape == "star5":
        radius = np.sqrt(object_size / np.pi) * 1.2
        inputdata.rel_permittivity, _ = draw.star5(
            radius=radius,
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb,
            rotate=0.0
        )
    elif shape == "star6":
        radius = np.sqrt(object_size / np.pi) * 1.2
        inputdata.rel_permittivity, _ = draw.star6(
            radius=radius,
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb,
            rotate=0.0
        )
    elif shape == "rhombus":
        size = np.sqrt(object_size)
        inputdata.rel_permittivity, _ = draw.rhombus(
            x_radius=size * 0.6,
            y_radius=size * 0.6,
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb,
            rotate=45.0 
        )
    elif shape == "trapezoid":
        size = np.sqrt(object_size)
        inputdata.rel_permittivity, _ = draw.trapezoid(
            upper_length=size * 0.6,
            lower_length=size,
            height=size * 0.7,
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb,
            rotate=0.0
        )
    elif shape == "ring":
        outer_radius = np.sqrt(object_size / np.pi)
        inputdata.rel_permittivity, _ = draw.ring(
            inner_radius=outer_radius * 0.4, 
            outer_radius=outer_radius,
            center=[0, 0],
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=(contrast_level+1)*epsilon_rb
            
        )
    else:
        raise ValueError(f"Shape '{shape}' is not supported.")
        
        

    # Build forward solver object
    solver = mom.MoM_CG_FFT(tolerance=.001,
                            maximum_iterations=5000)

    # Solve forward problem
    _ = solver.solve(inputdata,
                    PRINT_INFO=disp,
                    COMPUTE_SCATTERED_FIELD=True,
                    SAVE_INTERN_FIELD=True)
    
    # Reduce resolution for reconstruction (inverse crime)
    recover_resolution = (int(resolution[0]/1.5), int(resolution[1]/1.5))
    discretization = richmond.Richmond(config, recover_resolution, state=False)
    GS = richmond.richmond_data(config, recover_resolution)
    GD = richmond.richmond_state(config, recover_resolution)

    result = rst.Result(
        name='evaluated_result',
        method_name=algorithm.__name__,
        configuration=config
    )

    incident_field = solver.incident_field(recover_resolution, config)
    scattered_field = inputdata.scattered_field
    ground_truth_epsilon = inputdata.rel_permittivity
    
    recon_scattered, chi = algorithm(scattered_field, incident_field, GS, GD,
                                     recover_resolution)

    epsilon_r_recon = config.epsilon_rb * (np.real(chi) + 1)
    
    if epsilon_r_recon.ndim == 1:
        epsilon_r_recon = epsilon_r_recon.reshape(resolution)
        epsilon_r_recon=discretization.contrast_image(epsilon_r_recon,
        inputdata.resolution)
    else:
        epsilon_r_recon=discretization.contrast_image(epsilon_r_recon,
        inputdata.resolution)
  
    result = rst.Result(
        name='evaluated_result',
        method_name=algorithm.__name__,
        configuration=config,
        # rel_permittivity=epsilon_r_recon,
    )

    result.update_error(inputdata=inputdata,
                        scattered_field=recon_scattered,
                        rel_permittivity=epsilon_r_recon,
                        contrast=chi)
    
    if disp:
        inputdata.draw(show=True)
        print(result)

    return result