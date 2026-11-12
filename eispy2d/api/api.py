import numpy as np

from eispy2d.discretization import richmond
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.core import error
from eispy2d.core import inputdata as ipt
from eispy2d.utils import draw
from scipy.linalg import norm



def _build_params(params):
    p = params or {}
    return dict(
        wavelength            = p.get("wavelength", 1.0),
        image_size            = tuple(p.get("image_size", (2.0, 2.0))),
        number_measurements   = p.get("number_measurements", 10),
        number_sources        = p.get("number_sources", 10),
        observation_radius    = p.get("observation_radius", 6.0),
        background_permittivity = p.get("background_permittivity", 1.0),
        resolution            = tuple(p.get("resolution", (60, 60))),
        noise_level           = p.get("noise_level", 1.0),
        disp                  = p.get("disp", False),
        shape                 = p.get("shape", "triangle").lower(),
        center                = p.get("center", [0, 0]),
        rotate                = p.get("rotate", 0.0),
        contrast              = p.get("contrast", 0.25),
        object_size           = p.get("object_size", 1.0),
    )


def _draw_ground_truth(p, config, resolution):
    Lx, Ly = p["image_size"]
    rb = p["background_permittivity"]
    contrast = p["contrast"]
    center = p["center"]
    rotate = p["rotate"]
    object_size = p["object_size"]

    epsilon_rb = rb

    shape = p["shape"]
    # Draw figure
    if shape == "triangle":
        return draw.triangle(
            side_length=np.sqrt(object_size),  
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "square":
        return draw.square(
            side_length=np.sqrt(object_size),  
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "circle":
        return draw.circle(
            radius=np.sqrt(object_size / np.pi),
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast        
        )
    elif shape == "cross":
        cross_size = np.sqrt(object_size)
        return draw.cross(
            height=cross_size,
            width=cross_size,
            thickness=cross_size * 0.2,  
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "ellipse":
        radius = np.sqrt(object_size / np.pi)
        return draw.ellipse(
            x_radius=radius * 1.2, 
            y_radius=radius * 0.8,  
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "parallelogram":
        size = np.sqrt(object_size)
        return draw.parallelogram(
            length=size,
            height=size * 0.7,
            inclination=30.0,
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "polygon":
        radius = np.sqrt(object_size / (5 * np.sin(2*np.pi/5) / 2)) 
        return draw.polygon(
            number_sides=5,
            radius=radius,
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "random":
        
        radius = np.sqrt(object_size / np.pi)  
        return draw.random(
            number_sides=8, 
            maximum_radius=radius,
            minimum_radius=radius * 0.5, 
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast
        )
    elif shape == "random_gaussians":
        return draw.random_gaussians(
            number_distributions=5,
            maximum_spread=0.5,
            minimum_spread=0.2,
            distance_from_border=0.1,
            resolution=resolution,
            surface_area=(1.0, 1.0),
            rel_permittivity_amplitude=(contrast) + epsilon_rb,  # variation amplitude
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            background_rel_permittivity=epsilon_rb,
            background_conductivity=0.0,
            center=center,
            rotate=rotate,
            edge_smoothing=0.03
        )
    elif shape == "star4":
        radius = np.sqrt(object_size / np.pi) * 1.2 
        return draw.star4(
            radius=radius,
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "star5":
        radius = np.sqrt(object_size / np.pi) * 1.2
        return draw.star5(
            radius=radius,
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "star6":
        radius = np.sqrt(object_size / np.pi) * 1.2
        return draw.star6(
            radius=radius,
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "rhombus":
        size = np.sqrt(object_size)
        return draw.rhombus(
            x_radius=size * 0.6,
            y_radius=size * 0.6,
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "trapezoid":
        size = np.sqrt(object_size)
        return draw.trapezoid(
            upper_length=size * 0.6,
            lower_length=size,
            height=size * 0.7,
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast,
            rotate=rotate
        )
    elif shape == "ring":
        outer_radius = np.sqrt(object_size / np.pi)
        return draw.ring(
            inner_radius=outer_radius * 0.4, 
            outer_radius=outer_radius,
            center=center,
            axis_length_x=config.Lx,
            axis_length_y=config.Ly,
            resolution=resolution,
            background_rel_permittivity=epsilon_rb,
            object_rel_permittivity=epsilon_rb + contrast
            
        )
    else:
        raise error.WrongValueInput(
            "evaluate", "shape",
            "triangle, square, ellipse, circle, cross, trapezoid, polygon, "
            "parallelogram, rhombus, star4, star5, star6, ring, random, "
            "random_gaussians", shape)


def prepare_problem(params=None, forward_kwargs=None):
    p = _build_params(params)
    Lx, Ly = p["image_size"]

    config = cfg.Configuration(
        name='cfg_test',
        wavelength_unit=True,
        wavelength=p["wavelength"],
        number_measurements=p["number_measurements"],
        number_sources=p["number_sources"],
        image_size=[Ly, Lx],
        observation_radius=p["observation_radius"],
        background_permittivity=p["background_permittivity"],
        magnitude=1.0,
        perfect_dielectric=True,
    )
    
    if p["disp"]:
        config.draw(show=True)

    inputdata = ipt.InputData(
        name='iptTest',
        configuration=config,
        resolution=p["resolution"],
        noise=p["noise_level"],
        indicators=[rst.REL_PERMITTIVITY_PAD_ERROR,
                    rst.RESIDUAL_NORM_ERROR,
                    rst.EXECUTION_TIME],
    )
    inputdata.rel_permittivity, _ = _draw_ground_truth(p, config, p["resolution"])

    solver = mom.MoM_CG_FFT(**(forward_kwargs or
                                {"tolerance": .001, "maximum_iterations": 5000}))
    solver.solve(inputdata, PRINT_INFO=p["disp"],
                 COMPUTE_SCATTERED_FIELD=True,
                 SAVE_INTERN_FIELD=True)

    discretization = richmond.Richmond(config, p["resolution"], state=False)
    return config, inputdata, solver, discretization


def run_algorithm(algorithm, inputdata, solver, config,
                  discretization, recover_resolution, disp=False):
    GS = richmond.richmond_data(config, recover_resolution)
    GD = richmond.richmond_state(config, recover_resolution)

    incident_field  = solver.incident_field(recover_resolution, config)
    scattered_field = inputdata.scattered_field

    recon_scattered, chi = algorithm(
        scattered_field, incident_field, GS, GD, recover_resolution
    )

    epsilon_r_recon = config.epsilon_rb * (np.real(chi) + 1)
    if epsilon_r_recon.ndim == 1:
        epsilon_r_recon = epsilon_r_recon.reshape(recover_resolution)
    epsilon_r_recon = discretization.contrast_image(
        epsilon_r_recon, inputdata.resolution
    )

    result = rst.Result(
        name='evaluated_result',
        method_name=getattr(algorithm, '__name__', type(algorithm).__name__),
        configuration=config,
        rel_permittivity=epsilon_r_recon,
    )
    result.update_error(
        inputdata=inputdata,
        scattered_field=recon_scattered,
        rel_permittivity=epsilon_r_recon,
        contrast=chi,
    )

    if disp:
        result.plot_map(show=True, fontsize=15, groundtruth=inputdata)
        print(result)

    return result

def evaluate(algorithm, params=None):
    config, inputdata, solver, discretization = prepare_problem(params)
    p = _build_params(params)
    return run_algorithm(algorithm, inputdata, solver, config,
                         discretization, p["resolution"], p["disp"])
class EvaluateS:

    def __init__(self, algorithm, forward_kwargs=None, alias=None, disp=False):
        self.algorithm = algorithm
        self.name = getattr(algorithm, "__name__", type(algorithm).__name__)
        self.alias = alias or self.name
        self.disp = disp
        self.forward_kwargs = forward_kwargs or {
            "tolerance": .001, "maximum_iterations": 5000,
        }

    def copy(self):
        return EvaluateS(
            algorithm=self.algorithm,
            forward_kwargs=dict(self.forward_kwargs),
            alias=self.alias,
            disp=self.disp,
        )

    def solve(self, inputdata, discretization=None, print_info=False, **kwargs):
        disp = kwargs.get("disp", self.disp)
        config = inputdata.configuration
        recover_resolution = tuple(inputdata.resolution)

        if disp:
            config.draw(show=True)

        if discretization is None:
            discretization = richmond.Richmond(
                config, recover_resolution, state=False
            )

        solver = kwargs.get("solver")
        if solver is None:
            solver = mom.MoM_CG_FFT(**self.forward_kwargs)
            if inputdata.scattered_field is None:
                solver.solve(inputdata, PRINT_INFO=(print_info or disp),
                             COMPUTE_SCATTERED_FIELD=True,
                             SAVE_INTERN_FIELD=True)

        return run_algorithm(self.algorithm, inputdata, solver, config,
                             discretization, recover_resolution, disp=disp)
