r"""Geometric Shape Drawing Functions for Electromagnetic Inverse Scattering.

This module provides a comprehensive collection of functions for drawing various
geometric shapes with specified electromagnetic properties. These functions are
designed to create synthetic test objects for electromagnetic inverse scattering
simulations and algorithm validation.

The module supports drawing both simple and complex shapes with customizable
electromagnetic properties (relative permittivity and conductivity), positioning,
rotation, and scaling. All shapes can be drawn on existing images or create new
ones with specified resolution and background properties.

Key Features:
- Support for both dielectric and conductive objects
- Flexible positioning and rotation capabilities
- Multiple drawing modes (new image or overlay on existing)
- Consistent interface across all shape functions
- Proper coordinate system handling with origin at image center
- Rotation around object center with degree-based angles

Functions Overview:
- Basic shapes: square, circle, ellipse, triangle, line, cross
- Polygons: polygon, rhombus, trapezoid, parallelogram
- Stars: star4, star5, star6 (4, 5, and 6-pointed stars)
- Composite shapes: ring (annulus)
- Random shapes: random (irregular polygon)
- Patterns: wave, random_waves

Each function returns a tuple of (relative_permittivity, conductivity) arrays
representing the electromagnetic properties of the created scene.

Coordinate System:
- Origin (0, 0) is at the center of the image
- X-axis points right, Y-axis points up
- Angles are measured in degrees, counterclockwise from +X axis
- Image coordinates use standard matrix indexing (row, column)

Examples
--------
>>> import draw
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> # Create a simple dielectric square
>>> epsilon_r, sigma = draw.square(
...     side_length=0.5,
...     resolution=(64, 64),
...     object_rel_permittivity=2.0,
...     background_rel_permittivity=1.0
... )

>>> # Create a conducting circle
>>> epsilon_r, sigma = draw.circle(
...     radius=0.3,
...     resolution=(128, 128),
...     object_conductivity=1.0,
...     center=[0.1, -0.1],
...     rotate=45.0
... )

>>> # Overlay multiple objects
>>> epsilon_r, sigma = draw.square(
...     side_length=0.8,
...     resolution=(100, 100),
...     object_rel_permittivity=1.5
... )
>>> epsilon_r, sigma = draw.circle(
...     radius=0.2,
...     object_rel_permittivity=3.0,
...     rel_permittivity=epsilon_r,
...     conductivity=sigma
... )

>>> # Visualize results
>>> plt.figure(figsize=(10, 4))
>>> plt.subplot(1, 2, 1)
>>> plt.imshow(epsilon_r, cmap='jet', extent=[-1, 1, -1, 1])
>>> plt.title('Relative Permittivity')
>>> plt.colorbar()
>>> plt.subplot(1, 2, 2)
>>> plt.imshow(sigma, cmap='viridis', extent=[-1, 1, -1, 1])
>>> plt.title('Conductivity')
>>> plt.colorbar()
>>> plt.show()

Notes
-----
All functions follow a consistent parameter naming convention:
- Geometric parameters come first (size, dimensions)
- Image parameters follow (axis_length_x, axis_length_y, resolution)
- Background electromagnetic properties
- Object electromagnetic properties
- Positioning and rotation parameters
- Optional existing image arrays for overlay operations

The functions handle input validation and provide appropriate error messages
for missing required parameters. Either resolution or existing image arrays
must be provided.

For wave-based patterns, the functions support both periodic and random
wave generation with customizable amplitude and frequency characteristics.
"""

import numpy as np
from numpy import pi
from numpy import logical_and
from numpy import random as rnd

from eispy2d.core import error


def square(side_length, axis_length_x=2., axis_length_y=2., resolution=None,
           background_rel_permittivity=1., background_conductivity=0.,
           object_rel_permittivity=1., object_conductivity=0.,
           center=[0., 0.], rotate=0., rel_permittivity=None,
           conductivity=None):
    """Draw a square with specified electromagnetic properties.

    This function creates a square object with customizable electromagnetic
    properties, positioning, and rotation on a 2D grid. The square can be
    drawn on a new image or overlaid on existing permittivity and conductivity
    arrays.

    Parameters
    ----------
    side_length : float
        Length of the square's side in the same units as axis_length_x/y.
    axis_length_x : float, default=2.0
        Physical length of the image in the x-direction.
    axis_length_y : float, default=2.0
        Physical length of the image in the y-direction.
    resolution : tuple of int, optional
        Image resolution as (NY, NX). Required if rel_permittivity and
        conductivity are not provided.
    background_rel_permittivity : float, default=1.0
        Relative permittivity of the background medium.
    background_conductivity : float, default=0.0
        Conductivity of the background medium in S/m.
    object_rel_permittivity : float, default=1.0
        Relative permittivity of the square object.
    object_conductivity : float, default=0.0
        Conductivity of the square object in S/m.
    center : list of float, default=[0.0, 0.0]
        Center position of the square as [x, y] coordinates.
    rotate : float, default=0.0
        Rotation angle in degrees (counterclockwise).
    rel_permittivity : numpy.ndarray, optional
        Existing relative permittivity array to overlay on.
    conductivity : numpy.ndarray, optional
        Existing conductivity array to overlay on.

    Returns
    -------
    tuple of numpy.ndarray
        (epsilon_r, sigma) arrays with the square drawn.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_square', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    L = side_length
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    epsilon_r[logical_and(logical_and(yp >= -L/2, yp <= L/2),
                          logical_and(xp >= -L/2, xp <= L/2))] = epsilon_ro
    sigma[logical_and(logical_and(yp >= -L/2, yp <= L/2),
                      logical_and(xp >= -L/2, xp <= L/2))] = sigma_o

    return epsilon_r, sigma


def triangle(side_length, axis_length_x=2., axis_length_y=2., resolution=None,
             background_rel_permittivity=1., background_conductivity=0.,
             object_rel_permittivity=1., object_conductivity=0.,
             rel_permittivity=None, conductivity=None, rotate=0.,
             center=[0., 0.]):
    """Draw an equilateral triangle with specified electromagnetic properties.

    Parameters
    ----------
    side_length : float
        Length of the triangle's side.
    axis_length_x : float, default=2.0
        Physical length of the image in the x-direction.
    axis_length_y : float, default=2.0
        Physical length of the image in the y-direction.
    resolution : tuple of int, optional
        Image resolution as (NY, NX). Required if rel_permittivity and
        conductivity are not provided.
    background_rel_permittivity : float, default=1.0
        Background relative permittivity.
    background_conductivity : float, default=0.0
        Background conductivity in S/m.
    object_rel_permittivity : float, default=1.0
        Triangle's relative permittivity.
    object_conductivity : float, default=0.0
        Triangle's conductivity in S/m.
    rel_permittivity : numpy.ndarray, optional
        Existing permittivity array to overlay on.
    conductivity : numpy.ndarray, optional
        Existing conductivity array to overlay on.
    rotate : float, default=0.0
        Rotation angle in degrees (counterclockwise).
    center : list of float, default=[0.0, 0.0]
        Center position as [x, y] coordinates.

    Returns
    -------
    tuple of numpy.ndarray
        (epsilon_r, sigma) arrays with the triangle drawn.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_triangle', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    l = side_length
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    r = l/np.sqrt(3)
    h = l*np.sqrt(3)/2-l/np.sqrt(3)
    a1, b1 = (-h-r)/(l/2), r
    a2, b2 = (r+h)/(l/2), r

    # Set object
    triangle = logical_and(yp >= -h, logical_and(yp <= a2*xp + b2,
                                                 yp <= a1*xp + b1))
    epsilon_r[triangle] = epsilon_ro
    sigma[triangle] = sigma_o

    return epsilon_r, sigma


def star4(radius, axis_length_x=2., axis_length_y=2., resolution=None,
          background_rel_permittivity=1., background_conductivity=0.,
          object_rel_permittivity=1., object_conductivity=0.,
          rel_permittivity=None, conductivity=None,
          center=[0., 0.], rotate=0.,):
    """Draw a 4-point star.

    Parameters
    ----------
        radius : float
            Radius of the vertex from the center of the star.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_4star', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    a, b = radius, .5*radius
    rhombus1 = logical_and(-a*xp - b*yp >= -a*b,
                           logical_and(a*xp - b*yp >= -a*b,
                                       logical_and(a*xp+b*yp >= -a*b,
                                                   -a*xp+b*yp >= -a*b)))
    rhombus2 = logical_and(-b*xp - a*yp >= -a*b,
                           logical_and(b*xp - a*yp >= -a*b,
                                       logical_and(b*xp+a*yp >= -a*b,
                                                   -b*xp+a*yp >= -a*b)))
    epsilon_r[np.logical_or(rhombus1, rhombus2)] = epsilon_ro
    sigma[np.logical_or(rhombus1, rhombus2)] = sigma_o

    return epsilon_r, sigma


def star5(radius, axis_length_x=2., axis_length_y=2., resolution=None,
          background_rel_permittivity=1., background_conductivity=0.,
          object_rel_permittivity=1., object_conductivity=0.,
          rel_permittivity=None, conductivity=None,
          center=[0., 0.], rotate=0.,):
    """Draw a 5-point star.

    Parameters
    ----------
        radius : int
            Length from the center of the star to the main vertices.

        maximum_radius : float
            Maximum radius from the origin to each vertex.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_random', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Create vertices
    delta = 2*pi/5
    phi = np.array([0, 2*delta, 4*delta, delta, 3*delta, 0]) + pi/2 - 2*pi/5
    xv = radius*np.cos(phi)
    yv = radius*np.sin(phi)

    # Set object
    for i in range(NX):
        for j in range(NY):
            if winding_number(xp[j, i], yp[j, i], xv, yv):
                epsilon_r[j, i] = epsilon_ro
                sigma[j, i] = sigma_o

    return epsilon_r, sigma


def star6(radius, axis_length_x=2., axis_length_y=2., resolution=None,
          background_rel_permittivity=1., background_conductivity=0.,
          object_rel_permittivity=1., object_conductivity=0.,
          rel_permittivity=None, conductivity=None, rotate=0.,
          center=[0., 0.]):
    """Draw a six-pointed star (hexagram).

    Parameters
    ----------
        radius : float
            Length from the center to the maximum edge.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_star', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    r = radius
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    l = r*np.sqrt(3)
    h = l*np.sqrt(3)/2-l/np.sqrt(3)
    a1, b1 = (-h-r)/(l/2), r
    a2, b2 = (r+h)/(l/2), r

    # Set object
    triangle = logical_and(yp >= -h, logical_and(yp <= a2*xp + b2,
                                                 yp <= a1*xp + b1))

    # Set object
    epsilon_r[triangle] = epsilon_ro
    sigma[triangle] = sigma_o

    a1, b1 = (h+r)/(l/2), -r
    a2, b2 = (-r-h)/(l/2), -r
    triangle = logical_and(yp <= h, logical_and(yp >= a2*xp + b2,
                                                yp >= a1*xp + b1))

    epsilon_r[triangle] = epsilon_ro
    sigma[triangle] = sigma_o

    return epsilon_r, sigma


def ring(inner_radius, outer_radius, axis_length_x=2., axis_length_y=2.,
         resolution=None, center=[0., 0.], background_rel_permittivity=1.,
         background_conductivity=0., object_rel_permittivity=1.,
         object_conductivity=0., rel_permittivity=None,
         conductivity=None):
    """Draw a ring.

    Parameters
    ----------
        inner_radius, outer_radius : float
            Inner and outer radii of the ring.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_ring', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    ra, rb = inner_radius, outer_radius
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])

    # Set object
    epsilon_r[logical_and(x**2 + y**2 <= rb**2,
                          x**2 + y**2 >= ra**2)] = epsilon_ro
    sigma[logical_and(x**2 + y**2 <= rb**2,
                      x**2 + y**2 >= ra**2)] = sigma_o

    return epsilon_r, sigma


def circle(radius, axis_length_x=2., axis_length_y=2., resolution=None,
           background_rel_permittivity=1., background_conductivity=0.,
           object_rel_permittivity=1., object_conductivity=0.,
           rel_permittivity=None, conductivity=None,
           center=[0., 0.]):
    """Draw a circle.

    Parameters
    ----------
        radius : float
            Radius of the circle.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_circle', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    r = radius
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])

    # Set object
    epsilon_r[x**2 + y**2 <= r**2] = epsilon_ro
    sigma[x**2 + y**2 <= r**2] = sigma_o

    return epsilon_r, sigma


def ellipse(x_radius, y_radius, axis_length_x=2., axis_length_y=2.,
            resolution=None, background_rel_permittivity=1.,
            background_conductivity=0., object_rel_permittivity=1.,
            object_conductivity=0., rel_permittivity=None,
            conductivity=None, center=[0., 0.], rotate=0.):
    """Draw an ellipse.

    Parameters
    ----------
        x_radius, y_radius : float
            Ellipse radii in each axis.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_ellipse', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    a, b = x_radius, y_radius
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    epsilon_r[xp**2/a**2 + yp**2/b**2 <= 1.] = epsilon_ro
    sigma[xp**2/a**2 + yp**2/b**2 <= 1.] = sigma_o

    return epsilon_r, sigma


def cross(height, width, thickness, axis_length_x=2., axis_length_y=2.,
          resolution=None, background_rel_permittivity=1.,
          background_conductivity=0., object_rel_permittivity=1.,
          object_conductivity=0., center=[0., 0.], rotate=0.,
          rel_permittivity=None, conductivity=None):
    """Draw a cross.

    Parameters
    ----------
        height, width, thickness : float
            Parameters of the cross.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_cross', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    horizontal_bar = (
        logical_and(xp >= -width/2,
                    logical_and(xp <= width/2,
                                logical_and(yp >= -thickness/2,
                                            yp <= thickness/2)))
    )
    vertical_bar = (
        logical_and(yp >= -height/2,
                    logical_and(yp <= height/2,
                                logical_and(xp >= -thickness/2,
                                            xp <= thickness/2)))
    )
    epsilon_r[np.logical_or(horizontal_bar, vertical_bar)] = epsilon_ro
    sigma[np.logical_or(horizontal_bar, vertical_bar)] = sigma_o

    return epsilon_r, sigma


def line(length, thickness, axis_length_x=2., axis_length_y=2.,
         resolution=None, background_rel_permittivity=1.,
         background_conductivity=0., object_rel_permittivity=1.,
         object_conductivity=0., center=[0., 0.], rotate=0.,
         rel_permittivity=None, conductivity=None):
    """Draw a cross.

    Parameters
    ----------
        length, thickness : float
            Parameters of the line.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_line', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    line = (logical_and(xp >= -length/2,
                        logical_and(xp <= length/2,
                                    logical_and(yp >= -thickness/2,
                                                yp <= thickness/2))))
    epsilon_r[line] = epsilon_ro
    sigma[line] = sigma_o

    return epsilon_r, sigma


def polygon(number_sides, radius, axis_length_x=2., axis_length_y=2.,
            resolution=None, center=[0., 0.], rotate=0.,
            background_rel_permittivity=1., background_conductivity=0.,
            object_rel_permittivity=1., object_conductivity=0., 
            rel_permittivity=None, conductivity=None):
    """Draw a polygon with equal sides.

    Parameters
    ----------
        number_sides : int
            Number of sides.

        radius : float
            Radius from the center to each vertex.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_polygon', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    dphi = 2*pi/number_sides
    phi = np.arange(0, number_sides*dphi, dphi)
    xa = radius*np.cos(phi)
    ya = radius*np.sin(phi)
    polygon = np.ones(x.shape, dtype=bool)
    for i in range(number_sides):
        a = -(ya[i]-ya[i-1])
        b = xa[i]-xa[i-1]
        c = (xa[i]-xa[i-1])*ya[i-1] - (ya[i]-ya[i-1])*xa[i-1]
        polygon = logical_and(polygon, a*xp + b*yp >= c)
    epsilon_r[polygon] = epsilon_ro
    sigma[polygon] = sigma_o

    return epsilon_r, sigma


def random(number_sides, maximum_radius, minimum_radius=None, center=[0., 0.],
           axis_length_x=2., axis_length_y=2., resolution=None, 
           background_rel_permittivity=1., background_conductivity=0.,
           object_rel_permittivity=1., object_conductivity=0.,
           rel_permittivity=None, conductivity=None):
    """Draw a random polygon.

    Parameters
    ----------
        number_sides : int
            Number of sides of the polygon.

        maximum_radius : float
            Maximum radius from the origin to each vertex.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_random', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    if minimum_radius is None:
        minimum_radius = .4*maximum_radius

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])

    # Create vertices
    phi = rnd.normal(loc=np.linspace(0, 2*pi, number_sides, endpoint=False),
                     scale=0.5)
    phi[phi >= 2*pi] = phi[phi >= 2*pi] - np.floor(phi[phi >= 2*pi]
                                                   / (2*pi))*2*pi
    phi[phi < 0] = -((np.floor(phi[phi < 0]/(2*pi)))*2*pi - phi[phi < 0])
    phi = np.sort(phi)
    radius = (minimum_radius + (maximum_radius-minimum_radius)
              * rnd.rand(number_sides))
    xv = radius*np.cos(phi)
    yv = radius*np.sin(phi)

    # Set object
    for i in range(NX):
        for j in range(NY):
            if winding_number(x[j, i], y[j, i], xv, yv):
                epsilon_r[j, i] = epsilon_ro
                sigma[j, i] = sigma_o

    return epsilon_r, sigma


def rhombus(x_radius, y_radius, axis_length_x=2., axis_length_y=2.,
            resolution=None, center=[0., 0.], rotate=0.,
            background_rel_permittivity=1., background_conductivity=0.,
            object_rel_permittivity=1., object_conductivity=0., 
            rel_permittivity=None, conductivity=None):
    """Draw a rhombus.

    Parameters
    ----------
        x_radius, y_radius : float
            Radii in each axis.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_rhombus', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    a, b = y_radius, x_radius
    rhombus = logical_and(a/b*xp + yp <= a,
                          logical_and(-a/b*xp + yp >= -a,
                                      logical_and(a/b*xp+yp >= -a,
                                                  -a/b*xp+yp <= a)))

    epsilon_r[rhombus] = epsilon_ro
    sigma[rhombus] = sigma_o

    return epsilon_r, sigma


def trapezoid(upper_length, lower_length, height, axis_length_x=2.,
              axis_length_y=2., resolution=None, center=[0., 0.], rotate=0.,
              background_rel_permittivity=1., background_conductivity=0.,
              object_rel_permittivity=1., object_conductivity=0.,
              rel_permittivity=None, conductivity=None):
    """Draw a trapezoid.

    Parameters
    ----------
        upper_length, lower_length, height : float
            Dimensions.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_trapezoid',
                                      'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    ll, lu, h = lower_length, upper_length, height
    a1, b1, c1 = -h, (lu-ll)/2, -(lu-ll)*h/4 - h*ll/2
    a2, b2, c2 = h, (lu-ll)/2, (lu-ll)*h/4 - h*lu/2
    trapezoid = logical_and(a1*xp + b1*yp >= c1,
                            logical_and(a2*xp + b2*yp >= c2,
                                        logical_and(yp <= height/2,
                                                    yp >= -height/2)))

    epsilon_r[trapezoid] = epsilon_ro
    sigma[trapezoid] = sigma_o

    return epsilon_r, sigma


def parallelogram(length, height, inclination, center=[0., 0.], rotate=0.,
                  axis_length_x=2., axis_length_y=2., resolution=None,
                  background_rel_permittivity=1.,
                  background_conductivity=0., object_rel_permittivity=1.,
                  object_conductivity=0.,
                  rel_permittivity=None, conductivity=None):
    """Draw a paralellogram.

    Parameters
    ----------
        length, height : float
            Dimensions.

        inclination : float
            In degrees.

        axis_length_x, axis_length_y : float, default: 2.0
            Length of the size of the image.

        resolution : 2-tuple
            Image resolution, in y and x directions, i.e., (NY, NX).
            *Either this argument or rel_permittivity or
            conductivity must be given!*

        background_rel_permittivity : float, default: 1.0

        background_conductivity : float, default: 0.0

        object_rel_permittivity : float, default: 1.0

        object_conductivity : float, default: 0.0

        center : list, default: [0.0, 0.0]
            Center of the object in the image. The center of the image
            corresponds to the origin of the coordinates.

        rel_permittivity : :class:`numpy.ndarray`, default:None
            A predefined image in which the object will be drawn.

        conductivity : :class:`numpy.ndarray`, default: None
            A predefined image in which the object will be drawn.

        rotate : float, default: 0.0 degrees
            Rotation of the object around its center. In degrees.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_parallelogram',
                                      'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity
    epsilon_ro = object_rel_permittivity
    sigma_o = object_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) - center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) - center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Set object
    h, l, tga = height, length, np.tan(np.deg2rad(inclination))
    g = h/tga
    a = (l+g)/2
    b = a-g
    parallelogram = logical_and(yp >= tga*xp - h/2/g*(a+b),
                                logical_and(yp <= tga*xp + h/2/g*(a+b),
                                            logical_and(yp <= h/2,
                                                        yp >= -h/2)))

    epsilon_r[parallelogram] = epsilon_ro
    sigma[parallelogram] = sigma_o

    return epsilon_r, sigma


def wave(number_peaks, rel_permittivity_peak=1., conductivity_peak=0.,
         rel_permittivity_valley=None, conductivity_valley=None,
         resolution=None, number_peaks_y=None, axis_length_x=2.,
         axis_length_y=2., background_rel_permittivity=1.,
         background_conductivity=0., rel_permittivity=None,
         conductivity=None, wave_bounds_proportion=(1., 1.),
         center=[0., 0.], rotate=0.):
    """Draw a wave pattern with specified electromagnetic properties.

    Parameters
    ----------
    number_peaks : int
        Number of peaks for both directions or for x-axis (if number_peaks_y is None).
    rel_permittivity_peak : float, default=1.0
        Peak value of relative permittivity.
    conductivity_peak : float, default=0.0
        Peak value of conductivity in S/m.
    rel_permittivity_valley : float, optional
        Valley value of relative permittivity. If None, uses peak value.
    conductivity_valley : float, optional
        Valley value of conductivity. If None, uses peak value.
    resolution : tuple of int, optional
        Image resolution as (NY, NX).
    number_peaks_y : int, optional
        Number of peaks in y-direction. If None, uses number_peaks.
    axis_length_x : float, default=2.0
        Physical length in x-direction.
    axis_length_y : float, default=2.0
        Physical length in y-direction.
    background_rel_permittivity : float, default=1.0
        Background relative permittivity.
    background_conductivity : float, default=0.0
        Background conductivity in S/m.
    rel_permittivity : numpy.ndarray, optional
        Existing permittivity array to overlay on.
    conductivity : numpy.ndarray, optional
        Existing conductivity array to overlay on.
    wave_bounds_proportion : tuple, default=(1.0, 1.0)
        Proportion of image dimensions for wave area.
    center : list of float, default=[0.0, 0.0]
        Center position as [x, y] coordinates.
    rotate : float, default=0.0
        Rotation angle in degrees.

    Returns
    -------
    tuple of numpy.ndarray
        (epsilon_r, sigma) arrays with the wave pattern drawn.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_wave', 'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) + center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) + center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Wave area
    ly, lx = wave_bounds_proportion[0]*Ly, wave_bounds_proportion[1]*Lx
    wave = logical_and(xp >= -lx/2,
                       logical_and(xp <= lx/2,
                                   logical_and(yp >= -ly/2, yp <= ly/2)))

    # Wave parameters
    number_peaks_x = number_peaks
    if number_peaks_y is None:
        number_peaks_y = number_peaks
    Kx = 2*number_peaks_x-1
    Ky = 2*number_peaks_y-1

    # Set up valley magnitude
    if (rel_permittivity_peak == background_rel_permittivity
            and rel_permittivity_valley is None):
        rel_permittivity_valley = background_rel_permittivity
    elif rel_permittivity_valley is None:
        rel_permittivity_valley = rel_permittivity_peak
    if (conductivity_peak == background_conductivity
            and conductivity_valley is None):
        conductivity_valley = background_conductivity
    elif conductivity_valley is None:
        conductivity_valley = conductivity_peak

    # Relative permittivity
    epsilon_r[wave] = (np.cos(2*pi/(2*lx/Kx)*xp[wave])
                       * np.cos(2*pi/(2*ly/Ky)*yp[wave]))
    epsilon_r[logical_and(wave, epsilon_r >= 0)] = (
        rel_permittivity_peak*epsilon_r[logical_and(wave, epsilon_r >= 0)]
    )
    epsilon_r[logical_and(wave, epsilon_r < 0)] = (
        rel_permittivity_valley*epsilon_r[logical_and(wave, epsilon_r < 0)]
    )
    epsilon_r[wave] = epsilon_r[wave] + epsilon_rb
    epsilon_r[logical_and(wave, epsilon_r < 1.)] = 1.

    # Conductivity
    sigma[wave] = (np.cos(2*pi/(2*lx/Kx)*xp[wave])
                   * np.cos(2*pi/(2*ly/Ky)*yp[wave]))
    sigma[logical_and(wave, epsilon_r >= 0)] = (
        conductivity_peak*sigma[logical_and(wave, sigma >= 0)]
    )
    sigma[logical_and(wave, sigma < 0)] = (
        conductivity_valley*sigma[logical_and(wave, sigma < 0)]
    )
    sigma[wave] = sigma[wave] + sigma_b
    sigma[logical_and(wave, sigma < 0.)] = 0.

    return epsilon_r, sigma


def random_waves(number_waves, maximum_number_peaks,
                 maximum_number_peaks_y=None, resolution=None,
                 rel_permittivity_amplitude=0., conductivity_amplitude=0.,
                 axis_length_x=2., axis_length_y=2.,
                 background_rel_permittivity=1.,
                 background_conductivity=0.,
                 rel_permittivity=None, conductivity=None,
                 wave_bounds_proportion=(1., 1.), center=[0., 0.], rotate=0.,
                 edge_smoothing=0.03):
    """Draw random wave patterns with specified electromagnetic properties.

    Parameters
    ----------
    number_waves : int
        Number of wave components.
    maximum_number_peaks : int
        Maximum number of peaks (controls smallest wavelength).
    maximum_number_peaks_y : int, optional
        Maximum number of peaks in y-direction. If None, uses maximum_number_peaks.
    resolution : tuple of int, optional
        Image resolution as (NY, NX).
    rel_permittivity_amplitude : float, default=0.0
        Maximum amplitude of relative permittivity variation.
    conductivity_amplitude : float, default=0.0
        Maximum amplitude of conductivity variation in S/m.
    axis_length_x : float, default=2.0
        Physical length in x-direction.
    axis_length_y : float, default=2.0
        Physical length in y-direction.
    background_rel_permittivity : float, default=1.0
        Background relative permittivity.
    background_conductivity : float, default=0.0
        Background conductivity in S/m.
    rel_permittivity : numpy.ndarray, optional
        Existing permittivity array to overlay on.
    conductivity : numpy.ndarray, optional
        Existing conductivity array to overlay on.
    wave_bounds_proportion : tuple, default=(1.0, 1.0)
        Proportion of image dimensions for wave area.
    center : list of float, default=[0.0, 0.0]
        Center position as [x, y] coordinates.
    rotate : float, default=0.0
        Rotation angle in degrees.
    edge_smoothing : float, default=0.03
        Percentage of cells at boundary to smooth.

    Returns
    -------
    tuple of numpy.ndarray
        (epsilon_r, sigma) arrays with the random wave pattern drawn.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_random_waves',
                                      'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) + center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) + center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Wave area
    ly, lx = wave_bounds_proportion[0]*Ly, wave_bounds_proportion[1]*Lx
    wave = logical_and(xp >= -lx/2,
                       logical_and(xp <= lx/2,
                                   logical_and(yp >= -ly/2, yp <= ly/2)))

    # Wave parameters
    max_number_peaks_x = maximum_number_peaks
    if maximum_number_peaks_y is None:
        max_number_peaks_y = maximum_number_peaks
    m = np.round((max_number_peaks_x-1)*rnd.rand(number_waves)) + 1
    n = np.round((max_number_peaks_y-1)*rnd.rand(number_waves)) + 1
    lam_x = lx/m
    lam_y = ly/n
    phi = 2*pi*rnd.rand(2, number_waves)
    peaks = 0.5 + 0.5*rnd.rand(number_waves)

    # Boundary smoothing
    bd = np.ones(xp.shape)
    nx, ny = np.round(edge_smoothing*NX), np.round(edge_smoothing*NY)
    left_bd = logical_and(xp >= -lx/2, xp <= -lx/2+nx*dx)
    right_bd = logical_and(xp >= lx/2-nx*dx, xp <= lx/2)
    lower_bd = logical_and(yp >= -ly/2, yp <= -ly/2+ny*dy)
    upper_bd = logical_and(yp >= ly/2-ny*dy, yp <= ly/2)
    edge1 = logical_and(left_bd, lower_bd)
    edge2 = logical_and(left_bd, upper_bd)
    edge3 = logical_and(upper_bd, right_bd)
    edge4 = logical_and(right_bd, lower_bd)
    f_left = (2/nx/dx)*(xp+lx/2) - (1/nx**2/dx**2)*(xp + lx/2)**2
    f_right = ((2/nx/dx)*(xp-(lx/2-2*nx*dx))
               - (1/nx**2/dx**2)*(xp-(lx/2-2*nx*dx))**2)
    f_lower = (2/ny/dy)*(yp+ly/2) - (1/ny**2/dy**2)*(yp + ly/2)**2
    f_upper = (((2/ny/dy)*(yp-(ly/2-2*ny*dy))
                - (1/ny**2/dy**2)*(yp-(ly/2-2*nx*dy))**2))
    bd[left_bd] = f_left[left_bd]
    bd[right_bd] = f_right[right_bd]
    bd[lower_bd] = f_lower[lower_bd]
    bd[upper_bd] = f_upper[upper_bd]
    bd[edge1] = f_left[edge1]*f_lower[edge1]
    bd[edge2] = f_left[edge2]*f_upper[edge2]
    bd[edge3] = f_upper[edge3]*f_right[edge3]
    bd[edge4] = f_right[edge4]*f_lower[edge4]
    bd[np.logical_not(wave)] = 1.

    # Relative permittivity
    for i in range(number_waves):
        epsilon_r[wave] = (epsilon_r[wave]
                           + peaks[i]*np.cos(2*pi/(lam_x[i])*xp[wave]
                                             - phi[0, i])
                           * np.cos(2*pi/(lam_y[i])*yp[wave] - phi[1, i]))
    epsilon_r[wave] = (rel_permittivity_amplitude*epsilon_r[wave]
                       / np.amax(epsilon_r[wave]))
    epsilon_r[wave] = epsilon_r[wave] + epsilon_rb
    epsilon_r = epsilon_r*bd
    epsilon_r[logical_and(wave, epsilon_r < 1.)] = 1.

    # Conductivity
    for i in range(number_waves):
        sigma[wave] = (sigma[wave]
                       + peaks[i]*np.cos(2*pi/(lam_x[i])*xp[wave]
                                         - phi[0, i])
                       * np.cos(2*pi/(lam_y[i])*yp[wave] - phi[1, i]))
    sigma[wave] = (conductivity_amplitude*sigma[wave]
                   / np.amax(sigma[wave]))
    sigma[wave] = sigma[wave] + sigma_b
    sigma = sigma*bd
    sigma[logical_and(wave, sigma < 0.)] = 0.

    return epsilon_r, sigma


def random_gaussians(number_distributions, maximum_spread=.8,
                     minimum_spread=.5, distance_from_border=.1,
                     resolution=None, surface_area=(1., 1.),
                     rel_permittivity_amplitude=0., conductivity_amplitude=0.,
                     axis_length_x=2., axis_length_y=2.,
                     background_conductivity=0.,
                     background_rel_permittivity=1.,
                     rel_permittivity=None, center=[0., 0.],
                     conductivity=None, rotate=0., edge_smoothing=0.03):
    """Draw random Gaussian distributions with specified electromagnetic properties.

    Parameters
    ----------
    number_distributions : int
        Number of Gaussian distributions.
    maximum_spread : float, default=0.8
        Maximum spread of Gaussian (proportional to area).
    minimum_spread : float, default=0.5
        Minimum spread of Gaussian.
    distance_from_border : float, default=0.1
        Minimum distance from border for distribution centers.
    resolution : tuple of int, optional
        Image resolution as (NY, NX).
    surface_area : tuple, default=(1.0, 1.0)
        Proportion of image dimensions for distribution area.
    rel_permittivity_amplitude : float, default=0.0
        Maximum amplitude of relative permittivity variation.
    conductivity_amplitude : float, default=0.0
        Maximum amplitude of conductivity variation in S/m.
    axis_length_x : float, default=2.0
        Physical length in x-direction.
    axis_length_y : float, default=2.0
        Physical length in y-direction.
    background_conductivity : float, default=0.0
        Background conductivity in S/m.
    background_rel_permittivity : float, default=1.0
        Background relative permittivity.
    rel_permittivity : numpy.ndarray, optional
        Existing permittivity array to overlay on.
    center : list of float, default=[0.0, 0.0]
        Center position as [x, y] coordinates.
    conductivity : numpy.ndarray, optional
        Existing conductivity array to overlay on.
    rotate : float, default=0.0
        Rotation angle in degrees.
    edge_smoothing : float, default=0.03
        Percentage of cells at boundary to smooth.

    Returns
    -------
    tuple of numpy.ndarray
        (epsilon_r, sigma) arrays with the Gaussian distributions drawn.
    """
    # Check input requirements
    if resolution is None and (rel_permittivity is None
                               or conductivity is None):
        raise error.MissingInputError('draw_random_gaussians',
                                      'resolution or relative'
                                      + '_permittivity_map or '
                                      + 'conductivity')

    # Make variable names more simple
    Lx, Ly = axis_length_x, axis_length_y
    epsilon_rb = background_rel_permittivity
    sigma_b = background_conductivity

    # Set map variables
    if rel_permittivity is None:
        epsilon_r = epsilon_rb*np.ones(resolution)
    else:
        epsilon_r = rel_permittivity
    if conductivity is None:
        sigma = sigma_b*np.ones(resolution)
    else:
        sigma = conductivity

    # Set discretization variables
    if resolution is None:
        resolution = epsilon_r.shape
    NY, NX = resolution
    dx, dy = Lx/NX, Ly/NY

    # Get meshgrid
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2, Lx/2, dx) + center[1],
                       np.arange(-Ly/2+dy/2, Ly/2, dy) + center[0])
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)

    # Wave area
    ly, lx = surface_area[0]*Ly, surface_area[1]*Lx
    area = logical_and(xp >= -lx/2,
                       logical_and(xp <= lx/2,
                                   logical_and(yp >= -ly/2, yp <= ly/2)))

    # Boundary smoothing
    bd = np.ones(xp.shape)
    nx, ny = np.round(edge_smoothing*NX), np.round(edge_smoothing*NY)
    left_bd = logical_and(xp >= -lx/2, xp <= -lx/2+nx*dx)
    right_bd = logical_and(xp >= lx/2-nx*dx, xp <= lx/2)
    lower_bd = logical_and(yp >= -ly/2, yp <= -ly/2+ny*dy)
    upper_bd = logical_and(yp >= ly/2-ny*dy, yp <= ly/2)
    edge1 = logical_and(left_bd, lower_bd)
    edge2 = logical_and(left_bd, upper_bd)
    edge3 = logical_and(upper_bd, right_bd)
    edge4 = logical_and(right_bd, lower_bd)
    f_left = (2/nx/dx)*(xp+lx/2) - (1/nx**2/dx**2)*(xp + lx/2)**2
    f_right = ((2/nx/dx)*(xp-(lx/2-2*nx*dx))
               - (1/nx**2/dx**2)*(xp-(lx/2-2*nx*dx))**2)
    f_lower = (2/ny/dy)*(yp+ly/2) - (1/ny**2/dy**2)*(yp + ly/2)**2
    f_upper = (((2/ny/dy)*(yp-(ly/2-2*ny*dy))
                - (1/ny**2/dy**2)*(yp-(ly/2-2*nx*dy))**2))
    bd[left_bd] = f_left[left_bd]
    bd[right_bd] = f_right[right_bd]
    bd[lower_bd] = f_lower[lower_bd]
    bd[upper_bd] = f_upper[upper_bd]
    bd[edge1] = f_left[edge1]*f_lower[edge1]
    bd[edge2] = f_left[edge2]*f_upper[edge2]
    bd[edge3] = f_upper[edge3]*f_right[edge3]
    bd[edge4] = f_right[edge4]*f_lower[edge4]
    bd[np.logical_not(area)] = 1.

    # General parameters
    s = np.zeros((2, number_distributions))
    xmin, xmax = -lx/2+distance_from_border*lx, lx/2-distance_from_border*lx
    ymin, ymax = -ly/2+distance_from_border*ly, ly/2-distance_from_border*ly

    # Relative permittivity
    y0 = ymin + rnd.rand(number_distributions)*(ymax-ymin)
    x0 = xmin + rnd.rand(number_distributions)*(xmax-xmin)
    s[0, :] = (minimum_spread + (maximum_spread-minimum_spread)
               * rnd.rand(number_distributions))*ly/6
    s[1, :] = (minimum_spread + (maximum_spread-minimum_spread)
               * rnd.rand(number_distributions))*lx/6
    phi = 2*pi*rnd.rand(number_distributions)
    A = rnd.rand(number_distributions)
    for i in range(number_distributions):
        sy, sx = s[0, i], s[1, i]
        x = np.cos(phi[i])*xp[area] + np.sin(phi[i])*yp[area]
        y = -np.sin(phi[i])*xp[area] + np.cos(phi[i])*yp[area]
        epsilon_r[area] = epsilon_r[area] + A[i]*np.exp(-((x-x0[i])**2
                                                          / (2*sx**2)
                                                          + (y-y0[i])**2
                                                          / (2*sy**2)))
    epsilon_r[area] = epsilon_r[area] - np.amin(epsilon_r[area])
    epsilon_r[area] = (rel_permittivity_amplitude*epsilon_r[area]
                       / np.amax(epsilon_r[area]))*rnd.rand()
    epsilon_r = epsilon_r*bd
    epsilon_r[area] = epsilon_r[area] + epsilon_rb

    # Conductivity
    y0 = ymin + rnd.rand(number_distributions)*(ymax-ymin)
    x0 = xmin + rnd.rand(number_distributions)*(xmax-xmin)
    s[0, :] = (minimum_spread + (maximum_spread-minimum_spread)
               * rnd.rand(number_distributions))*ly/6
    s[1, :] = (minimum_spread + (maximum_spread-minimum_spread)
               * rnd.rand(number_distributions))*lx/6
    phi = 2*pi*rnd.rand(number_distributions)
    A = rnd.rand(number_distributions)
    for i in range(number_distributions):
        sy, sx = s[0, i], s[1, i]
        x = np.cos(phi[i])*xp[area] + np.sin(phi[i])*yp[area]
        y = -np.sin(phi[i])*xp[area] + np.cos(phi[i])*yp[area]
        sigma[area] = sigma[area] + A[i]*np.exp(-((x-x0[i])**2/(2*sx**2)
                                                  + (y-y0[i])**2/(2*sy**2)))
    sigma[area] = sigma[area] - np.amin(sigma[area])
    sigma[area] = (conductivity_amplitude*sigma[area]
                   / np.amax(sigma[area]))
    sigma = sigma*bd
    sigma[area] = sigma[area] + sigma_b

    return epsilon_r, sigma


def isleft(x0, y0, x1, y1, x2, y2):
    r"""Check if a point is left, on, or right of an infinite line.

    The point to be tested is (x2, y2). The infinite line is defined by
    (x0, y0) -> (x1, y1).

    Parameters
    ----------
    x0, y0 : float
        A point on the infinite line.
    x1, y1 : float
        Another point on the infinite line.
    x2, y2 : float
        The point to be tested.

    Returns
    -------
    float
        < 0 if point is on the left,
        = 0 if point is on the line,
        > 0 if point is on the right.

    References
    ----------
    .. [1] Sunday, D 2012, Inclusion of a Point in a Polygon, accessed
       15 July 2020, <http://geomalgorithms.com/a03-_inclusion.html>
    """
    return (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)


def winding_number(x, y, xv, yv):
    r"""Check if a point is within a polygon using the Winding Number Algorithm.

    Parameters
    ----------
    x, y : float
        The point to be tested.
    xv, yv : numpy.ndarray
        1D arrays with vertex coordinates of the polygon.

    Returns
    -------
    bool
        True if the point is inside the polygon, False otherwise.

    References
    ----------
    .. [1] Sunday, D 2012, Inclusion of a Point in a Polygon, accessed
       15 July 2020, <http://geomalgorithms.com/a03-_inclusion.html>
    """
    # The first vertex must come after the last one within the array
    if xv[-1] != xv[0] or yv[-1] != yv[0]:
        _xv = np.hstack((xv.flatten(), xv[0]))
        _yv = np.hstack((yv.flatten(), yv[0]))
        n = xv.size
    else:
        _xv = np.copy(xv)
        _yv = np.copy(yv)
        n = xv.size-1

    wn = 0  # the  winding number counter

    # Loop through all edges of the polygon
    for i in range(n):  # edge from V[i] to V[i+1]

        if (_yv[i] <= y):  # start yv <= y
            if (_yv[i+1] > y):  # an upward crossing
                # P left of edge
                if (isleft(_xv[i], _yv[i], _xv[i+1], _yv[i+1], x, y) > 0):
                    wn += 1  # have  a valid up intersect

        else:  # start yv > y (no test needed)
            if (_yv[i+1] <= y):  # a downward crossing
                # P right of edge
                if (isleft(_xv[i], _yv[i], _xv[i+1], _yv[i+1], x, y) < 0):
                    wn -= 1  # have  a valid down intersect
    if wn == 0:
        return False
    else:
        return True
