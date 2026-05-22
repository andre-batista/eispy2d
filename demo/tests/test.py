import sys
sys.path.insert(0, '../../../eispy2d')

from eispy2d.core import configuration as cfg

# Build configuration object
config = cfg.Configuration(name='cfg_test',
                           wavelength=1.,
                           wavelength_unit=True,
                           number_measurements=25,
                           number_sources=25,
                           image_size=[4., 4.],
                           observation_radius=6.,
                           background_permittivity=1.,
                           magnitude=1.,
                           perfect_dielectric=True)

# Computing the degrees of freedom for the equivalent problem
dof = cfg.degrees_of_freedom(1., epsilon_r=1.25, frequency=config.f)
print('The DOF for this problem is: %d\n' % dof)

# Print object info
print(config)

# Plot configuration setup
config.draw(show=True)