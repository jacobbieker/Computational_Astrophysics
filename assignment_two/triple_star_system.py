from .GravStellar import GravitationalStellar

import numpy as np

from amuse.units import units, constants, nbody_system
from amuse.units.quantities import zero
from amuse.datamodel import Particle, Particles
from amuse.support.console import set_printing_strategy
from amuse.io import store

from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary

from amuse.community.huayno.interface import Huayno
from amuse.community.smalln.interface import SmallN
from amuse.community.hermite0.interface import Hermite
from amuse.community.seba.interface import SeBa
from amuse.community.sse.interface import SSE

from amuse.ext.solarsystem import get_position

# TODO: Makea plot of the orbital parameters
# TODO and the wall-clock computer time as a function of the time stepsize.
# TODO Based on these curves, can you decide what is the best
# TODO time step size for simulation a triplesystem like θMuscae

"""
TODO

1. Make a plot of the orbital parameters and the wall-clock computer time as a function of the time stepsize. 
Based on these curves, can you decide what is the best time step size for simulation a triplesystem likeθMuscae

2. Use theinterlaced temporal discretization method and study if 
this is really a preferred way of integrating two numerical solvers

3. Run several simulations, subsequently reducing the time step for each of them and record thesemi-major axis 
and eccentricity of the inner and the outer binary, and their relative inclination.

4. Find in the literature the masses, mass-loss rates of the three stars of θMuscae and their orbital parameters. 
 Generate a triple system with the appropriate orbital elements. 
 Might need to use the new_binary_from_orbital_elements mentioned

5. Answer the Questions at the bottom


"""

def get_orbital_period(orbital_separation, total_mass):
    return 2 * np.pi * (orbital_separation ** 3 / (constants.G * total_mass)).sqrt()

def get_semi_major_axis(orbital_period, total_mass):
    return (constants.G * total_mass * orbital_period ** 2 / (4 * np.pi ** 2)) ** (1. / 3)


# Initial Conditions
mutual_inclination = 30
ma = 180
aop = 180
lon = 0

M1 = 60
M2 = 30
M3 = 20
period = 19 | units.day
separation = 0.63 | units.AU


period_or_semimajor = 1

stellar_start_time = 4.0 | units.Myr
end_time = 0.55|units.Myr
triple = Particles(3)
triple[0].mass = M1
triple[1].mass = M2
triple[2].mass = M3

grav_stellar = GravitationalStellar()
grav_stellar.add_particles(triple)
triple = grav_stellar.age_stars(stellar_start_time)

# Inner binary
tmp_stars = Particles(2)
tmp_stars[0].mass = triple[0].mass
tmp_stars[1].mass = triple[1].mass


if period_or_semimajor == 1:
    semimajor_axis_init = get_semi_major_axis(period, triple[0].mass+triple[1].mass)
else:
    period_init = get_orbital_period(separation, triple[0].mass+triple[1].mass)

delta_time = 0.1*period_init

r, v = get_position(triple[0].mass, triple[1].mass, eccentricity_init, semimajor_axis_init, ma, inc, aop, lon, delta_time)
tmp_stars[0].position = r
tmp_stars[1].position = v
tmp_stars.move_to_center()

# Outer binary

r, v = get_position(triple[0].mass+triple[1].mass, triple[2].mass, eccentricity_out_init, semimajor_axis_out_init, 0, 0, 0, 0, delta_time)
tertiary = Particle()
tertiary.mass = triple[2].mass
tertiary.position = r
tertiary.velocity = v
tmp_stars.add_particle(tertiary)
tmp_stars.move_to_center()

triple.position = tmp_stars.position
triple.velocity = tmp_stars.velocity

grav_stellar.set_initial_parameters(semimajor_axis_init, eccentricity_init,
                                    semimajor_axis_out_init,eccentricity_out_init)

grav_stellar.set_gravity(semimajor_axis_out_init)

grav_stellar.evolve_model(end_time)