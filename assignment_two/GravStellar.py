from amuse.lab import *
import numpy as np
import matplotlib.pyplot as plt
from amuse.ext.solarsystem import get_position

from amuse.units import units, constants, nbody_system
from amuse.units.quantities import zero
from amuse.datamodel import Particle, Particles
from amuse.support.console import set_printing_strategy

from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary

from amuse.community.huayno.interface import Huayno
from amuse.community.smalln.interface import SmallN
from amuse.community.hermite0.interface import Hermite
from amuse.community.seba.interface import SeBa
from amuse.community.sse.interface import SSE


class Gravitational_Stellar(object):

    def __init__(self, integration_scheme, stellar_mass_loss_timestep_fraction, gravity_model=SeBa, stellar_model=Hermite):
        self.integration_scheme = integration_scheme
        self.stellar_mass_loss_timestep_fraction = stellar_mass_loss_timestep_fraction
        self.particles = None
        self.gravity_model = gravity_model
        self.stellar_model = stellar_model

        self.stellar_time = 0.0

        self.gravity = None
        self.stellar = self.stellar_model()

        self.channel_from_stellar = None
        self.channel_from_framework_to_gravity = None
        self.channel_from_gravity_to_framework = None


    def add_particles(self, particles):
        self.particles = particles

    def determine_timestep(self):
        star_timesteps = []
        for particle in self.particles:
            mass_loss = particle.mass_change
            delta_mass_max = self.stellar_mass_loss_timestep_fraction * particle.mass
            star_timesteps.append(delta_mass_max / mass_loss)

        return star_timesteps


    def evolve_model(self, end_time):

        time = 0.0 | end_time.unit

        stellar_time_point = self.stellar_time + time

        while time < end_time:

            if self.integration_scheme == "interlaced":

                star_timesteps = self.determine_timestep()

                smallest_timestep = min(star_timesteps)

                smallest_timestep *= self.stellar_mass_loss_timestep_fraction

                half_timestep = float(smallest_timestep/2.)

                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, half_timestep)

                time = self.advance_gravity(time, smallest_timestep)

                stellar_time_point, delta_energy = self.advance_stellar(stellar_time_point, half_timestep)

                delta_energy_stellar += delta_energy

                return NotImplementedError
            elif self.integration_scheme == "gravity_first":
                return NotImplementedError
            elif self.integration_scheme == "stellar_first":
                return NotImplementedError
            return NotImplementedError

    def advance_stellar(self, timestep, delta_time):
        Initial_Energy = self.gravity.kinetic_energy + self.gravity.potential_energy
        timestep += delta_time
        self.stellar.evolve_model(timestep)
        self.channel_from_stellar.copy_attributes(["mass"])
        self.channel_from_framework_to_gravity.copy_attributes(["mass"])
        return timestep, self.gravity.kinetic_energy + self.gravity.potential_energy - Initial_Energy

    def advance_gravity(self, timestep, delta_time):
        timestep += delta_time
        self.gravity.evolve_model(timestep)
        self.channel_from_gravity_to_framework.copy()
        return timestep

    def set_gravity(self, starting_semimajor_axis):
        converter = nbody_system.nbody_to_si(self.particles.mass.sum(), starting_semimajor_axis)
        self.gravity = self.gravity_model(converter)
        self.gravity.particles.add_particles(self.particles)

        self.channel_from_framework_to_gravity = self.particles.new_channel_to(self.gravity.particles)
        self.channel_from_gravity_to_framework = self.gravity.particles.new_channel_to(self.particles)

        total_initial_energy = self.gravity.kinetic_energy + self.gravity.potential_energy
        self.total_previous_energy = total_initial_energy

        self.gravity.particles.move_to_center()

    def age_stars(self, stellar_start_time):
        mult_stars = Particles(len(self.particles))
        for i in range(len(self.particles)):
            mult_stars[i].mass = self.particles[i].mass

        # Start Stellar Evolution
        self.stellar.particles.add_particles(mult_stars)
        self.channel_from_stellar = self.stellar.particles.new_channel_to(mult_stars)
        self.stellar.evolve_model(stellar_start_time)
        self.channel_from_stellar.copy_attributes(['mass'])
        for i in range(len(self.particles)):
            self.particles[i].mass = mult_stars[i].mass

        self.stellar_time = stellar_start_time


class GravStellar(object):
    """
    Class to do the different timestep updates for the stellar and gravitational updates

    Should be given bodies used, and the run the
    """

    def __init__(self, timestep, end_time, number_timesteps, integration_scheme, stellar_mass_loss_frac, stars, stellar_start_time,
                 gravity_model=SeBa, stellar_model=Hermite, verbose=False):

        self.timestep = timestep
        self.number_timesteps = number_timesteps
        self.num_bodies = len(stars)
        self.verbose = verbose
        self.gravity_model = gravity_model
        self.stellar_model = stellar_model
        self.stars = stars
        self.end_time = end_time
        self.integration_scheme = integration_scheme
        self.stellar_mass_loss_time_step_fraction = stellar_mass_loss_frac
        self.stellar_start_time = stellar_start_time

        self.gravity = None
        self.stellar = None

        self.channel_from_stellar = None
        self.channel_from_framework_to_gravity = None
        self.channel_from_gravity_to_framework = None

        self.eccentricity_out_history = []
        self.eccentricity_in_history = []
        self.semimajor_axis_in_history = []
        self.semimajor_axis_out_history = []
        self.timestep_history = []


        self.time = 0.0 | end_time.unit


    def advance_stellar(self, ts, dt):
        E0 = self.gravity.kinetic_energy + self.gravity.potential_energy
        ts += dt
        self.stellar.evolve_model(ts)
        channel_from_stellar.copy_attributes(["mass"])
        channel_from_framework_to_gd.copy_attributes(["mass"])
        return ts, gravity.kinetic_energy + self.gravity.potential_energy - E0

    def advance_gravity(self, tg, dt):
        tg += dt
        self.gravity.evolve_model(tg)
        channel_from_gd_to_framework.copy()
        return tg
    def get_orbital_period(self):
        return 2*np.pi*(a**3/(constants.G*Mtot)).sqrt()

    def get_semi_major_axis(self):
        return (constants.G*Mtot*P**2/(4*np.pi**2))**(1./3)

    def get_mass_loss(self):
        return NotImplementedError

    def get_orbital_elements_of_triple(self):
        inner_binary = self.stars[0]+self.stars[1]
        outer_binary = Particles(1)
        outer_binary[0].mass = inner_binary.mass.sum()
        outer_binary[0].position = inner_binary.center_of_mass()
        outer_binary[0].velocity = inner_binary.center_of_mass_velocity()
        outer_binary.add_particle(self.stars[2])
        _, _, semimajor_axis_in, eccentricity_in, _, _, _, _ \
            = orbital_elements_from_binary(inner_binary, G=constants.G)
        _, _, semimajor_axis_out, eccentricity_out, _, _, _, _ \
            = orbital_elements_from_binary(outer_binary, G=constants.G)
        return semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out

    def evolve(self):

        dt_diag = self.end_time / float(self.number_timesteps)

        self.timestep_history = [self.time.value_in(units.Myr)]
        self.semimajor_axis_in_history = [semimajor_axis_in/semimajor_axis_in_0]

        while self.time < self.end_time:

            # TODO: Add determining timestep

            if self.integration_scheme == 1:

                ts, dE_se = self.advance_stellar(ts, dt)
                time = self.advance_gravity(time, dt)

            elif self.integration_scheme == 2:

                time = self.advance_gravity(time, dt)
                ts, dE_se = self.advance_stellar(ts, dt)

            else:

                dE_se = zero
                #ts, dE_se = advance_stellar(ts, dt/2)
                time = self.advance_gravity(time, dt)
                #ts, dE = advance_stellar(ts, dt/2)
                #dE_se += dE

            if self.time >= self.t_diag:

                t_diag = time + dt_diag

                Ekin = self.gravity.kinetic_energy
                Epot = self.gravity.potential_energy
                Etot = Ekin + Epot
                dE = Etot_prev - Etot
                Mtot = self.stars.mass.sum()
                if self.verbose:
                    print("T=", time, end=' ')
                    print("M=", Mtot, "(dM[SE]=", Mtot / Mtriple, ")", end=' ')
                    print("E= ", Etot, "Q= ", Ekin / Epot, end=' ')
                    print("dE=", (Etot_init - Etot) / Etot, "ddE=", (Etot_prev - Etot) / Etot, end=' ')
                    print("(dE[SE]=", dE_se / Etot, ")")
                Etot_init -= dE
                Etot_prev = Etot
                semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out = self.get_orbital_elements_of_triple()
                if self.verbose:
                    print("Triple elements t=", (4 | units.Myr) + time,
                          "inner:", self.stars[0].mass, self.stars[1].mass, semimajor_axis_in, eccentricity_in,
                          "outer:", self.stars[2].mass, semimajor_axis_out, eccentricity_out)

                self.timestep_history.append(time.value_in(units.Myr))
                self.semimajor_axis_in_history.append(semimajor_axis_in/ain_0)
                self.eccentricity_in_history.append(eccentricity_in/ein_0)
                self.semimajor_axis_out_history.append(semimajor_axis_out/aout_0)
                self.eccentricity_out_history.append(eccentricity_out/eout_0)

                if eccentricity_out > 1.0 or semimajor_axis_out <= zero:
                    print("Binary ionized or merged")
                    break

        self.gravity.stop()
        self.stellar.stop()

        return self.timestep_history, self.semimajor_axis_in_history, self.eccentricity_in_history, \
               self.semimajor_axis_out_history, self.eccentricity_out_history

    def set_gravity(self, stars):
        converter = nbody_system.nbody_to_si(stars.mass.sum(), aout_0)
        self.gravity = self.gravity_model(converter)
        self.gravity.particles.add_particles(stars)

        self.channel_from_framework_to_gravity = stars.new_channel_to(self.gravity.particles)
        self.channel_from_gravity_to_framework = self.gravity.particles.new_channel_to(stars)

        total_initial_energy = self.gravity.kinetic_energy + self.gravity.potential_energy
        self.total_previous_energy = total_initial_energy

        self.gravity.particles.move_to_center()



    def age_stars(self):
        mult_stars = Particles(self.num_bodies)
        for i in range(self.num_bodies):
            mult_stars[i].mass = self.stars[i].mass

        # Start Stellar Evolution

        stellar = self.stellar_model()
        stellar.particles.add_particles(mult_stars)
        self.channel_from_stellar = stellar.particles.new_channel_to(mult_stars)
        stellar.evolve_model(self.stellar_start_time)
        self.channel_from_stellar.copy_attributes(['mass'])
        for i in range(self.num_bodies):
            self.stars[i].mass = mult_stars[i].mass



