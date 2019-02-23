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

import time as t


class GravitationalStellar(object):

    def __init__(self, integration_scheme="interlaced", stellar_mass_loss_timestep_fraction=0.1, gravity_model=SeBa,
                 stellar_model=Hermite, verbose=True):
        self.integration_scheme = integration_scheme
        self.stellar_mass_loss_timestep_fraction = stellar_mass_loss_timestep_fraction
        self.particles = None
        self.gravity_model = gravity_model
        self.stellar_model = stellar_model

        self.stellar_time = 0.0

        self.verbose = verbose

        self.gravity = None
        self.stellar = self.stellar_model()

        self.channel_from_stellar = None
        self.channel_from_framework_to_gravity = None
        self.channel_from_gravity_to_framework = None

        self.eccentricity_out_history = []
        self.eccentricity_in_history = []
        self.semimajor_axis_in_history = []
        self.semimajor_axis_out_history = []
        self.timestep_history = []

        self.semimajor_axis_init = None
        self.semimajor_axis_out_init = None
        self.eccentricity_out_init = None
        self.eccentricity_init = None

        self.elapsed_sim_time = 0.0
        self.elapsed_amuse_time = 0.0
        self.elapsed_total_time = 0.0

    def add_particles(self, particles):
        self.particles = particles

    def determine_timestep(self):
        star_timesteps = []
        for particle in self.particles:
            mass_loss = particle.mass_change
            delta_mass_max = self.stellar_mass_loss_timestep_fraction * particle.mass
            star_timesteps.append(delta_mass_max / mass_loss)

        return star_timesteps

    def evolve_model(self, end_time, number_steps=10000):

        start_time_all = t.time()

        time = 0.0 | end_time.unit

        delta_time_diagnostic = end_time / float(number_steps)

        stellar_time_point = self.stellar_time + time

        total_initial_energy = self.gravity.kinetic_energy + self.gravity.potential_energy
        total_previous_energy = total_initial_energy

        total_particle_mass = self.particles.mass.sum()

        semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out = self.get_orbital_elements_of_triple()
        self.timestep_history.append(time.value_in(units.Myr))
        self.semimajor_axis_in_history.append(semimajor_axis_in / self.semimajor_axis_init)
        self.eccentricity_in_history.append(eccentricity_in / self.eccentricity_init)
        self.semimajor_axis_out_history.append(semimajor_axis_out / self.semimajor_axis_out_init)
        self.eccentricity_out_history.append(eccentricity_out / self.eccentricity_out_init)

        while time < end_time:

            star_timesteps = self.determine_timestep()

            smallest_timestep = min(star_timesteps)

            smallest_timestep *= self.stellar_mass_loss_timestep_fraction

            if self.integration_scheme == "gravity_first":
                time = self.advance_gravity(time, smallest_timestep)
                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, smallest_timestep)

            elif self.integration_scheme == "stellar_first":
                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, smallest_timestep)
                time = self.advance_gravity(time, smallest_timestep)

            else:

                half_timestep = float(smallest_timestep / 2.)

                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, half_timestep)

                time = self.advance_gravity(time, smallest_timestep)

                stellar_time_point, delta_energy = self.advance_stellar(stellar_time_point, half_timestep)

                delta_energy_stellar += delta_energy

            if time >= delta_time_diagnostic:
                # Sees if diagnositcs should be printed out

                delta_time_diagnostic = time + delta_time_diagnostic

                kinetic_energy = self.gravity.kinetic_energy
                potential_energy = self.gravity.potential_energy
                total_energy = kinetic_energy + potential_energy
                delta_total_energy = total_previous_energy - total_energy

                total_mass = self.particles.mass.sum()
                if self.verbose:
                    print("T=", time, end=' ')
                    print("M=", total_mass, "(dM[SE]=", total_mass / total_particle_mass, ")", end=' ')
                    print("E= ", total_energy, "Q= ", kinetic_energy / potential_energy, end=' ')
                    print("dE=", (total_initial_energy - total_energy) / total_energy, "ddE=",
                          (total_previous_energy - total_energy) / total_energy, end=' ')
                    print("(dE[SE]=", delta_energy_stellar / total_energy, ")")

                total_initial_energy -= delta_total_energy
                total_previous_energy = total_energy

                # Break if binary is broken up
                semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out = self.get_orbital_elements_of_triple()

                if self.verbose:
                    print("Triple elements t=", (4 | units.Myr) + time,
                          "inner:", self.particles[0].mass, self.particles[1].mass, semimajor_axis_in, eccentricity_in,
                          "outer:", self.particles[2].mass, semimajor_axis_out, eccentricity_out)

                self.timestep_history.append(time.value_in(units.Myr))
                self.semimajor_axis_in_history.append(semimajor_axis_in / self.semimajor_axis_init)
                self.eccentricity_in_history.append(eccentricity_in / self.eccentricity_init)
                self.semimajor_axis_out_history.append(semimajor_axis_out / self.semimajor_axis_out_init)
                self.eccentricity_out_history.append(eccentricity_out / self.eccentricity_out_init)

                if eccentricity_out > 1.0 or semimajor_axis_out <= zero:
                    print("Binary ionized or merged")
                    break

            self.gravity.stop()
            self.stellar.stop()

            end_time_all = t.time()

            total_time_elapsed = end_time_all - start_time_all

            self.elapsed_total_time += total_time_elapsed

            self.elapsed_amuse_time = self.elapsed_total_time - self.elapsed_sim_time

            return self.timestep_history, self.semimajor_axis_in_history, self.eccentricity_in_history, \
                   self.semimajor_axis_out_history, self.eccentricity_out_history

    def advance_stellar(self, timestep, delta_time):
        Initial_Energy = self.gravity.kinetic_energy + self.gravity.potential_energy
        timestep += delta_time
        start_sim_time = t.time()
        self.stellar.evolve_model(timestep)
        elapsed_sim_time = t.time() - start_sim_time
        self.elapsed_sim_time += elapsed_sim_time
        self.channel_from_stellar.copy_attributes(["mass"])
        self.channel_from_framework_to_gravity.copy_attributes(["mass"])
        return timestep, self.gravity.kinetic_energy + self.gravity.potential_energy - Initial_Energy

    def advance_gravity(self, timestep, delta_time):
        timestep += delta_time
        start_sim_time = t.time()
        self.gravity.evolve_model(timestep)
        elapsed_sim_time = t.time() - start_sim_time
        self.elapsed_sim_time += elapsed_sim_time
        self.channel_from_gravity_to_framework.copy()
        return timestep

    def set_gravity(self, starting_semimajor_axis):
        converter = nbody_system.nbody_to_si(self.particles.mass.sum(), starting_semimajor_axis)
        self.gravity = self.gravity_model(converter)
        self.gravity.particles.add_particles(self.particles)

        self.channel_from_framework_to_gravity = self.particles.new_channel_to(self.gravity.particles)
        self.channel_from_gravity_to_framework = self.gravity.particles.new_channel_to(self.particles)

        self.gravity.particles.move_to_center()

    def age_stars(self, stellar_start_time):
        start_time_all = t.time()
        mult_stars = Particles(len(self.particles))
        for i in range(len(self.particles)):
            mult_stars[i].mass = self.particles[i].mass

        # Start Stellar Evolution
        self.stellar.particles.add_particles(mult_stars)
        self.channel_from_stellar = self.stellar.particles.new_channel_to(mult_stars)
        start_sim_time = t.time()
        self.stellar.evolve_model(stellar_start_time)
        end_sim_time = t.time()
        self.channel_from_stellar.copy_attributes(['mass'])
        for i in range(len(self.particles)):
            self.particles[i].mass = mult_stars[i].mass

        self.stellar_time = stellar_start_time

        end_time_all = t.time()

        self.elapsed_sim_time += end_sim_time - start_sim_time

        self.elapsed_total_time += start_time_all - end_time_all

        self.elapsed_amuse_time = self.elapsed_total_time - self.elapsed_sim_time

        return self.particles

    def get_orbital_elements_of_triple(self):
        inner_binary = self.particles[0] + self.particles[1]
        outer_binary = Particles(1)
        outer_binary[0].mass = inner_binary.mass.sum()
        outer_binary[0].position = inner_binary.center_of_mass()
        outer_binary[0].velocity = inner_binary.center_of_mass_velocity()
        outer_binary.add_particle(self.particles[2])
        _, _, semimajor_axis_in, eccentricity_in, _, _, _, _ \
            = orbital_elements_from_binary(inner_binary, G=constants.G)
        _, _, semimajor_axis_out, eccentricity_out, _, _, _, _ \
            = orbital_elements_from_binary(outer_binary, G=constants.G)
        return semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out

    def set_initial_parameters(self, semimajor_axis_init, eccentricity_init, semimajor_axis_out_init,
                               eccentricity_out_init):
        self.semimajor_axis_init = semimajor_axis_init
        self.semimajor_axis_out_init = semimajor_axis_out_init
        self.eccentricity_out_init = eccentricity_out_init
        self.eccentricity_init = eccentricity_init