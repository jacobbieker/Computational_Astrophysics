from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

#from .GravStellar import GravitationalStellar

import matplotlib.pyplot as plt
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

from prepare_figure import single_frame
from distinct_colours import get_distinct

# TODO: Makea plot of the orbital parameters
# TODO and the wall-clock computer time as a function of the time stepsize.
# TODO Based on these curves, can you decide what is the best
# TODO time step size for simulation a triplesystem like θMuscae

import time as t


class GravitationalStellar(object):

    def __init__(self, integration_scheme="interlaced", stellar_mass_loss_timestep_fraction=0.1, gravity_model=Hermite,
                 stellar_model=SeBa, verbose=True):
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

            # Starting mass
            prev_masses = self.particles.mass

            if self.integration_scheme == "gravity_first":
                time = self.advance_gravity(time, smallest_timestep)
                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, smallest_timestep)

            elif self.integration_scheme == "stellar_first":
                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, smallest_timestep)
                time = self.advance_gravity(time, smallest_timestep)

            else:

                half_timestep = smallest_timestep / 2.

                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, half_timestep)

                time = self.advance_gravity(time, smallest_timestep)

                stellar_time_point, delta_energy = self.advance_stellar(stellar_time_point, half_timestep)

                delta_energy_stellar += delta_energy

            # Now update the mass loss after the timestep
            new_masses = self.particles.mass
            for i in range(len(self.particles)):
                # TODO Come up with better method to update mass loss
                mass_change = (prev_masses[i] - new_masses[i]) / (self.stellar_mass_loss_timestep_fraction | units.yr)
                self.particles[i].mass_change = mass_change

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

        # Set the mass loss rate for the stars here: based off the book
        mass_changes_lit = [1.1E-5 | units.MSun / units.yr, 4.2E-7 | units.MSun / units.yr, 4.9E-8 | units.MSun / units.yr]
        for i in range(len(self.particles)):
            self.particles[i].mass_change =mass_changes_lit[i]


        end_time_all = t.time()

        if self.verbose:
            print("T=", self.stellar.model_time.in_(units.Myr))
            print("M=", self.stellar.particles.mass.in_(units.MSun))
            print("Masses at time T:", self.particles[0].mass, self.particles[1].mass, self.particles[2].mass)

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


def plot_results(computer_time, eccentricity_out, eccentricity_in, semimajor_axis_in, semimajor_axis_out,
                 stellar_mass_fraction):
    x_label = "$a/a_{0}$"
    y_label = "$e/e_{0}$"
    fig = single_frame(x_label, y_label, logx=False, logy=False,
                       xsize=10, ysize=8)
    color = get_distinct(4)

    time, ai, ei, ao, eo = computer_time, semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out
    plt.plot(ai, ei, c=color[0], label='inner')
    plt.plot(ao, eo, c=color[1], label='outer')

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)

    save_file = 'semi_and_eccen' \
                + '_dtse={:.3f}'.format(stellar_mass_fraction) \
                + '.png'
    plt.savefig(save_file)
    print('\nSaved figure in file', save_file, '\n')

    # Now plot the orbital params as function of timestep

    # First Eccentricity vs time

    plt.cla()
    plt.plot(computer_time, eccentricity_out, label='Eccentricity_out')
    plt.plot(computer_time, eccentricity_in, label='Eccentricity_in')
    plt.ylabel("Eccentricity")
    plt.xlabel("Computer Wall time")
    plt.savefig("eccentricity_vs_time_dts={:.3f}".format(stellar_mass_fraction) + ".png")

    # then Semimajor axis vs time

    plt.cla()
    plt.plot(computer_time, semimajor_axis_in, label='Semimajor Axis Out')
    plt.plot(computer_time, semimajor_axis_out, label='Semimajor Axis in')
    plt.ylabel("Semimajor Axis")
    plt.xlabel("Computer Wall time")
    plt.savefig("semimajor_axis_vs_time_dts={:.3f}".format(stellar_mass_fraction) + ".png")


def get_orbital_period(orbital_separation, total_mass):
    return 2 * np.pi * (orbital_separation ** 3 / (constants.G * total_mass)).sqrt()


def get_semi_major_axis(orbital_period, total_mass):
    return (constants.G * total_mass * orbital_period ** 2 / (4 * np.pi ** 2)) ** (1. / 3)


# Initial Conditions

eccentricity_init = 0.2
eccentricity_out_init = 0.6
semimajor_axis_out_init = 100 | units.AU

stellar_mass_loss_fraction = 0.9

M1 = 60 | units.MSun
M2 = 30 | units.MSun
M3 = 20 | units.MSun
period_init = 19 | units.day
semimajor_axis_init = 0.63 | units.AU

period_or_semimajor = 1  # 1: Get semimajor_axis from period, Otherwise: get period from semimjaor_axis

stellar_start_time = 4.0 | units.Myr
end_time = 0.55 | units.Myr
triple = Particles(3)
triple[0].mass = M1
triple[1].mass = M2
triple[2].mass = M3

grav_stellar = GravitationalStellar(stellar_mass_loss_timestep_fraction=stellar_mass_loss_fraction)
grav_stellar.add_particles(triple)
triple = grav_stellar.age_stars(stellar_start_time)
# Inner binary
tmp_stars = Particles(2)
tmp_stars[0].mass = triple[0].mass
tmp_stars[1].mass = triple[1].mass

if period_or_semimajor == 1:
    semimajor_axis_init = get_semi_major_axis(period_init, triple[0].mass + triple[1].mass)
else:
    period_init = get_orbital_period(semimajor_axis_init, triple[0].mass + triple[1].mass)

delta_time = 0.1 * period_init
mutual_inclination = 50  # Between inner and outer binary
inclination = 30  # Between the inner two stars
mean_anomaly = 180
argument_of_perigee = 180
longitude_of_the_ascending_node = 0

# Inner binary

r, v = get_position(triple[0].mass, triple[1].mass, eccentricity_init, semimajor_axis_init, mean_anomaly, inclination,
                    argument_of_perigee, longitude_of_the_ascending_node, delta_time)
tmp_stars[1].position = r
tmp_stars[1].velocity = v
tmp_stars.move_to_center()

# Outer binary

r, v = get_position(triple[0].mass + triple[1].mass, triple[2].mass, eccentricity_out_init, semimajor_axis_out_init, 0,
                    mutual_inclination, 0, 0, delta_time)
tertiary = Particle()
tertiary.mass = triple[2].mass
tertiary.position = r
tertiary.velocity = v
tmp_stars.add_particle(tertiary)
tmp_stars.move_to_center()

triple.position = tmp_stars.position
triple.velocity = tmp_stars.velocity

grav_stellar.set_initial_parameters(semimajor_axis_init, eccentricity_init,
                                    semimajor_axis_out_init, eccentricity_out_init)

grav_stellar.set_gravity(semimajor_axis_out_init)

timestep_history, semimajor_axis_in_history, eccentricity_in_history, \
semimajor_axis_out_history, eccentricity_out_history = grav_stellar.evolve_model(end_time)

plot_results(timestep_history, eccentricity_out_history, eccentricity_in_history, semimajor_axis_in_history,
             semimajor_axis_out_history, stellar_mass_loss_fraction)
