from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# from .GravStellar import GravitationalStellar

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

import time as t


class GravitationalStellar(object):
    """
    This class encapsulates both gravitational and stellar dynamics into one.
    """

    def __init__(self, integration_scheme="interlaced", stellar_mass_loss_timestep_fraction=0.1, gravity_model=Hermite,
                 stellar_model=SeBa, verbose=True, interpolate=True, inclination=60):
        self.integration_scheme = integration_scheme
        self.stellar_mass_loss_timestep_fraction = stellar_mass_loss_timestep_fraction
        self.particles = None
        self.gravity_model = gravity_model
        self.stellar_model = stellar_model

        self.stellar_time = 0.0

        self.verbose = verbose
        self.interpolate = interpolate
        self.masses = []
        self.times = []
        self.inclination = inclination
        self.period_init = 0.0

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
        self.mass_history = []
        self.inclination_history = []

        self.semimajor_axis_init = None
        self.semimajor_axis_out_init = None
        self.eccentricity_out_init = None
        self.eccentricity_init = None

        self.elapsed_sim_time = 0.0
        self.elapsed_amuse_time = 0.0
        self.elapsed_total_time = 0.0

    def add_particles(self, particles):
        """
        Adds particles to the object
        :param particles: AMUSE Particles
        """
        self.particles = particles

    def determine_timestep(self):
        """
        Determines the timestep based off the stellar mass fraction that can be lost per timestep
        :return: The list of all possible timesteps
        """
        star_timesteps = []
        wind_loss = -1 * self.stellar.particles.wind_mass_loss_rate.in_(units.MSun / units.yr)
        # Max loss rate: stellar_mass_loss_timestep_fraction * total_mass of the star, usually 0.1% or 0.001
        for index, particle in enumerate(self.particles):
            mass_loss = wind_loss[index]  # MSun/yr
            change = 1. / mass_loss  # yr/MSun
            change *= particle.mass  # yr/MSun * MSun = yr
            change *= self.stellar_mass_loss_timestep_fraction
            star_timesteps.append(change)

        return star_timesteps

    def evolve_model(self, end_time, number_steps=100):
        """
        Evolves the combined gravitational and stellar evolution codes to the given end time
        :param end_time: End time of the simulation
        :param number_steps: Number of diagnostic timesteps to use
        :return: Lists of the histories of orbital parameters
        """

        start_time_all = t.time()

        time = 0.0 | end_time.unit

        delta_time_diagnostic = end_time / float(number_steps)

        stellar_time_point = self.stellar_time + time

        total_initial_energy = self.gravity.kinetic_energy + self.gravity.potential_energy
        total_previous_energy = total_initial_energy

        total_particle_mass = self.particles.mass.sum()

        semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out = self.get_orbital_elements_of_triple()
        # Set the first original values
        self.timestep_history.append(time.value_in(units.Myr))
        self.semimajor_axis_in_history.append(semimajor_axis_in / self.semimajor_axis_init)
        self.eccentricity_in_history.append(eccentricity_in / self.eccentricity_init)
        self.semimajor_axis_out_history.append(semimajor_axis_out / self.semimajor_axis_out_init)
        self.eccentricity_out_history.append(eccentricity_out / self.eccentricity_out_init)
        self.inclination_history.append(self.inclination/self.inclination)

        if self.interpolate:

            # Create arrays of stellar times and masses for interpolation.

            self.times = [time]
            self.masses = [self.particles.mass.copy()]
            while time <= (end_time + (end_time / 2.)):
                time += 1.e-3 | units.Myr
                self.stellar.evolve_model(self.stellar_time + time)
                self.channel_from_stellar.copy_attributes(["mass"])
                self.times.append(time)
                self.masses.append(self.particles.mass.copy())

            time = 0.0 | end_time.unit

        while time < end_time:

            star_timesteps = self.determine_timestep()

            smallest_timestep = min(star_timesteps)

            smallest_timestep = smallest_timestep.value_in(units.yr) | units.yr

            if time + smallest_timestep > delta_time_diagnostic:
                # If timestep goes past the delta_time_diagnostic, changes it to exactly get to the diagnositc time
                smallest_timestep = delta_time_diagnostic - time

            if self.integration_scheme == "gravity_first":
                # Integrates a full step of gravity then a full step of stellar
                time = self.advance_gravity(time, smallest_timestep)
                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, smallest_timestep)

            elif self.integration_scheme == "stellar_first":
                # Integrates a full step of stellar then a full step of gravity
                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, smallest_timestep)
                time = self.advance_gravity(time, smallest_timestep)

            elif self.integration_scheme == "diagnostic":
                # Integrates gravity to the diagnositc time, does not include stellar evolution
                smallest_timestep = delta_time_diagnostic - time

                if smallest_timestep > 0 | smallest_timestep.unit:
                    time = self.advance_gravity(time, smallest_timestep)

            else:
                # Interpolated time steps, first a half step of stellar, then a full step of gravity, then a
                # half step of stellar again

                half_timestep = smallest_timestep / 2.

                stellar_time_point, delta_energy_stellar = self.advance_stellar(stellar_time_point, half_timestep)

                time = self.advance_gravity(time, smallest_timestep)

                stellar_time_point, delta_energy = self.advance_stellar(stellar_time_point, half_timestep)

                delta_energy_stellar += delta_energy

            # print("End time: ", end_time)
            total_mass = self.particles.mass.sum()

            semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out = self.get_orbital_elements_of_triple()

            inclination, mutual_inclination = self.get_inclination()

            if self.verbose:
                print("Triple elements t=", (4 | units.Myr) + time,
                      "inner:", self.particles[0].mass, self.particles[1].mass, semimajor_axis_in, eccentricity_in,
                      "outer:", self.particles[2].mass, semimajor_axis_out, eccentricity_out)

            # Saves every timestep of the simulation to have better graphs
            self.timestep_history.append(time.value_in(units.yr))
            self.mass_history.append(total_mass.value_in(units.MSun))
            self.semimajor_axis_in_history.append(semimajor_axis_in / self.semimajor_axis_init)
            self.eccentricity_in_history.append(eccentricity_in / self.eccentricity_init)
            self.semimajor_axis_out_history.append(semimajor_axis_out / self.semimajor_axis_out_init)
            self.eccentricity_out_history.append(eccentricity_out / self.eccentricity_out_init)
            self.inclination_history.append(inclination/self.inclination)

            if time >= delta_time_diagnostic:
                # Sees if diagnositcs should be printed out

                delta_time_diagnostic = time + delta_time_diagnostic

                kinetic_energy = self.gravity.kinetic_energy
                potential_energy = self.gravity.potential_energy
                total_energy = kinetic_energy + potential_energy
                delta_total_energy = total_previous_energy - total_energy

                total_mass = self.particles.mass.sum()
                try:
                    print(delta_energy_stellar)
                except:
                    delta_energy_stellar = 0.
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

                if eccentricity_out > 1.0 or semimajor_axis_out <= zero:
                    print("Binary ionized or merged")
                    break

        self.gravity.stop()
        self.stellar.stop()

        end_time_all = t.time()

        total_time_elapsed = end_time_all - start_time_all

        # Saves the wall time of the simulation time, total elapsed time, and amuse time

        self.elapsed_total_time += total_time_elapsed

        self.elapsed_amuse_time = self.elapsed_total_time - self.elapsed_sim_time

        return self.timestep_history, self.mass_history, self.semimajor_axis_in_history, self.eccentricity_in_history, \
               self.semimajor_axis_out_history, self.eccentricity_out_history, self.inclination_history

    def advance_stellar(self, timestep, delta_time):
        """
        Advances the stellar evolution code, either from interpolating from saved values, or by directly simulating it
        :param timestep: Starting time
        :param delta_time: Amount of time to go forward
        :return: Timestep and change in energy of the system
        """
        Initial_Energy = self.gravity.kinetic_energy + self.gravity.potential_energy
        timestep += delta_time

        if self.interpolate:
            interpolate_t = timestep - self.stellar_time
            print(interpolate_t)
            i = int(interpolate_t.value_in(units.Myr) / 1.e-3)
            mass = self.masses[i] + (interpolate_t - self.times[i]) * (self.masses[i + 1] - self.masses[i]) / (
                        1.e-3 | units.Myr)
            self.particles.mass = mass

        else:
            start_sim_time = t.time()
            self.stellar.evolve_model(timestep)
            elapsed_sim_time = t.time() - start_sim_time
            self.elapsed_sim_time += elapsed_sim_time
            self.channel_from_stellar.copy_attributes(["mass"])

        self.channel_from_framework_to_gravity.copy_attributes(["mass"])
        return timestep, self.gravity.kinetic_energy + self.gravity.potential_energy - Initial_Energy

    def advance_gravity(self, timestep, delta_time):
        """
        Advances the gravity simulation from the given timestep by the delta_time
        :param timestep: Starting time of the simulation
        :param delta_time: Amount of time to go forward
        :return: Returns the new time for the system
        """
        timestep += delta_time
        start_sim_time = t.time()
        self.gravity.evolve_model(timestep)
        elapsed_sim_time = t.time() - start_sim_time
        self.elapsed_sim_time += elapsed_sim_time
        self.channel_from_gravity_to_framework.copy()
        return timestep

    def set_gravity(self, starting_semimajor_axis):
        """
        Sets up the gravity model of the system and opens the channel
        :param starting_semimajor_axis: The start semimajor axis of the system
        """
        converter = nbody_system.nbody_to_si(self.particles.mass.sum(), starting_semimajor_axis)
        self.gravity = self.gravity_model(converter)
        # Set different timesteps based off the model
        if self.gravity_model == Hermite:
            self.gravity.parameters.timestep_parameter = 0.01
        elif self.gravity_model == SmallN:
            self.gravity.parameters.timestep_parameter = 0.01
            self.gravity.parameters.full_unperturbed = 0
        elif self.gravity_model == Huayno:
            self.gravity.parameters.inttype_parameter = 20
            self.gravity.parameters.timestep = (1. / 256) * self.period_init

        self.gravity.particles.add_particles(self.particles)

        self.channel_from_framework_to_gravity = self.particles.new_channel_to(self.gravity.particles)
        self.channel_from_gravity_to_framework = self.gravity.particles.new_channel_to(self.particles)

        self.gravity.particles.move_to_center()

    def age_stars(self, stellar_start_time):
        """
        Ages the particles of the GravitationalStellar object up to the stellar start time
        :param stellar_start_time: Time to age the stars
        :return: the particles aged to that time
        """
        start_time_all = t.time()
        # Start Stellar Evolution
        self.stellar.particles.add_particles(self.particles)
        self.channel_from_stellar = self.stellar.particles.new_channel_to(self.particles)
        start_sim_time = t.time()
        self.stellar.evolve_model(stellar_start_time)
        end_sim_time = t.time()
        self.channel_from_stellar.copy_attributes(['mass'])

        self.stellar_time = stellar_start_time

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
        """
        Returns the eccentricity and semimajor axis of the inner and outer binaries
        :return:
        """
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

    def get_inclination(self):
        """
        Obtains and returns the inclinations of the system
        """
        inner_binary = self.particles[0] + self.particles[1]
        outer_binary = Particles(1)
        outer_binary[0].mass = inner_binary.mass.sum()
        outer_binary[0].position = inner_binary.center_of_mass()
        outer_binary[0].velocity = inner_binary.center_of_mass_velocity()
        outer_binary.add_particle(self.particles[2])
        _, _, _, _, _, inclination, _, _ \
            = orbital_elements_from_binary(inner_binary, G=constants.G)
        _, _, _, _, _, mutual_inclination, _, _ \
            = orbital_elements_from_binary(outer_binary, G=constants.G)
        return inclination, mutual_inclination

    def set_initial_parameters(self, semimajor_axis_init, eccentricity_init, semimajor_axis_out_init,
                               eccentricity_out_init, period_init):
        """
        Sets the initial orbital parameters
        :param semimajor_axis_init: Semimajor axis of the inner binary
        :param eccentricity_init: Eccentricity of the inner binary
        :param semimajor_axis_out_init: Semimajor axis of the outer binary
        :param eccentricity_out_init: Eccentricity of the outer binary
        :param period_init: Period of the inner system
        """
        self.semimajor_axis_init = semimajor_axis_init
        self.semimajor_axis_out_init = semimajor_axis_out_init
        self.eccentricity_out_init = eccentricity_out_init
        self.eccentricity_init = eccentricity_init
        self.period_init = period_init


def plot_results(computer_time, eccentricity_out, eccentricity_in, semimajor_axis_in, semimajor_axis_out,
                 stellar_mass_fraction, grav_stellar):
    """
    Plot outputs from a single simulation
    :param computer_time: list of times from simulation
    :param eccentricity_out: List of eccentricities of outer binary
    :param eccentricity_in: List of eccentricites of the inner binary
    :param semimajor_axis_in: List of semimajor axis of inner binary
    :param semimajor_axis_out: List of semimajor axis of outer binary
    :param stellar_mass_fraction: Stellar mass fraction to determine timestep
    :param grav_stellar: GravitationalStellar object
    :return:
    """
    x_label = "$a/a_{0}$"
    y_label = "$e/e_{0}$"
    fig = single_frame(x_label, y_label, logx=False, logy=False,
                       xsize=10, ysize=8)
    color = get_distinct(4)
    plt.cla()

    time, ai, ei, ao, eo = computer_time, semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out
    plt.plot(ai, ei, c=color[0], label='inner')
    plt.plot(ao, eo, c=color[1], label='outer')

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.tight_layout()
    save_file = 'semi_and_eccen' \
                + '_dtse={:.7f}'.format(stellar_mass_fraction) \
                + '.png'
    plt.savefig(save_file)
    print('\nSaved figure in file', save_file, '\n')

    grav_models = ['hermite', 'huayno', 'smalln']
    if grav_stellar.gravity_model == Hermite:
        grav_model = grav_models[0]
    elif grav_stellar.gravity_model == Huayno:
        grav_model = grav_models[1]
    else:
        grav_model = grav_models[2]

    # Now plot the orbital params as function of timestep

    # First Eccentricity vs time

    plt.cla()
    plt.plot(computer_time, eccentricity_out, label='Outer')
    plt.plot(computer_time, eccentricity_in, label='Inner')
    plt.legend(loc='best')
    plt.ylabel("Eccentricity/(Initial Eccentricity)")
    plt.xlabel("Time (years)")
    plt.title("Wall Clock Time: {} s\n Sim Time: {} s".format(np.round(grav_stellar.elapsed_total_time, 3),
                                                              np.round(grav_stellar.elapsed_sim_time, 3)))
    plt.tight_layout()
    plt.savefig(
        "eccentricity_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method={}".format(stellar_mass_fraction, grav_stellar.inclination, grav_model,
                                                                      grav_stellar.integration_scheme) + ".png")

    # then Semimajor axis vs time

    plt.cla()
    plt.plot(computer_time, semimajor_axis_in, label="Outer")
    plt.plot(computer_time, semimajor_axis_out, label='Inner')
    plt.legend(loc='best')
    plt.ylabel("Semimajor Axis/(Initial Semimajor Axis)")
    plt.xlabel("Time (years)")
    plt.title("Wall Clock Time: {} s\n Sim Time: {} s".format(np.round(grav_stellar.elapsed_total_time, 3),
                                                              np.round(grav_stellar.elapsed_sim_time, 3)))
    plt.tight_layout()
    plt.savefig(
        "semimajor_axis_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method={}".format(stellar_mass_fraction, grav_stellar.inclination, grav_model,
                                                                        grav_stellar.integration_scheme) + ".png")
    plt.cla()


def plot_all_results(computer_times, eccentricity_outs, eccentricity_ins, semimajor_axis_ins, semimajor_axis_outs,
                     stellar_mass_fraction, grav_stellars, inclinations):
    """
    Plots results combining multiple outputs into one. Specifically, this makes a plot of all 4 different integration
    methods in one. Saves out all files.
    :param computer_times: List of time lists
    :param eccentricity_outs: List of lists of the outer binary eccentricities
    :param eccentricity_ins:  List of lists of the inner binary eccentricities
    :param semimajor_axis_ins: List of lists of the inner binary semimajor axis
    :param semimajor_axis_outs: List of lists of the outer binary semimajor axis
    :param stellar_mass_fraction: The stellar mass fraction used to determine the timestep
    :param grav_stellars: List of GravitationalStellar objects
    :param inclinations: List of lists of the inclination between the binaries
    :return:
    """
    schemes = ["interlaced", "stellar_first", "gravity_first", "diagnostic"]
    grav_models = ['hermite', 'huayno', 'smalln']
    if grav_stellars[0].gravity_model == Hermite:
        grav_model = grav_models[0]
    elif grav_stellars[0].gravity_model == Huayno:
        grav_model = grav_models[1]
    else:
        grav_model = grav_models[2]
    avg_wall_time = 0.0
    avg_sim_time = 0.0
    plt.cla()
    for j in range(len(computer_times)):
        plt.plot(computer_times[j], semimajor_axis_ins[j], label="inner, {}".format(schemes[j]))
        plt.plot(computer_times[j], semimajor_axis_outs[j], label="outer, {}".format(schemes[j]))
        avg_wall_time += grav_stellars[j].elapsed_total_time
        avg_sim_time += grav_stellars[j].elapsed_sim_time

    avg_wall_time /= len(computer_times)
    avg_sim_time /= len(computer_times)

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Semimajor Axis/(Initial Semimajor Axis)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("ALL_semimajor_axis_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method=all".format(stellar_mass_fraction,
                                                                                       grav_stellars[0].inclination, grav_model) + ".png")
    plt.cla()

    for j in range(len(computer_times)):
        plt.plot(computer_times[j], semimajor_axis_ins[j], label="inner, {}".format(schemes[j]))

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Semimajor Axis/(Initial Semimajor Axis)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("semimajor_axis_inner_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method=all".format(stellar_mass_fraction,
                                                                                         grav_stellars[0].inclination, grav_model) + ".png")
    plt.cla()

    for j in range(len(computer_times)):
        plt.plot(computer_times[j], semimajor_axis_outs[j], label="outer, {}".format(schemes[j]))

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Semimajor Axis/(Initial Semimajor Axis)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("semimajor_axis_outer_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method=all".format(stellar_mass_fraction,
                                                                                               grav_stellars[0].inclination, grav_model) + ".png")
    plt.cla()

    # Now plot the orbital params as function of timestep
    for j in range(len(computer_times)):
        plt.plot(computer_times[j], eccentricity_ins[j], label="inner, {}".format(schemes[j]))
        plt.plot(computer_times[j], eccentricity_outs[j], label="outer, {}".format(schemes[j]))

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Eccentricity/(Initial Eccentricity)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("ALL_eccentricity_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method=all".format(stellar_mass_fraction,
                                                                               grav_stellars[0].inclination, grav_model) + ".png")
    plt.cla()

    for j in range(len(computer_times)):
        plt.plot(computer_times[j], eccentricity_ins[j], label="inner, {}".format(schemes[j]))

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Eccentricity/(Initial Eccentricity)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("eccentricity_inner_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method=all".format(stellar_mass_fraction,
                                                                                       grav_stellars[0].inclination, grav_model) + ".png")
    plt.cla()

    for j in range(len(computer_times)):
        plt.plot(computer_times[j], eccentricity_outs[j], label="outer, {}".format(schemes[j]))

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Eccentricity/(Initial Eccentricity)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("eccentricity_outer_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method=all".format(stellar_mass_fraction,
                                                                                             grav_stellars[0].inclination, grav_model) + ".png")
    plt.cla()


    for j in range(len(computer_times)):
        plt.plot(computer_times[j], inclinations[j], label="{}".format(schemes[j]))

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Inclination/(Initial Inclination)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("ALL_inclination_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method=all".format(stellar_mass_fraction,
                                                                                             grav_stellars[0].inclination, grav_model) + ".png")
    plt.cla()


def get_orbital_period(orbital_separation, total_mass):
    return 2 * np.pi * (orbital_separation ** 3 / (constants.G * total_mass)).sqrt()


def get_semi_major_axis(orbital_period, total_mass):
    return (constants.G * total_mass * orbital_period ** 2 / (4 * np.pi ** 2)) ** (1. / 3)


def generate_initial_muscea(integration_method='interlaced', stellar_mass_loss_fraction=0.000001,
                            inclination=60, gravity_model=Huayno):
    # Initial Conditions
    eccentricity_init = 0.2
    eccentricity_out_init = 0.6
    semimajor_axis_out_init = 100 | units.AU
    mean_anomaly = 180
    argument_of_perigee = 180
    longitude_of_the_ascending_node = 0

    M1 = 60 | units.MSun
    M2 = 30 | units.MSun
    M3 = 20 | units.MSun
    period_init = 19 | units.day
    semimajor_axis_init = 0.63 | units.AU

    period_or_semimajor = 1  # 1: Get semimajor_axis from period, Otherwise: get period from semimjaor_axis

    stellar_start_time = 4.0 | units.Myr
    end_time = 0.001 | units.Myr
    triple = Particles(3)
    triple[0].mass = M1
    triple[1].mass = M2
    triple[2].mass = M3

    grav_stellar = GravitationalStellar(stellar_mass_loss_timestep_fraction=stellar_mass_loss_fraction,
                                        gravity_model=gravity_model,
                                        integration_scheme=integration_method, inclination=inclination)
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

    # Inner binary

    r, v = get_position(triple[0].mass, triple[1].mass, eccentricity_init, semimajor_axis_init, mean_anomaly,
                        inclination,
                        argument_of_perigee, longitude_of_the_ascending_node, delta_time)
    tmp_stars[1].position = r
    tmp_stars[1].velocity = v
    tmp_stars.move_to_center()

    # Outer binary

    r, v = get_position(triple[0].mass + triple[1].mass, triple[2].mass, eccentricity_out_init, semimajor_axis_out_init,
                        0,
                        0, 0, 0, delta_time)
    tertiary = Particle()
    tertiary.mass = triple[2].mass
    tertiary.position = r
    tertiary.velocity = v
    tmp_stars.add_particle(tertiary)
    tmp_stars.move_to_center()

    triple.position = tmp_stars.position
    triple.velocity = tmp_stars.velocity

    grav_stellar.set_initial_parameters(semimajor_axis_init, eccentricity_init,
                                        semimajor_axis_out_init, eccentricity_out_init, period_init)

    grav_stellar.set_gravity(semimajor_axis_out_init)

    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, inclination_history = grav_stellar.evolve_model(end_time)
    print("Number of Timesteps: ", len(timestep_history))
    plot_results(timestep_history, eccentricity_out_history, eccentricity_in_history, semimajor_axis_in_history,
                 semimajor_axis_out_history, stellar_mass_loss_fraction, grav_stellar)

    return timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
           semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history


def run_simulation(gravity_model=Huayno, stellar_mass_loss_fraction=0.001):
    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                      stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                      inclination=90, gravity_model=gravity_model)

    stimestep_history, smass_history, ssemimajor_axis_in_history, seccentricity_in_history, \
    ssemimajor_axis_out_history, seccentricity_out_history, sgrav_stellar, sinclination_history = generate_initial_muscea("stellar_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=90, gravity_model=gravity_model)

    gtimestep_history, gmass_history, gsemimajor_axis_in_history, geccentricity_in_history, \
    gsemimajor_axis_out_history, geccentricity_out_history, ggrav_stellar, ginclination_history = generate_initial_muscea("gravity_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=90, gravity_model=gravity_model)

    dtimestep_history, dmass_history, dsemimajor_axis_in_history, deccentricity_in_history, \
    dsemimajor_axis_out_history, deccentricity_out_history, dgrav_stellar, dinclination_history = generate_initial_muscea("diagnostic",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=90, gravity_model=gravity_model)

    plot_all_results([timestep_history, stimestep_history, gtimestep_history, dtimestep_history],
                     [eccentricity_out_history, seccentricity_out_history, geccentricity_out_history,
                      deccentricity_out_history],
                     [eccentricity_in_history, seccentricity_in_history, geccentricity_in_history,
                      deccentricity_in_history],
                     [semimajor_axis_in_history, ssemimajor_axis_in_history, gsemimajor_axis_in_history,
                      dsemimajor_axis_in_history],
                     [semimajor_axis_out_history, ssemimajor_axis_out_history, gsemimajor_axis_out_history,
                      dsemimajor_axis_out_history],
                     stellar_mass_loss_fraction,
                     [grav_stellar, sgrav_stellar, ggrav_stellar, dgrav_stellar],
                     [inclination_history, sinclination_history, ginclination_history, dinclination_history])

    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                      stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                      inclination=60, gravity_model=gravity_model)

    stimestep_history, smass_history, ssemimajor_axis_in_history, seccentricity_in_history, \
    ssemimajor_axis_out_history, seccentricity_out_history, sgrav_stellar, sinclination_history = generate_initial_muscea("stellar_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=60, gravity_model=gravity_model)

    gtimestep_history, gmass_history, gsemimajor_axis_in_history, geccentricity_in_history, \
    gsemimajor_axis_out_history, geccentricity_out_history, ggrav_stellar, ginclination_history = generate_initial_muscea("gravity_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=60, gravity_model=gravity_model)

    dtimestep_history, dmass_history, dsemimajor_axis_in_history, deccentricity_in_history, \
    dsemimajor_axis_out_history, deccentricity_out_history, dgrav_stellar, dinclination_history = generate_initial_muscea("diagnostic",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=60, gravity_model=gravity_model)

    plot_all_results([timestep_history, stimestep_history, gtimestep_history, dtimestep_history],
                     [eccentricity_out_history, seccentricity_out_history, geccentricity_out_history,
                      deccentricity_out_history],
                     [eccentricity_in_history, seccentricity_in_history, geccentricity_in_history,
                      deccentricity_in_history],
                     [semimajor_axis_in_history, ssemimajor_axis_in_history, gsemimajor_axis_in_history,
                      dsemimajor_axis_in_history],
                     [semimajor_axis_out_history, ssemimajor_axis_out_history, gsemimajor_axis_out_history,
                      dsemimajor_axis_out_history],
                     stellar_mass_loss_fraction,
                     [grav_stellar, sgrav_stellar, ggrav_stellar, dgrav_stellar],
                     [inclination_history, sinclination_history, ginclination_history, dinclination_history])
    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                      stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                      inclination=30, gravity_model=gravity_model)

    stimestep_history, smass_history, ssemimajor_axis_in_history, seccentricity_in_history, \
    ssemimajor_axis_out_history, seccentricity_out_history, sgrav_stellar, sinclination_history = generate_initial_muscea("stellar_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=30, gravity_model=gravity_model)

    gtimestep_history, gmass_history, gsemimajor_axis_in_history, geccentricity_in_history, \
    gsemimajor_axis_out_history, geccentricity_out_history, ggrav_stellar, ginclination_history = generate_initial_muscea("gravity_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=30, gravity_model=gravity_model)

    dtimestep_history, dmass_history, dsemimajor_axis_in_history, deccentricity_in_history, \
    dsemimajor_axis_out_history, deccentricity_out_history, dgrav_stellar, dinclination_history = generate_initial_muscea("diagnostic",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=30, gravity_model=gravity_model)

    plot_all_results([timestep_history, stimestep_history, gtimestep_history, dtimestep_history],
                     [eccentricity_out_history, seccentricity_out_history, geccentricity_out_history,
                      deccentricity_out_history],
                     [eccentricity_in_history, seccentricity_in_history, geccentricity_in_history,
                      deccentricity_in_history],
                     [semimajor_axis_in_history, ssemimajor_axis_in_history, gsemimajor_axis_in_history,
                      dsemimajor_axis_in_history],
                     [semimajor_axis_out_history, ssemimajor_axis_out_history, gsemimajor_axis_out_history,
                      dsemimajor_axis_out_history],
                     stellar_mass_loss_fraction,
                     [grav_stellar, sgrav_stellar, ggrav_stellar, dgrav_stellar],
                     [inclination_history, sinclination_history, ginclination_history, dinclination_history])

    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                      stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                      inclination=1, gravity_model=gravity_model)

    stimestep_history, smass_history, ssemimajor_axis_in_history, seccentricity_in_history, \
    ssemimajor_axis_out_history, seccentricity_out_history, sgrav_stellar, sinclination_history = generate_initial_muscea("stellar_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=1, gravity_model=gravity_model)

    gtimestep_history, gmass_history, gsemimajor_axis_in_history, geccentricity_in_history, \
    gsemimajor_axis_out_history, geccentricity_out_history, ggrav_stellar, ginclination_history = generate_initial_muscea("gravity_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=1, gravity_model=gravity_model)

    dtimestep_history, dmass_history, dsemimajor_axis_in_history, deccentricity_in_history, \
    dsemimajor_axis_out_history, deccentricity_out_history, dgrav_stellar, dinclination_history = generate_initial_muscea("diagnostic",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=1, gravity_model=gravity_model)

    plot_all_results([timestep_history, stimestep_history, gtimestep_history, dtimestep_history],
                     [eccentricity_out_history, seccentricity_out_history, geccentricity_out_history,
                      deccentricity_out_history],
                     [eccentricity_in_history, seccentricity_in_history, geccentricity_in_history,
                      deccentricity_in_history],
                     [semimajor_axis_in_history, ssemimajor_axis_in_history, gsemimajor_axis_in_history,
                      dsemimajor_axis_in_history],
                     [semimajor_axis_out_history, ssemimajor_axis_out_history, gsemimajor_axis_out_history,
                      dsemimajor_axis_out_history],
                     stellar_mass_loss_fraction,
                     [grav_stellar, sgrav_stellar, ggrav_stellar, dgrav_stellar],
                     [inclination_history, sinclination_history, ginclination_history, dinclination_history])

    return grav_stellar.elapsed_total_time, grav_stellar.elapsed_sim_time

def get_timestep_walltime_differences(gravity_model):
    timesteps = np.linspace(0.1, 0.0000001, 25)
    wall_times = []
    sim_times = []
    amuse_times = []
    grav_models = ['hermite', 'huayno', 'smalln']
    if gravity_model == Hermite:
        grav_model = grav_models[0]
    elif gravity_model == Huayno:
        grav_model = grav_models[1]
    else:
        grav_model = grav_models[2]
    for timestep in timesteps:
        timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
        semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                          stellar_mass_loss_fraction=timestep,
                                                                                                                          inclination=1, gravity_model=gravity_model)
        wall_times.append(grav_stellar.elapsed_total_time)
        sim_times.append(grav_stellar.elapsed_sim_time)
        amuse_times.append(grav_stellar.elapsed_total_time - grav_stellar.elapsed_sim_time)

    plt.cla()
    plt.plot(timesteps, wall_times, label='Wall')
    plt.plot(timesteps, sim_times, label='Sim')
    plt.plot(timesteps, amuse_times, label='AMUSE')
    plt.title("Timestep vs Wall Time")
    plt.xlabel("Timestep (Stellar Mass Loss Fraction)")
    plt.ylabel("Wall Time (s)")
    plt.legend(loc='best')
    plt.savefig("timestep_vs_walltime_grav={}".format(grav_model) + ".png")
    plt.cla()


if __name__ in ('__main__', '__plot__'):
    gravity_model = Huayno
    do_timestep = True
    if do_timestep:
        get_timestep_walltime_differences(gravity_model)

    stellar_mass_loss_fractions = [0.001, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
    for fraction in stellar_mass_loss_fractions:
        run_simulation(gravity_model, fraction)