from amuse.community.ph4.interface import ph4
from amuse.community.bhtree.interface import BHTree
from amuse.units import nbody_system
from amuse.ext.bridge import bridge
from amuse.units import units
from amuse.datamodel import Particle, Particles
from amuse.community.huayno.interface import Huayno
from amuse.ic.salpeter import new_powerlaw_mass_distribution

import time


class HybridGravity(object):

    def __init__(self, direct_code=ph4, tree_code=BHTree, mass_cut=6. | units.MSun, timestep=0.01):
        self.direct_code = direct_code()
        self.tree_code = tree_code()
        self.mass_cut = mass_cut

        self.combined_gravity = bridge()
        self.combined_gravity.timestep = timestep

        self.combined_gravity.add_system(self.direct_code, (self.tree_code,))
        self.combined_gravity.add_system(self.tree_code, (self.direct_code,))

        self.channel_from_direct = None
        self.channel_from_particles_to_direct = None
        self.channel_from_tree = None
        self.channel_from_particles_to_tree = None

        self.direct_particles = Particles()
        self.tree_particles = Particles()

        self.timestep_history = []
        self.energy_history = []
        self.half_mass_history = []
        self.core_radii_history = []
        self.mass_history = []

        self.elapsed_time = None

    def add_particles(self, particles):
        """
        Adds particles, splitting them up based on the mass_cut set
        :param particles:
        :return:
        """
        for particle in particles:
            if particles.mass >= self.mass_cut:
                self.direct_particles.add_particle(particle)
            else:
                self.tree_particles.add_particle(particle)

        self.add_particles_to_direct(self.direct_particles)
        self.add_particles_to_tree(self.tree_particles)

    def get_total_energy(self):
        """
        Returns the combined energy of the tree and direct code
        :return:
        """

        return self.direct_code.get_total_energy() + self.tree_code.get_total_energy()

    def get_total_radii(self):
        """
        Returns the total radii for both direct and tree
        :return:
        """
        return self.direct_code.get_total_radius(), self.tree_code.get_total_radius()

    def get_total_mass(self):
        """
        Returns the total mass of the system
        :return:
        """

        return self.direct_code.get_total_mass() + self.tree_code.get_total_mass()

    def evolve_model(self, end_time, timestep_length=0.1 | units.Myr):
        """
        Evolves the system until the end time

        :param end_tme:
        :param number_of_steps:
        :return:
        """

        start_time = time.time()

        sim_time = 0.0 | end_time.unit

        total_initial_energy = self.get_total_energy()
        total_previous_energy = total_initial_energy

        total_particle_mass = self.get_total_mass()

        self.timestep_history.append(sim_time.value_in(units.Myr))
        self.mass_history.append(self.get_total_mass() / self.get_total_mass())
        self.energy_history.append(self.get_total_energy() / self.get_total_energy())
        # TODO Add the core radii and half-mass here

        while sim_time < end_time:
            sim_time += timestep_length

            self.combined_gravity.evolve_model(timestep_length)

            new_energy = self.combined_gravity.potential_energy + self.combined_gravity.kinetic_energy

            self.energy_history.append(new_energy / total_initial_energy)
            self.channel_from_direct.copy()
            self.channel_from_tree.copy()

        self.direct_code.stop()
        self.tree_code.stop()

        end_time = time.time()

        self.elapsed_time += end_time - start_time

        return self.timestep_history, self.mass_history, self.energy_history, self.half_mass_history, self.core_radii_history

    def add_particles_to_direct(self, particles):
        """
        Adds particles to the direct Nbody code
        :param particles:
        :return:
        """
        self.direct_code.add_particles(particles)
        self.channel_from_direct = self.direct_code.particles.new_channel_to(particles)
        self.channel_from_particles_to_direct = self.direct_particles.new_channel_to(self.direct_code.particles)

    def add_particles_to_tree(self, particles):
        """
        Adds particles to the tree NBody code
        :param particles:
        :return:
        """
        self.tree_code.add_particles(particles)
        self.channel_from_tree = self.tree_code.particles.new_channel_to(particles)
        self.channel_from_particles_to_tree = self.tree_particles.new_channel_to(self.tree_code.particles)
