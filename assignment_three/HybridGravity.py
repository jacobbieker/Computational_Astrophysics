from amuse.community.ph4.interface import ph4
from amuse.community.bhtree.interface import BHTree
from amuse.ic.plummer import new_plummer_model
from amuse.units import nbody_system
from amuse.ext.bridge import bridge
from amuse.units import units
from amuse.datamodel import Particle, Particles


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
        self.channel_from_tree = None

        self.direct_particles = Particles()
        self.tree_particles = Particles()

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

    def evolve_model(self, end_tme, number_of_steps):
        """
        Evolves the system until the end time

        :param end_tme:
        :param number_of_steps:
        :return:
        """



        return NotImplementedError

    def add_particles_to_direct(self, particles):
        """
        Adds particles to the direct Nbody code
        :param particles:
        :return:
        """
        self.direct_code.add_particles(particles)
        self.channel_from_direct = self.direct_code.particles.new_channel_to(particles)

    def add_particles_to_tree(self, particles):
        """
        Adds particles to the tree NBody code
        :param particles:
        :return:
        """
        self.tree_code.add_particles(particles)
        self.channel_from_tree = self.tree_code.particles.new_channel_to(particles)
