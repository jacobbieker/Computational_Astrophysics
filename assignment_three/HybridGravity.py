from amuse.community.ph4.interface import ph4
from amuse.community.bhtree.interface import BHTree
from amuse.units import nbody_system
from amuse.ext.bridge import bridge
from amuse.units import units
from amuse.datamodel import Particle, Particles
from amuse.community.huayno.interface import Huayno
from amuse.community.hermite0.interface import Hermite
from amuse.community.smalln.interface import SmallN
from amuse.ic.salpeter import new_powerlaw_mass_distribution

import time


class HybridGravity(object):

    def __init__(self, direct_code=ph4, tree_code=BHTree, mass_cut=6. | units.MSun, timestep=0.01, flip_split=False, convert_nbody=None):
        """
        This is the initialization for the HybridGravity solver. For the most flexibility, as well as to allow this one
        class to fulfill the requirements of the assignment, it is able to be run with a single gravity solver, or two gravity solvers
        with the particles split into two different populations by the mass cut.

        While not a complete copy of the interfaces for other GravitationalDynamics codes, it is designed to be similar,
        with the important parts being only needing to call add_particles and evolve_model to run the simulation.

        By default, all particles larger than the mass cut are sent to the direct_code for integration, and all those less
        than the mass cut are sent to the tree_code. This is changed by flip_split to the other order.


        :param direct_code: AMUSE Interface to a Nbody solver, such as ph4, Huayno, Hermite, or SmallN, ideally a direct Nbody solver,
        Can also be a string, in which case "ph4", "huayno", "hermite", or "smalln" are the acceptable options
        :param tree_code: AMUSE Interface to a NBody solver, such as BHTree, or others, ideally a tree-code Nbody solver,
        Can also be a string, in which case "bhtree" is the acceptable option
        :param mass_cut: The mass cutoff. Those particles larger than the mass cutoff are sent, by default, to the direct_code
        those below it are sent to the tree_code
        :param timestep: The timestep for the system
        :param flip_split: Whether to flip how the split works, so that the stars more massive than the mass cutoff are sent to
        the tree_code instead of the direct_code. Defaults to False.
        :param convert_nbody: The converter to use if needed to convert the nbody units to physical units, defaults to None

        """

        if direct_code is not None:
            if isinstance(direct_code, str):
                if direct_code.lower() == "smalln":
                    self.direct_code = SmallN(convert_nbody)
                elif direct_code.lower() == "huayno":
                    self.direct_code = Huayno(convert_nbody)
                elif direct_code.lower() == "hermite":
                    self.direct_code = Hermite(convert_nbody)
                elif direct_code.lower() == "ph4":
                    self.direct_code = ph4(convert_nbody)
                else:
                    raise NotImplementedError
            else:
                self.direct_code = direct_code(convert_nbody)
        else:
            self.direct_code = None

        if tree_code is not None:
            if isinstance(tree_code, str):
                if tree_code.lower() == "bhtree":
                    self.tree_code = BHTree(convert_nbody)
                else:
                    raise NotImplementedError
            else:
                self.tree_code = tree_code(convert_nbody)
        else:
            self.tree_code = None

        self.mass_cut = mass_cut

        self.flip_split = flip_split
        # Whether to flip the split so that particles more massive than mass_cut go to the tree code instead of direct
        if self.tree_code is None:
            self.combined_gravity = self.direct_code
        elif self.direct_code is None:
            self.combined_gravity = self.tree_code
        else:
            # So use both gravities
            # Create the bridge for the two gravities
            self.combined_gravity = None
            #self.combined_gravity.timestep = timestep

            #self.combined_gravity.add_system(self.direct_code, (self.tree_code,))
            #self.combined_gravity.add_system(self.tree_code, (self.direct_code,))

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

        self.elapsed_time = 0.0
        self.timestep = timestep

    def _create_bridge(self):
        # So use both gravities
        # Create the bridge for the two gravities
        self.combined_gravity = bridge()
        self.combined_gravity.timestep = self.timestep

        self.combined_gravity.add_system(self.direct_code, (self.tree_code,))
        self.combined_gravity.add_system(self.tree_code, (self.direct_code,))

    def add_particles(self, particles):
        """
        Adds particles, splitting them up based on the mass_cut set
        :param particles: The Particles() object containing the particles to add
        """

        if self.direct_code is None:
            self.tree_particles.add_particles(particles)

        elif self.tree_code is None:
            self.direct_particles.add_particles(particles)
        else:
            # Need to split based on the mass cut
            for particle in particles:
                if particle.mass >= self.mass_cut:
                    if self.flip_split:
                        self.tree_particles.add_particle(particle)
                    else:
                        self.direct_particles.add_particle(particle)
                else:
                    if self.flip_split:
                        self.direct_particles.add_particle(particle)
                    else:
                        self.tree_particles.add_particle(particle)

        if self.direct_code is None:
            self.add_particles_to_tree(self.tree_particles)
        elif self.tree_code is None:
            self.add_particles_to_direct(self.direct_particles)
        else:
            self.add_particles_to_direct(self.direct_particles)
            self.add_particles_to_tree(self.tree_particles)
            # Now create the bridge, since both codes used
            self._create_bridge()

    def get_total_energy(self):
        """
        Returns the combined energy of the tree and direct code
        :return: Returns the total energy of the system
        """
        if self.direct_code is None:
            return self.tree_code.get_total_energy()
        elif self.tree_code is None:
            return self.direct_code.get_total_energy()
        else:
            return self.direct_code.get_total_energy() + self.tree_code.get_total_energy()

    def get_total_radii(self):
        """
        Returns the total radii for both direct and tree
        :return: Returns the total radii of the system
        """
        return self.direct_code.get_total_radius(), self.tree_code.get_total_radius()

    def get_total_mass(self):
        """
        Returns the total mass of the system
        :return: The total mass of the system
        """

        if self.direct_code is None:
            return self.tree_code.get_total_mass()
        elif self.tree_code is None:
            return self.direct_code.get_total_mass()
        else:
            return self.direct_code.get_total_mass() + self.tree_code.get_total_mass()

    def evolve_model(self, end_time, timestep_length=0.1):
        """
        Evolves the system until the end time, saving out information at set time intervals

        :param end_tme: The end time of the simulation, with AMUSE units
        :param number_of_steps: Number of steps to run for
        :return: Timestep history, mass history, energy history, half-mass history, and core-radii history
        all relative to the initial conditions
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
            sim_time += timestep_length | end_time.unit

            self.combined_gravity.evolve_model(timestep_length)
            if self.channel_from_direct is not None:
                self.channel_from_direct.copy()
            if self.channel_from_tree is not None:
                self.channel_from_tree.copy()

            new_energy = self.combined_gravity.potential_energy() + self.combined_gravity.kinetic_energy()

            self.energy_history.append(new_energy / total_initial_energy)

        if self.direct_code is not None:
            self.direct_code.stop()
        if self.tree_code is not None:
            self.tree_code.stop()

        end_time = time.time()

        self.elapsed_time += end_time - start_time

        return self.timestep_history, self.mass_history, self.energy_history, self.half_mass_history, self.core_radii_history

    def add_particles_to_direct(self, particles):
        """
        Adds particles to the direct Nbody code
        :param particles: A Particles() object containing the particles to add
        """
        self.direct_code.particles.add_particles(particles)
        self.channel_from_direct = self.direct_code.particles.new_channel_to(particles)
        self.channel_from_particles_to_direct = self.direct_particles.new_channel_to(self.direct_code.particles)

    def add_particles_to_tree(self, particles):
        """
        Adds particles to the tree NBody code
        :param particles: A Particles() object containing the particles to add
        """
        self.tree_code.particles.add_particles(particles)
        self.channel_from_tree = self.tree_code.particles.new_channel_to(particles)
        self.channel_from_particles_to_tree = self.tree_particles.new_channel_to(self.tree_code.particles)
