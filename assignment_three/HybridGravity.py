from amuse.ext.bridge import bridge
from amuse.units import units
from amuse.datamodel import Particles
from amuse.units import nbody_system
from amuse.community.huayno.interface import Huayno
from amuse.community.hermite0.interface import Hermite
from amuse.community.smalln.interface import SmallN
from amuse.community.ph4.interface import ph4
from amuse.community.bhtree.interface import BHTree
from amuse.community.octgrav.interface import Octgrav
from amuse.community.bonsai.interface import Bonsai
from amuse.io import write_set_to_file
from amuse.io import read_set_from_file

import pickle
import time


class HybridGravity(object):

    def __init__(self, direct_code=ph4, tree_code=BHTree, mass_cut=6. | units.MSun, timestep=0.1, flip_split=False,
                 convert_nbody=None, number_of_workers=1, tree_converter=None, direct_converter=None, input_args=None):
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
        Can also be a string, in which case "bhtree", "octgrav", and "bonsai" are the acceptable options
        :param mass_cut: The mass cutoff. Those particles larger than the mass cutoff are sent, by default, to the direct_code
        those below it are sent to the tree_code
        :param timestep: The timestep for the system
        :param flip_split: Whether to flip how the split works, so that the stars more massive than the mass cutoff are sent to
        the tree_code instead of the direct_code. Defaults to False.
        :param convert_nbody: The converter to use if needed to convert the nbody units to physical units, defaults to None

        """

        self.input_args = input_args

        self.converter = convert_nbody
        if tree_converter is not None:
            self.tree_converter = tree_converter
        else:
            self.tree_converter = convert_nbody
        if direct_converter is not None:
            self.direct_converter = direct_converter
        else:
            self.direct_converter = convert_nbody

        if direct_code is not None:
            if isinstance(direct_code, str):
                if direct_code.lower() == "smalln":
                    self.direct_code = SmallN(direct_converter, number_of_workers=number_of_workers)
                elif direct_code.lower() == "huayno":
                    self.direct_code = Huayno(direct_converter, number_of_workers=number_of_workers)
                elif direct_code.lower() == "hermite":
                    self.direct_code = Hermite(direct_converter, number_of_workers=number_of_workers)
                elif direct_code.lower() == "ph4":
                    self.direct_code = ph4(direct_converter, number_of_workers=number_of_workers)
                else:
                    raise NotImplementedError
            else:
                self.direct_code = direct_code(direct_converter, number_of_workers=number_of_workers)
        else:
            self.direct_code = None

        if tree_code is not None:
            if isinstance(tree_code, str):
                if tree_code.lower() == "bhtree":
                    self.tree_code = BHTree(tree_converter, number_of_workers=number_of_workers)
                elif tree_code.lower() == "bonsai":
                    self.tree_code = Bonsai(tree_converter, number_of_workers=number_of_workers)
                elif tree_code.lower() == "octgrav":
                    self.tree_code = Octgrav(tree_converter, number_of_workers=number_of_workers)
                else:
                    raise NotImplementedError
            else:
                self.tree_code = tree_code(tree_converter, number_of_workers=number_of_workers)
        else:
            self.tree_code = None

        if self.tree_code is None:
            self.combined_gravity = self.direct_code
        elif self.direct_code is None:
            self.combined_gravity = self.tree_code
        else:
            # So use both gravities
            self.combined_gravity = None

        self.mass_cut = mass_cut
        self.num_workers = number_of_workers

        self.flip_split = flip_split

        self.channel_from_direct = None
        self.channel_from_particles_to_direct = None
        self.channel_from_tree = None
        self.channel_from_particles_to_tree = None

        self.direct_particles = Particles()
        self.tree_particles = Particles()
        self.timestep = timestep

        # Whether to flip the split so that particles more massive than mass_cut go to the tree code instead of direct

        self.timestep_history = []
        self.energy_ratio_history = []
        self.half_mass_ratio_history = []
        self.core_radius_ratio_history = []
        self.mass_history = []

        self.energy_history = []
        self.half_mass_history = []
        self.core_radii_history = []

        self.particle_masses = []
        self.tree_locations = []
        self.combined_locations = []

        self.elapsed_time = 0.0

    def _create_bridge(self):
        # So use both gravities
        # Create the bridge for the two gravities
        self.combined_gravity = bridge()
        self.combined_gravity.add_system(self.tree_code, (self.direct_code,))
        self.combined_gravity.add_system(self.direct_code, (self.tree_code,))

    def get_core_radius(self):
        """
        Returns the core-radius for the system
        :return:
        """
        _, core_radius, _ = self.combined_gravity.particles.densitycentre_coreradius_coredens(
            unit_converter=self.converter)
        return core_radius

    def get_half_mass(self):
        """
        Returns the half-mass distance for the system
        :return:
        """
        total_radius = \
        self.combined_gravity.particles.LagrangianRadii(mf=[0.5], cm=self.combined_gravity.particles.center_of_mass(),
                                                        unit_converter=self.converter)[0][0]
        return total_radius

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

            # Now here, if the converters exist, should scale to them too to the correct standard
            #if self.direct_converter is not None and self.tree_converter is not None:
            #    self.direct_particles.scale_to_standard(convert_nbody=self.direct_converter)
            #    self.tree_particles.scale_to_standard(convert_nbody=self.tree_converter)

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
        return self.combined_gravity.potential_energy + self.combined_gravity.kinetic_energy

    def get_total_mass(self):
        """
        Returns the total mass of the system
        :return: The total mass of the system
        """

        return self.combined_gravity.particles.mass.sum()

    def evolve_model(self, end_time, timestep_length=0.1 | units.Myr):
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
        total_particle_mass = self.get_total_mass()
        initial_core_radii = self.get_core_radius()
        initial_half_mass = self.get_half_mass()

        self.timestep_history.append(sim_time.value_in(units.Myr))
        self.mass_history.append(self.get_total_mass() / self.get_total_mass())
        self.energy_ratio_history.append(self.get_total_energy() / self.get_total_energy())
        self.half_mass_ratio_history.append(self.get_half_mass() / self.get_half_mass())
        self.core_radius_ratio_history.append(self.get_core_radius() / self.get_core_radius())

        self.core_radii_history.append(self.get_core_radius())
        self.half_mass_history.append(self.get_half_mass())
        self.energy_history.append(self.get_total_energy())

        self.combined_locations.append((self.combined_gravity.particles.x.value_in(units.parsec), self.combined_gravity.particles.y.value_in(units.parsec), self.combined_gravity.particles.z.value_in(units.parsec)))
        self.particle_masses.append(self.combined_gravity.particles.mass.value_in(units.MSun))

        self.combined_gravity.timestep = self.timestep | units.Myr

        timestep_check = end_time / 10.

        while sim_time < end_time:
            sim_time += timestep_length

            if self.channel_from_direct is not None:
                self.channel_from_direct.copy()
            if self.channel_from_tree is not None:
                self.channel_from_tree.copy()

            self.combined_gravity.evolve_model(sim_time, timestep=timestep_length)

            new_energy = self.get_total_energy()

            self.energy_ratio_history.append((new_energy) / total_initial_energy)
            self.half_mass_ratio_history.append(self.get_half_mass() / initial_half_mass)
            self.core_radius_ratio_history.append(self.get_core_radius() / initial_core_radii)
            self.mass_history.append(self.get_total_mass() / total_particle_mass)
            self.timestep_history.append(sim_time.value_in(units.Myr))
            self.core_radii_history.append(self.get_core_radius())
            self.half_mass_history.append(self.get_half_mass())
            self.energy_history.append(self.get_total_energy())
            self.combined_locations.append((self.combined_gravity.particles.x.value_in(units.parsec), self.combined_gravity.particles.y.value_in(units.parsec), self.combined_gravity.particles.z.value_in(units.parsec)))
            self.particle_masses.append(self.combined_gravity.particles.mass.value_in(units.MSun))

            end_wall_time = time.time()

            self.elapsed_time = end_wall_time - start_time
            # Save model history as a checkpoint every tenth of the total simulation
            if sim_time >= timestep_check:
                if self.input_args is None:
                    write_set_to_file(self.combined_gravity.particles, "Sim_N_{}_MC_{}_W_{}.hdf".format(len(self.combined_gravity.particles), self.mass_cut, self.num_workers), "amuse")
                else:
                    write_set_to_file(self.combined_gravity.particles, "Sim_N_{}_MC_{}_W_{}_DC_{}_TC_{}.hdf".format(len(self.combined_gravity.particles), self.mass_cut.value_in(units.MSun), self.num_workers, self.input_args['direct_code'], self.input_args['tree_code']), "amuse")
                self.save_model_history("Sim_N_{}_MC_{}_W_{}.p".format(len(self.combined_gravity.particles), self.mass_cut, self.num_workers), input_dict=self.input_args)
                timestep_check += end_time / 10.

        if self.direct_code is not None:
            self.direct_code.stop()
        if self.tree_code is not None:
            self.tree_code.stop()

        return self.timestep_history, self.mass_history, self.energy_ratio_history, self.half_mass_ratio_history, self.core_radius_ratio_history

    def return_model_history(self):
        """
        Returns the output of the model as a dictionary so that it can be returned or analyzed later
        """

        model_information = {"timestep_history": self.timestep_history,
                             "energy_history": self.energy_history,
                             "half_mass_history": self.half_mass_history,
                             "core_radius_history": self.core_radii_history,
                             "mass_cut": self.mass_cut,
                             "flip_split": self.flip_split,
                             "timestep": self.timestep,
                             "wall_time": self.elapsed_time,
                             "num_direct": len(self.direct_particles),
                             "num_tree": len(self.tree_particles),
                             "particle_history": self.particle_masses,
                             "combined_particles_locations": self.combined_locations,
                            }

        return model_information

    def save_model_history(self, output_file, input_dict=None):
        """
        Saves out the model history and input options
        :param output_file:
        :param input_dict:
        """

        if input_dict is not None:
            model_dict = [input_dict, self.return_model_history()]
        else:
            model_dict = self.return_model_history()

        with open(output_file, "wb") as pickle_file:
            pickle.dump(model_dict, pickle_file)


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
