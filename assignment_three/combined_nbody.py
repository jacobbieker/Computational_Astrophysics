import numpy as np
from amuse.units import nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.units import units
from HybridGravity import HybridGravity
import argparse

import matplotlib.pyplot as plt


def make_converters(input_args, input_particles, all_converter):
    """
    Creates the tree and direct converters from the total set of particles
    :param input_args: Input arguments from argparse
    :param input_particles: Input set of particles to split
    :param all_converter: The conveter for the whole set of particles
    :return: tree_converter and direct_converter
    """
    # Now get the masses in each one for the different converters
    direct_particles = Particles()
    tree_particles = Particles()
    tree_converter = None
    direct_converter = None
    if input_args['tree_code'] is not None and input_args['direct_code'] is not None:
        if input_args['method'] == 'mass':
            for particle in input_particles:
                if particle.mass >= input_args['mass_cut'] | units.MSun:
                    if input_args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if input_args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        elif input_args['method'] == 'core_radius':
            _, core_radius, _ = input_particles.densitycentre_coreradius_coredens(
                unit_converter=all_converter)
            for particle in input_particles:
                if np.sqrt((particle.x - input_particles.center_of_mass().x) ** 2 + (
                        particle.y - input_particles.center_of_mass().y) ** 2 + (
                                   particle.z - input_particles.center_of_mass().z) ** 2) <= input_args['mass_cut']\
                        * core_radius:
                    if input_args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if input_args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        elif input_args['method'] == 'half_mass':
            half_mass_radius = \
                input_particles.LagrangianRadii(mf=[0.5], cm=input_particles.center_of_mass(),
                                                unit_converter=all_converter)[0][0]
            for particle in input_particles:
                if np.sqrt((particle.x - input_particles.center_of_mass().x) ** 2 + (
                        particle.y - input_particles.center_of_mass().y) ** 2 + (
                                   particle.z - input_particles.center_of_mass().z) ** 2) <= input_args['mass_cut']\
                        * half_mass_radius:
                    if input_args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if input_args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        elif input_args['method'] == 'virial_radius':
            virial_radius = input_particles.virial_radius()
            for particle in input_particles:
                if np.sqrt((particle.x - input_particles.center_of_mass().x) ** 2 + (
                        particle.y - input_particles.center_of_mass().y) ** 2 + (
                                   particle.z - input_particles.center_of_mass().z) ** 2) <= input_args['mass_cut']\
                        * virial_radius:
                    if input_args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if input_args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        if input_args['tree_code'] is not None:
            tree_converter = nbody_system.nbody_to_si(tree_particles.mass.sum(), tree_particles.virial_radius())
        if input_args['direct_code'] is not None:
            direct_converter = nbody_system.nbody_to_si(direct_particles.mass.sum(), direct_particles.virial_radius())
    else:
        # No splitting, so all the converters are the same
        tree_converter = all_converter
        direct_converter = all_converter

    return tree_converter, direct_converter


def get_args():
    """
    Obtains and returns a dictionary of the command line arguments for this program

    :return: Command line arguments and their values in a dictionary
    """

    def str2bool(v):
        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_bodies", required=False, default=10000, type=int, help="Number of bodies to simulate")
    ap.add_argument("-dc", "--direct_code", required=False, default="ph4", type=str,
                    help="Direct Code Integrator (ph4, Huayno, Hermite, or SmallN) or None")
    ap.add_argument("-tc", '--tree_code', required=False, default='bhtree', type=str,
                    help="Tree Code Integrator (BHTree, Bonsai, Octgrav etc.) or None")
    ap.add_argument("-mc", '--mass_cut', required=False, default=2., type=float,
                    help="Mass Cutoff for splitting bodies, in units MSun (default = 2.), if splitting by radius, set for what multiple of the radius to use")
    ap.add_argument("-f", "--flip_split", required=False, default=False, type=str2bool,
                    help="Flip the splitting procedure, if True, all particles above mass_cut are sent to the tree code"
                         " if False (default), all particles above mass_cut are sent to the direct code")
    ap.add_argument("-t", "--timestep", required=False, default=0.1, type=float,
                    help="Timestep to save out information in Myr, default is 0.1 Myr")
    ap.add_argument("-end", "--end_time", required=False, default=10., type=float,
                    help="End time of simulation in Myr, defaults to 10. Myr")
    ap.add_argument("-r", "--virial_radius", required=False, default=3., type=float,
                    help="Virial Radius in kpc, defaults to 3.")
    ap.add_argument("-c", "--use_converter", required=False, default=True, type=str2bool,
                    help="Whether to use the converter from nbody units to physical units, with units 1 MSun, 1 kpc, defaults to True")
    ap.add_argument("-w", "--workers", required=False, default=1, type=int,
                    help="Number of workers each gravity code should use.")
    ap.add_argument("-s", "--seed", required=False, default=5227, type=int,
                    help="Seed for random numbers")
    ap.add_argument("-m", "--method", required=False, default="mass", type=str,
                    help="Method of splitting particles, either by mass cut ('mass'), virial raidus ('virial_radius'), core radius ('core radius'), or half mass radius ('half_mass')")

    args_dict = vars(ap.parse_args())

    if args_dict['direct_code'].lower() == "none":
        args_dict['direct_code'] = None

    if args_dict['tree_code'].lower() == "none":
        args_dict['tree_code'] = None

    return args_dict


def plot_sanity_checks(all_particles, direct_code_particles=None, tree_code_particles=None, stellar_distribution=None):
    """
        Plot some sanity checks to check the input to the gravity model

    :param all_particles: Set of all particles
    :param direct_code_particles: All particles in the direct code
    :param tree_code_particles: All particles in the tree code
    :param stellar_distribution: The distribution from new_powerlaw_mass_distrubtion output
    """

    plt.hist(all_particles.mass.value_in(units.MSun), histtype='step', label="Scaled Masses")
    if direct_code_particles is not None:
        plt.hist(direct_code_particles.mass.value_in(units.MSun), histtype='step', label="Direct Code Particles")
    if tree_code_particles is not None:
        plt.hist(tree_code_particles.mass.value_in(units.MSun), histtype='step', label="Tree Code Particles")
    if stellar_distribution is not None:
        plt.hist(stellar_distribution.value_in(units.MSun), histtype='step', label="Original Distribution")
    plt.yscale("log")
    plt.legend(loc="best")
    plt.xlabel("Stellar Mass (MSun)")
    plt.ylabel("Count")
    plt.title("Scaling of Stars")
    plt.savefig("Scaling_Cluster.png")
    plt.cla()


if __name__ in ('__main__', '__plot__'):
    args = get_args()
    np.random.seed(args['seed'])  # Set for reproducability
    mZAMS = new_powerlaw_mass_distribution(args['num_bodies'], 0.1 | units.MSun, 100 | units.MSun, alpha=-2.0)
    cluster_mass = mZAMS.sum()  # (args['num_bodies']) | units.MSun

    if args['use_converter']:
        converter = nbody_system.nbody_to_si(cluster_mass, args['virial_radius'] | units.parsec)
    else:
        converter = None

    particles = new_plummer_model(args['num_bodies'], convert_nbody=converter)
    particles.mass = mZAMS
    particles.scale_to_standard(convert_nbody=converter)
    tree_converter, direct_converter = make_converters(args, particles, converter)

    plot_sanity_checks(all_particles=particles, stellar_distribution=mZAMS)
    gravity = HybridGravity(direct_code=args['direct_code'],
                            tree_code=args['tree_code'],
                            mass_cut=args['mass_cut'] | units.MSun,
                            timestep=args['timestep'],
                            flip_split=args['flip_split'],
                            convert_nbody=converter,
                            tree_converter=tree_converter,
                            direct_converter=direct_converter,
                            number_of_workers=args['workers'],
                            input_args=args,
                            method=args['method'],
                            radius_multiple=args['mass_cut'])
    gravity.add_particles(particles, method=args['method'])

    timestep_history, mass_history, energy_history, half_mass_history, core_radii_history = gravity.evolve_model(
        args['end_time'] | units.Myr)

    gravity.save_model_history(output_file="History_DC_{}_TC_{}_ClusterMass_{}_"
                                           "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                                           "Timestep_{}_EndTime_{}_Method_{}.p".format(args['direct_code'],
                                                                                       args['tree_code'],
                                                                                       cluster_mass,
                                                                                       args['virial_radius'],
                                                                                       args['mass_cut'],
                                                                                       str(args['flip_split']),
                                                                                       args['num_bodies'],
                                                                                       args['timestep'],
                                                                                       args['end_time'],
                                                                                       args['method']),
                               input_dict=args)
