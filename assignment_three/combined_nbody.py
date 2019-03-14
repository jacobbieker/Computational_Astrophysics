import numpy as np
from amuse.lab import *
from amuse.units import nbody_system
from amuse.datamodel import Particle, Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.units import units
from HybridGravity import HybridGravity
from amuse.community.huayno.interface import Huayno
import amuse.datamodel.particle_attributes
import argparse

import matplotlib.pyplot as plt
import pickle


def load_history_from_file(filename):
    pickleFile = pickle.load(open(filename, 'rb'), fix_imports=True, encoding='latin1')

    dict_data = pickleFile[1]
    input_args = pickleFile[0]

    # Depending on version, might have wall time history or not

    try:
        walltime = dict_data['total_elapsed_time']
    except KeyError:
        walltime = dict_data['wall_time']

    timesteps = dict_data['timestep_history'] * 0.1
    energies = dict_data['energy_history']
    half_mass = dict_data['half_mass_history']
    core_radii = dict_data['core_radius_history']
    mass_cut = dict_data['mass_cut'].value_in(units.MSun)
    flip_split = dict_data['flip_split']
    integrators = (input_args['direct_code'], input_args['tree_code'])

    return timesteps, energies, half_mass, core_radii, mass_cut, flip_split, walltime, integrators

def calc_delta_energy(energies):
    """
    Given the format saved, get the energy change over time
    :param energies:
    :return: The delta_E
    """

    initial_energy = energies[0]
    delta_energy = []
    for energy in energies:
        delta_energy.append((initial_energy - energy)/initial_energy)

    return delta_energy

def plot_outputs(only_direct_name, only_tree_name, combined_names):
    """
    Makes most of the plot outputs for the report
    Given a direct_name, tree_name, and set of combined_names, load the data into dataframes and run from there

    Plots the energy distribution
    The Wall time vs mass_cut

    The Final energy error as function of split

    The Relative Error Energy

    Half mass and core radii as function of time


    :param only_direct_name:
    :param only_tree_name:
    :param combined_names:
    :return:
    """

    direct_data = load_history_from_file(only_direct_name)
    tree_data = load_history_from_file(only_tree_name)
    combined_datas = []
    for filename in combined_names:
        combined_datas.append(load_history_from_file(filename))


    # First is the energy error over time
    plt.plot(direct_data[0], calc_delta_energy(direct_data[1]), label='Direct', c='r')
    plt.plot(tree_data[0], calc_delta_energy(tree_data[1]), label='Tree', linestyle='dashed', c='b')
    for combined_data in combined_datas:
        plt.plot(combined_data[0], calc_delta_energy(combined_data[1]), label='Cut: {}'.format(combined_data[4]))
    plt.ylabel('(E_init - E_curr)/E_init')
    plt.xlabel('Simulation Time [Myr]')
    plt.title("Relative Energy Error")
    plt.legend(loc='best')
    plt.savefig("Relative_Energy_Error_DC_{}_TC_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1]), dpi=300)

    # Now the Half Mass and Core Radii Over time
    plt.plot(direct_data[0], direct_data[2], label='Direct Half Mass', c='r')
    plt.plot(tree_data[0], tree_data[2], label='Tree Half Mass', linestyle='dashed', c='r')
    for combined_data in combined_datas:
        plt.plot(combined_data[0], combined_data[2], label='Cut: {} Half Mass'.format(combined_data[4]))
    plt.plot(direct_data[0], direct_data[3], label='Direct Core Radii', c='b')
    plt.plot(tree_data[0], tree_data[3], label='Tree Core Radii', linestyle='dashed', c='b')
    for combined_data in combined_datas:
        plt.plot(combined_data[0], combined_data[3], label='Cut: {} Core Radii'.format(combined_data[4]))
    plt.ylabel('Value (parsecs)')
    plt.xlabel('Simulation Time [Myr]')
    plt.title("Half Mass and Core Radii")
    plt.legend(loc='best')
    plt.savefig("Half_Mass_Core_Radii_DC_{}_TC_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1]), dpi=300)

    # Final Energy Error as function of the split

    # First need to change it to the available cuts
    final_error = []
    done_splits = []
    flipped_final_error = []
    flipped_done_splits = []

    final_error.append(calc_delta_energy([direct_data[1][-1]]))
    final_error.append(calc_delta_energy([tree_data[1][-1]]))
    done_splits.append(1.0)
    done_splits.append(0.0)

    flipped_final_error.append(calc_delta_energy([direct_data[1][-1]]))
    flipped_final_error.append(calc_delta_energy([tree_data[1][-1]]))
    flipped_done_splits.append(0.0)
    flipped_done_splits.append(1.0)
    for combined_data in combined_datas:
        if not combined_data[5]:
            final_error.append(calc_delta_energy([combined_data[1][-1]]))
            done_splits.append(combined_data[3])
        else:
            flipped_final_error.append(calc_delta_energy([combined_data[1][-1]]))
            flipped_done_splits.append(combined_data[3])

    plt.plot(done_splits, final_error, label='Direct >= Cut', c='r')
    plt.plot(flipped_done_splits, flipped_final_error, label='Tree >= Cut', c='b')
    plt.xlabel("Mass Split (MSun)")
    plt.ylabel("(E_init - E_final)/E_init")
    plt.title("Final Energy Error by Mass Cut")
    plt.legend(loc='best')
    plt.savefig("Mass_Split_Final_Error_DC_{}_TC_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1]), dpi=300)

    # Wall Time vs Mass Cut
    final_walltime = []
    mass_cut_list = []
    flipped_final_walltime = []
    flipped_mass_cut_list = []

    final_walltime.append(direct_data[6])
    final_walltime.append(tree_data[6])
    mass_cut_list.append(1.0)
    mass_cut_list.append(0.0)

    flipped_final_walltime.append(direct_data[6])
    flipped_final_walltime.append(tree_data[6])
    flipped_mass_cut_list.append(0.0)
    flipped_mass_cut_list.append(1.0)
    for combined_data in combined_datas:
        if not combined_data[5]:
            final_walltime.append([combined_data[6]])
            mass_cut_list.append(combined_data[3])
        else:
            flipped_final_walltime.append(combined_data[6])
            flipped_mass_cut_list.append(combined_data[3])

    plt.plot(mass_cut_list, final_walltime, label='Direct >= Cut', c='r')
    plt.plot(flipped_mass_cut_list, flipped_final_walltime, label='Tree >= Cut', c='b')
    plt.xlabel("Mass Split (MSun)")
    plt.ylabel("Walltime (sec)")
    plt.title("Walltime vs Mass Split")
    plt.legend(loc='best')
    plt.savefig("Mass_Split_Walltime_DC_{}_TC_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1]), dpi=300)

    return NotImplementedError


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
    ap.add_argument("-mc", '--mass_cut', required=False, default=6., type=float,
                    help="Mass Cutoff for splitting bodies, in units MSun (default = 6.)")
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
    np.random.seed(args['seed']) # Set for reproducability
    print(args)
    mZAMS = new_powerlaw_mass_distribution(args['num_bodies'], 0.1 | units.MSun, 100 | units.MSun, alpha=-2.0)
    cluster_mass = mZAMS.sum() # (args['num_bodies']) | units.MSun

    print(np.sum(mZAMS))
    print(mZAMS.sum())
    print(cluster_mass)
    if args['use_converter']:
        converter = nbody_system.nbody_to_si(cluster_mass, args['virial_radius'] | units.parsec)
    else:
        converter = None

    particles = new_plummer_model(args['num_bodies'], convert_nbody=converter)
    particles.mass = mZAMS
    particles.scale_to_standard(convert_nbody=converter)

    print(particles.virial_radius().value_in(units.parsec))

    # Now get the masses in each one for the different converters
    direct_particles = Particles()
    tree_particles = Particles()
    tree_converter = None
    direct_converter = None
    if args['tree_code'] is not None and args['direct_code'] is not None:
        if args['method'] == 'mass':
            for particle in particles:
                if particle.mass >= args['mass_cut'] | units.MSun:
                    if args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        elif args['method'] == 'core_radius':
            _, core_radius, _ = particles.densitycentre_coreradius_coredens(
                unit_converter=converter)
            for particle in particles:
                if np.sqrt((particle.x - particles.center_of_mass().x)**2 + (particle.y - particles.center_of_mass().y)**2 + (particle.z - particles.center_of_mass().z)**2) <= core_radius:
                    if args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        elif args['method'] == 'half_mass':
            half_mass_radius = \
                particles.LagrangianRadii(mf=[0.5], cm=particles.center_of_mass(),
                                          unit_converter=converter)[0][0]
            for particle in particles:
                if np.sqrt((particle.x - particles.center_of_mass().x)**2 + (particle.y - particles.center_of_mass().y)**2 + (particle.z - particles.center_of_mass().z)**2) <= half_mass_radius:
                    if args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        elif args['method'] == 'virial_radius':
            virial_radius = particles.virial_radius()
            for particle in particles:
                if np.sqrt((particle.x - particles.center_of_mass().x)**2 + (particle.y - particles.center_of_mass().y)**2 + (particle.z - particles.center_of_mass().z)**2) <= virial_radius:
                    if args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        if args['tree_code'] is not None:
            tree_converter = nbody_system.nbody_to_si(tree_particles.mass.sum(), tree_particles.virial_radius())
        if args['direct_code'] is not None:
            direct_converter = nbody_system.nbody_to_si(direct_particles.mass.sum(), direct_particles.virial_radius())
        plot_sanity_checks(particles, direct_particles, tree_particles, mZAMS)
        print(direct_particles.mass.sum().value_in(units.MSun))
        print(tree_particles.mass.sum().value_in(units.MSun))
        print(particles.mass.sum().value_in(units.MSun))
        print(direct_particles.virial_radius().value_in(units.parsec))
        print(tree_particles.virial_radius().value_in(units.parsec))
        print(particles.virial_radius().value_in(units.parsec))
    else:
        # No splitting, so all the converters are the same
        tree_converter = converter
        direct_converter = converter
        plot_sanity_checks(all_particles=particles, stellar_distribution=mZAMS)
    # plt.plot(particles.x.value_in(units.parsec), particles.y.value_in(units.parsec))
    # plt.show()
    # exit()
    # set_standard scale to rescale it
    gravity = HybridGravity(direct_code=args['direct_code'],
                            tree_code=args['tree_code'],
                            mass_cut=args['mass_cut'] | units.MSun,
                            timestep=args['timestep'],
                            flip_split=args['flip_split'],
                            convert_nbody=converter,
                            tree_converter=tree_converter,
                            direct_converter=direct_converter,
                            number_of_workers=args['workers'],
                            input_args=args)
    gravity.add_particles(particles, method=args['method'])

    timestep_history, mass_history, energy_history, half_mass_history, core_radii_history = gravity.evolve_model(
        args['end_time'] | units.Myr)

    gravity.save_model_history(output_file="History_DC_{}_TC_{}_ClusterMass_{}_"
                                           "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                                           "Timestep_{}_EndTime_{}.p".format(args['direct_code'],
                                                                  args['tree_code'],
                                                                  cluster_mass,
                                                                  args['virial_radius'],
                                                                  args['mass_cut'],
                                                                  str(args['flip_split']),
                                                                  args['num_bodies'],
                                                                  args['timestep'],
                                                                    args['end_time']),
                               input_dict=args)

    print("Timestep length: {}".format(len(timestep_history)))

    plt.plot(timestep_history, mass_history, label="Mass")
    plt.plot(timestep_history, energy_history, label="Energy")
    plt.plot(timestep_history, half_mass_history, label="Half-Mass")
    plt.plot(timestep_history, core_radii_history, label="Core Radii")
    plt.xlabel("Timestep")
    plt.ylabel("Ratio Current/Initial")
    plt.title(
        "Histories: DC {} TC {} WallTime: {} s".format(args['direct_code'], args['tree_code'], np.round(gravity.elapsed_time, 3)))
    plt.legend(loc='best')
    plt.savefig("History_DC_{}_TC_{}_ClusterMass_{}_Radius_{}_Cut_{}_Flip_{}_Stars_{}_Timestep_{}.png".format(
        args['direct_code'], args['tree_code'], cluster_mass, args['virial_radius'], args['mass_cut'],
        str(args['flip_split']), args['num_bodies'], args['timestep']))
    plt.cla()

    print(max(energy_history))
    print(min(energy_history))
