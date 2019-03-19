import glob
import numpy as np
from amuse.lab import *
from amuse.units import nbody_system
from amuse.datamodel import Particle, Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.units import units
from amuse.community.huayno.interface import Huayno
import amuse.datamodel.particle_attributes
import argparse
import matplotlib.colors


import matplotlib.pyplot as plt
import pickle


def sort_lists(list_one, list_two):
    """
    Sort lists based off values in the first list
    :param list_one:
    :param list_two:
    :return: Sorted lists
    """
    list_two = [x for _,x in sorted(zip(list_one, list_two))]
    list_one = sorted(list_one)

    return list_one, list_two

def get_average_model_time(walltimes, threads):
    """
    To get a better control over how long each one took, take the mode of the walltimes
    and multiply that by the number of the total threads

    Multiplied by the number of steps, and get a better value for the run time.
    :param walltimes: List of walltime for each timestep
    :param threads: The number of worker threads used in the code
    :return:
    """
    walltimes = np.diff(walltimes)
    if len(walltimes) <= 5:
        print(walltimes)
    median_walltime = np.median(walltimes)
    median_walltime *= threads
    median_walltime *= len(walltimes)

    return median_walltime


def load_history_from_file(filename):
    """
    Loads the history and diagnostics from the pickle filename

    :param filename:
    :return:
    """
    pickleFile = pickle.load(open(filename, 'rb'), fix_imports=True, encoding='latin1')

    dict_data = pickleFile[1]
    input_args = pickleFile[0]

    # Depending on version, might have wall time history or not

    walltimes = dict_data['wall_time']
    threads = input_args['workers']
    if input_args['direct_code'] is not None and input_args['tree_code'] is not None:
        threads *= 2
    walltime = get_average_model_time(walltimes, threads)

    total_particles = dict_data['num_direct'] + dict_data['num_tree']
    fraction_tree = dict_data['num_tree']/(dict_data['num_direct'] + dict_data['num_tree'])


    timesteps = np.asarray(dict_data['timestep_history'][0:102])
    energies = np.asarray(dict_data['energy_history'][0:102])
    half_mass = np.asarray(dict_data['half_mass_history'][0:102])
    core_radii = np.asarray(dict_data['core_radius_history'][0:102])
    # To fix the units
    # Get the Walltime for 10 Myr by taking the total walltime, dividing by the number of steps in the list, then 102
    walltime /= len(dict_data['timestep_history'])
    walltime *= len(timesteps)

    for i in range(len(half_mass)):
        half_mass[i] = half_mass[i].value_in(units.parsec)
        core_radii[i] = core_radii[i].value_in(units.parsec)

    mass_cut = np.asarray(dict_data['mass_cut'].value_in(units.MSun)) # Works for both mass and other methods
    flip_split = np.asarray(dict_data['flip_split'])
    integrators = (input_args['direct_code'], input_args['tree_code'])
    timesteps = np.asarray(timesteps)
    energies = np.asarray(energies)
    half_mass = np.asarray(half_mass)
    core_radii = np.asarray(core_radii)

    return timesteps, energies, half_mass, core_radii, mass_cut, flip_split, walltime, integrators, fraction_tree, total_particles


def calc_delta_energy(energies):
    """
    Given the format saved, get the energy change over time
    :param energies:
    :return: The delta_E
    """

    initial_energy = energies[0]
    delta_energy = []
    for energy in energies:
        delta_energy.append((initial_energy - energy) / initial_energy)

    return delta_energy


def plot_outputs(only_direct_name, only_tree_name, combined_names, method="Mass"):
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
        try:
            combined_datas.append(load_history_from_file(filename))
        except:
            continue

    # Set colormap here
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=np.min([x[4] for x in combined_datas]), vmax=np.max([x[4] for x in combined_datas]))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # First is the energy error over time
    for combined_data in combined_datas:
        if combined_data[5]:
            plt.plot(combined_data[0], calc_delta_energy(combined_data[1]), color=cmap(norm(combined_data[4])))
    plt.plot(direct_data[0], calc_delta_energy(direct_data[1]), label='Direct', c='black', linestyle='-.', linewidth=3.0)
    plt.plot(tree_data[0], calc_delta_energy(tree_data[1]), label='Tree', linestyle='--', c='black', linewidth=3.0)
    plt.ylabel('(E_init - E_curr)/E_init')
    plt.xlabel('Simulation Time [Myr]')
    plt.title("Relative Energy Error Method: {} Flipped".format(method))
    plt.legend(loc='best')
    sm.set_array([])
    cbar = plt.colorbar(sm)
    if method == "Mass":
        cbar.set_label("Mass Cut (MSun)")
    else:
        cbar.set_label("Multiple")
    plt.savefig("Relative_Energy_Error_DC_{}_TC_{}_Method_{}_Flipped.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method),
                dpi=300)
    plt.show()
    plt.cla()

    # First is the energy error over time
    for combined_data in combined_datas:
        if not combined_data[5]:
            plt.plot(combined_data[0], calc_delta_energy(combined_data[1]), color=cmap(norm(combined_data[4])))
    plt.plot(direct_data[0], calc_delta_energy(direct_data[1]), label='Direct', c='black', linestyle='dashed', linewidth=3.0)
    plt.plot(tree_data[0], calc_delta_energy(tree_data[1]), label='Tree', c='black', linewidth=3.0)
    plt.ylabel('(E_init - E_curr)/E_init')
    plt.xlabel('Simulation Time [Myr]')
    plt.title("Relative Energy Error Method: {}".format(method))
    plt.legend(loc='best')
    sm.set_array([])
    cbar = plt.colorbar(sm)
    if method == "Mass":
        cbar.set_label("Mass Cut (MSun)")
    else:
        cbar.set_label("Multiple")
    plt.savefig("Relative_Energy_Error_DC_{}_TC_{}_Method_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method),
                dpi=300)
    plt.show()
    plt.cla()

    # Now the Half Mass and Core Radii Over time
    for combined_data in combined_datas:
        if combined_data[5]:
            plt.plot(combined_data[0], combined_data[2], color=cmap(norm(combined_data[4])))
    plt.plot(direct_data[0], direct_data[2], label='Direct', c='black', linestyle='dashed', linewidth=3.0)
    plt.plot(tree_data[0], tree_data[2], label='Tree', c='black', linewidth=3.0)
    plt.ylabel('Half Mass Radius (parsecs)')
    plt.xlabel('Simulation Time [Myr]')
    plt.title("Half Mass Method: {} Flipped".format(method))
    plt.legend(loc='best')
    sm.set_array([])
    cbar = plt.colorbar(sm)
    if method == "Mass":
        cbar.set_label("Mass Cut (MSun)")
    else:
        cbar.set_label("Multiple")
    plt.savefig("Half_Mass_DC_{}_TC_{}_Method_{}_Flipped.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method), dpi=300)
    plt.show()
    plt.cla()

    for combined_data in combined_datas:
        if not combined_data[5]:
            plt.plot(combined_data[0], combined_data[2], color=cmap(norm(combined_data[4])))
    plt.plot(direct_data[0], direct_data[2], label='Direct', c='black', linestyle='dashed', linewidth=3.0)
    plt.plot(tree_data[0], tree_data[2], label='Tree', c='black', linewidth=3.0)
    plt.ylabel('Half Mass Radius (parsecs)')
    plt.xlabel('Simulation Time [Myr]')
    plt.title("Half Mass Method: {}".format(method))
    plt.legend(loc='best')
    sm.set_array([])
    cbar = plt.colorbar(sm)
    if method == "Mass":
        cbar.set_label("Mass Cut (MSun)")
    else:
        cbar.set_label("Multiple")
    plt.savefig("Half_Mass_DC_{}_TC_{}_Method_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method), dpi=300)
    plt.show()
    plt.cla()

    for combined_data in combined_datas:
        if combined_data[5]:
            plt.plot(combined_data[0], combined_data[3], color=cmap(norm(combined_data[4])))
    plt.plot(direct_data[0], direct_data[3], label='Direct', c='black', linestyle='dashed', linewidth=3.0)
    plt.plot(tree_data[0], tree_data[3], label='Tree', c='black', linewidth=3.0)
    plt.ylabel('Core Radius (parsecs)')
    plt.xlabel('Simulation Time [Myr]')
    plt.title("Core Radius Method: {} Flipped".format(method))
    plt.legend(loc='best')
    sm.set_array([])
    cbar = plt.colorbar(sm)
    if method == "Mass":
        cbar.set_label("Mass Cut (MSun)")
    else:
        cbar.set_label("Multiple")
    plt.savefig("Core_Radii_DC_{}_TC_{}_Method_{}_Flipped.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method), dpi=300)
    plt.show()
    plt.cla()

    for combined_data in combined_datas:
        if not combined_data[5]:
            plt.plot(combined_data[0], combined_data[3], color=cmap(norm(combined_data[4])))
    plt.plot(direct_data[0], direct_data[3], label='Direct', c='black', linestyle='dashed', linewidth=3.0)
    plt.plot(tree_data[0], tree_data[3], label='Tree', c='black', linewidth=3.0)
    plt.ylabel('Core Radius (parsecs)')
    plt.xlabel('Simulation Time [Myr]')
    plt.title("Core Radius Method: {}".format(method))
    plt.legend(loc='best')
    sm.set_array([])
    cbar = plt.colorbar(sm)
    if method == "Mass":
        cbar.set_label("Mass Cut (MSun)")
    else:
        cbar.set_label("Multiple")
    plt.savefig("Core_Radii_DC_{}_TC_{}_Method_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method), dpi=300)
    plt.show()
    plt.cla()

    # Final Energy Error as function of the split

    # First need to change it to the available cuts
    final_error = []
    done_splits = []
    flipped_final_error = []
    flipped_done_splits = []

    final_error.append((direct_data[1][0] - direct_data[1][-1]) / direct_data[1][0])
    final_error.append((tree_data[1][0] - tree_data[1][-1]) / tree_data[1][0])
    done_splits.append(1.0)
    done_splits.append(0.0)

    flipped_final_error.append((direct_data[1][0] - direct_data[1][-1]) / direct_data[1][0])
    flipped_final_error.append((tree_data[1][0] - tree_data[1][-1]) / tree_data[1][0])
    flipped_done_splits.append(0.0)
    flipped_done_splits.append(1.0)
    for combined_data in combined_datas:
        print((combined_data[1][0] - combined_data[1][-1]) / combined_data[1][0])
        if not combined_data[5]:
            final_error.append((combined_data[1][0] - combined_data[1][-1]) / combined_data[1][0])
            done_splits.append(combined_data[4])
        else:
            flipped_final_error.append((combined_data[1][0] - combined_data[1][-1]) / combined_data[1][0])
            flipped_done_splits.append(combined_data[4])
    flipped_final_error = np.asarray(flipped_final_error)

    flipped_done_splits, flipped_final_error = sort_lists(flipped_done_splits, flipped_final_error)
    done_splits, final_error = sort_lists(done_splits, final_error)

    plt.plot(done_splits, final_error, label='Direct >= Cut', c='r')
    plt.plot(flipped_done_splits, flipped_final_error, label='Tree >= Cut', c='b')
    if method == "Mass":
        plt.xlabel("Mass Split (MSun)")
    else:
        plt.xlabel("Split (Multiple of Value)")
    plt.ylabel("(E_init - E_final)/E_init")
    plt.title("Final Energy Error by Cut Method: {}".format(method))
    plt.legend(loc='best')
    plt.savefig("Mass_Split_Final_Error_DC_{}_TC_{}_Method_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method),
                dpi=300)
    plt.show()
    plt.cla()

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
            final_walltime.append(combined_data[6])
            mass_cut_list.append(combined_data[4])
        else:
            flipped_final_walltime.append(combined_data[6])
            flipped_mass_cut_list.append(combined_data[4])

    mass_cut_list, final_walltime = sort_lists(mass_cut_list, final_walltime)
    flipped_mass_cut_list, flipped_final_walltime = sort_lists(flipped_mass_cut_list, flipped_final_walltime)
    plt.plot(mass_cut_list, final_walltime, label='Direct >= Cut', c='r')
    plt.plot(flipped_mass_cut_list, flipped_final_walltime, label='Tree >= Cut', c='b')
    if method == "Mass":
        plt.xlabel("Mass Split (MSun)")
    else:
        plt.xlabel("Split (Multiple of Value)")
    plt.ylabel("Walltime (sec)")
    plt.title("Walltime vs Split Method: {}".format(method))
    plt.legend(loc='best')
    plt.savefig("Mass_Split_Walltime_DC_{}_TC_{}_Method_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method), dpi=300)
    plt.show()
    plt.cla()

    # Now Percentage by time ran
    final_walltime = []
    mass_cut_list = []

    final_walltime.append(direct_data[6])
    final_walltime.append(tree_data[6])
    mass_cut_list.append(direct_data[8])
    mass_cut_list.append(direct_data[8])

    for combined_data in combined_datas:
        final_walltime.append(combined_data[6])
        mass_cut_list.append(combined_data[8])

    mass_cut_list, final_walltime = sort_lists(mass_cut_list, final_walltime)
    plt.plot(mass_cut_list, final_walltime, c='r')
    plt.xlabel("Fraction of Particles in Tree Code")
    plt.ylabel("Walltime (sec)")
    plt.title("Walltime vs Fraction in Tree Method: {}".format(method))
    plt.savefig("Fractional_Walltime_DC_{}_TC_{}_Method_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method), dpi=300)
    plt.show()
    plt.cla()

    # Now Final Energy Error by Percentage
    final_error = []
    done_splits = []

    final_error.append((direct_data[1][0] - direct_data[1][-1]) / direct_data[1][0])
    final_error.append((tree_data[1][0] - tree_data[1][-1]) / tree_data[1][0])
    done_splits.append(direct_data[8])
    done_splits.append(tree_data[8])

    for combined_data in combined_datas:
        print((combined_data[1][0] - combined_data[1][-1]) / combined_data[1][0])
        final_error.append((combined_data[1][0] - combined_data[1][-1]) / combined_data[1][0])
        done_splits.append(combined_data[8])

    done_splits, final_error = sort_lists(done_splits, final_error)
    plt.plot(done_splits, final_error, c='b')
    plt.xlabel("Fraction of Particles in Tree Code")
    plt.ylabel("(E_init - E_final)/E_init")
    plt.title("Final Energy Error by Particle Fraction Method: {}".format(method))
    plt.savefig("Particle_Fraction_Final_Error_DC_{}_TC_{}_Method_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1], method),
                dpi=300)
    plt.show()
    plt.cla()


def make_walltime_vs_points_plot(direct_filenames, tree_filenames, combined_filenames):
    """
    Creates a plot of the wall time vs the number of points
    :param combined_filenames:
    :return:
    """
    direct_datas = []
    for filename in direct_filenames:
        direct_datas.append(load_history_from_file(filename))
    tree_datas = []
    for filename in tree_filenames:
        tree_datas.append(load_history_from_file(filename))

    combined_datas = []
    flipped_combined_datas = []
    for filename in combined_filenames:
        data = load_history_from_file(filename)
        if not data[5]:
            combined_datas.append(data)
        else:
            flipped_combined_datas.append(data)

    wall_time = []
    num_particles = []

    for combined_data in combined_datas:
        wall_time.append(combined_data[6])
        num_particles.append(combined_data[9])

    flipped_wall_time = []
    flipped_num_particles = []

    for combined_data in flipped_combined_datas:
        flipped_wall_time.append(combined_data[6])
        flipped_num_particles.append(combined_data[9])

    direct_wall_time = []
    direct_num_particles = []

    for combined_data in direct_datas:
        direct_wall_time.append(combined_data[6])
        direct_num_particles.append(combined_data[9])

    tree_wall_time = []
    tree_num_particles = []

    for combined_data in tree_datas:
        tree_wall_time.append(combined_data[6])
        tree_num_particles.append(combined_data[9])

    # Now sorting
    tree_wall_time = [x for _,x in sorted(zip(tree_num_particles, tree_wall_time))]
    tree_num_particles = sorted(tree_num_particles)
    direct_wall_time = [x for _,x in sorted(zip(direct_num_particles, direct_wall_time))]
    direct_num_particles = sorted(direct_num_particles)
    wall_time = [x for _,x in sorted(zip(num_particles, wall_time))]
    num_particles = sorted(num_particles)
    flipped_wall_time = [x for _,x in sorted(zip(flipped_num_particles, wall_time))]
    flipped_num_particles = sorted(flipped_num_particles)
    plt.plot(num_particles, wall_time, c='g', label='Flip = False')
    plt.plot(flipped_num_particles, flipped_wall_time, c='y', label='Flip = True')
    plt.plot(direct_num_particles, direct_wall_time, c='b', label='Direct')
    plt.plot(tree_num_particles, tree_wall_time, c='r', label='Tree')
    plt.xlabel("Number of Particles")
    plt.ylabel("Walltime (seconds)")
    plt.title("Walltime vs Number of Particles")
    plt.legend(loc='best')
    plt.savefig("Walltime_vs_Number_DC_{}_TC_{}.png".format(combined_datas[0][7][0], combined_datas[0][7][1]),
                dpi=300)
    plt.show()
    plt.cla()


if __name__ in ('__main__', '__plot__'):
    mixed_ph4_bhtree = []
    for file in glob.glob("Base_Test/combined/*.p"):
        mixed_ph4_bhtree.append(file)
    #for file in glob.glob("Base_Test/combined_10/*.p"):
    #    mixed_ph4_bhtree.append(file)
    bh_tree_only = '/home/jacob/Development/comp_astro/assignment_three/Base_Test/Checkpoint_DC_None_TC_bhtree_ClusterMass_6958.065386227829_Radius_3.0_Cut_6.0_Flip_False_Stars_10000_Timestep_0.1_EndTime_100.0.p'
    ph4_only = '/home/jacob/Development/comp_astro/assignment_three/Base_Test/Checkpoint_DC_ph4_TC_None_ClusterMass_6958.065386227829_Radius_3.0_Cut_6.0_Flip_False_Stars_10000_Timestep_0.1_EndTime_100.0.p'

    direct_filenames = []
    tree_filenames = []
    combined_filenames = []
    for file in glob.glob("walltime_by_n/direct/*.p"):
        direct_filenames.append(file)
    for file in glob.glob("walltime_by_n/combined/*.p"):
        combined_filenames.append(file)
    for file in glob.glob("walltime_by_n/tree/*.p"):
        tree_filenames.append(file)
    #make_walltime_vs_points_plot(direct_filenames, tree_filenames, combined_filenames)
    plot_outputs(ph4_only, bh_tree_only, mixed_ph4_bhtree)
    #exit()
    half_mass_files = []
    core_radius_files = []
    virial_files = []
    for file in glob.glob("/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/radii/*half_mass.p"):
        half_mass_files.append(file)
    for file in glob.glob("/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/radii/*core_radius.p"):
        core_radius_files.append(file)
    for file in glob.glob("/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/radii/*virial_radius.p"):
        virial_files.append(file)
    plot_outputs(ph4_only, bh_tree_only, half_mass_files, method="Half Mass")
    plot_outputs(ph4_only, bh_tree_only, core_radius_files, method="Core Radius")
    plot_outputs(ph4_only, bh_tree_only, virial_files, method="Virial Radius")
