from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
from amuse.lab import *
from amuse.units import units
from amuse.io import read_set_from_file
from amuse.ic.plummer import new_plummer_model
import os
from amuse.units import nbody_system
from amuse.ic.salpeter import new_powerlaw_mass_distribution
import matplotlib.patches as mpatches


def convert_from_pickle(filename):
    """
    Converts from the pickle file to the format needed for the animations
    :param filename:
    :return:
    """
    pickleFile = pickle.load(open(filename, 'rb'), fix_imports=True, encoding='latin1')

    np.random.seed(5227)  # Set for reproducability
    mZAMS = new_powerlaw_mass_distribution(10000, 0.1 | units.MSun, 100 | units.MSun, alpha=-2.0)
    cluster_mass = mZAMS.sum()  # (args['num_bodies']) | units.MSun
    converter = nbody_system.nbody_to_si(cluster_mass, 3 | units.parsec)
    particles = new_plummer_model(10000, convert_nbody=converter)
    particles.mass = mZAMS
    particles.scale_to_standard(convert_nbody=converter)
    center_of_mass = particles.center_of_mass()
    virial_radius = particles.virial_radius()
    _, core_radius, _ = particles.densitycentre_coreradius_coredens(
        unit_converter=converter)
    half_mass = particles.LagrangianRadii(mf=[0.5],
                                          cm=center_of_mass,
                                          unit_converter=converter)[0][0]

    dict_data = pickleFile[1]
    args = pickleFile[0]

    locations = dict_data['combined_particles_locations']
    masses = dict_data['particle_history'][0]
    xdata = []
    ydata = []
    zdata = []
    for list in locations:
        xdata.append(list[0])
        ydata.append(list[1])
        zdata.append(list[2])

    # Now it is a list of lists each, each sblist being the positions at that timestep
    # So make each a numpy array of arrays
    # This then means each df thing should be an array of them
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    zdata = np.asarray(zdata)
    num_timesteps = len(xdata)

    # Add the colors, based on the mthod and cut
    cut = dict_data['mass_cut']
    if "method" in dict_data:
        method = dict_data['method']
    else:
        method = 'mass'

    # Get the scaling for the points
    scaling_list = []
    for mass in masses:
        scaling_list.append(mass / np.max(masses))

    scaling_list = np.asarray(scaling_list)

    tree_color = 'b'
    direct_color = 'r'

    colors = []
    if args['tree_code'] is not None and args['direct_code'] is not None:
        if method == 'mass':
            for particle in masses:
                if particle >= cut.value_in(units.MSun):
                    if dict_data['flip_split']:
                        colors.append(tree_color)
                    else:
                        colors.append(direct_color)
                else:
                    if dict_data['flip_split']:
                        colors.append(direct_color)
                    else:
                        colors.append(tree_color)
        elif method == "virial_radius":
            for index, (x_val, y_val, z_val) in enumerate(zip(xdata[0], ydata[0], zdata[0])):
                if np.sqrt((x_val - center_of_mass.x.value_in(units.parsec)) ** 2 + (
                        y_val - center_of_mass.y.value_in(units.parsec)) ** 2 + (
                                   z_val - center_of_mass.z.value_in(units.parsec)) ** 2) <= args['mass_cut'] * \
                        virial_radius.value_in(units.parsec):
                    if dict_data['flip_split']:
                        colors.append(tree_color)
                    else:
                        colors.append(direct_color)
                else:
                    if dict_data['flip_split']:
                        colors.append(direct_color)
                    else:
                        colors.append(tree_color)
        elif method == "half_mass":
            for index, (x_val, y_val, z_val) in enumerate(zip(xdata[0], ydata[0], zdata[0])):
                if np.sqrt((x_val - center_of_mass.x.value_in(units.parsec)) ** 2 + (
                        y_val - center_of_mass.y.value_in(units.parsec)) ** 2 + (
                                   z_val - center_of_mass.z.value_in(units.parsec)) ** 2) <= args['mass_cut'] * \
                        half_mass.value_in(units.parsec):
                    if dict_data['flip_split']:
                        colors.append(tree_color)
                    else:
                        colors.append(direct_color)
                else:
                    if dict_data['flip_split']:
                        colors.append(direct_color)
                    else:
                        colors.append(tree_color)
        elif method == "core_radius":
            for index, (x_val, y_val, z_val) in enumerate(zip(xdata[0], ydata[0], zdata[0])):
                if np.sqrt((x_val - center_of_mass.x.value_in(units.parsec)) ** 2 + (
                        y_val - center_of_mass.y.value_in(units.parsec)) ** 2 + (
                                   z_val - center_of_mass.z.value_in(units.parsec)) ** 2) <= args['mass_cut'] * \
                        core_radius.value_in(units.parsec):
                    if dict_data['flip_split']:
                        colors.append(tree_color)
                    else:
                        colors.append(direct_color)
                else:
                    if dict_data['flip_split']:
                        colors.append(direct_color)
                    else:
                        colors.append(tree_color)
    elif args['tree_code'] is None and args['direct_code'] is not None:
        colors = np.asarray([direct_color for _ in range(len(scaling_list))])
    elif args['tree_code'] is not None and args['direct_code'] is None:
        colors = np.asarray([tree_color for _ in range(len(scaling_list))])

    return xdata, ydata, zdata, colors, args, num_timesteps, scaling_list


def create_3d_array(direct_positions, tree_positions, false_positions, true_positions, direct_colors, tree_colors,
                    false_colors, true_colors, input_args, num_timesteps, scaling_list, axlims=None):
    """
    This creates a 2 x 2 showing the different effects that the cuts have on the animation
    :param direct_positions:
    :param tree_positions:
    :param false_positions:
    :param true_positions:
    :param false_colors:
    :param true_colors:
    :param input_args:
    :param num_timesteps:
    :param scaling_list:
    :param axlims:
    :return:
    """

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)

    def update_graph(num):
        """
        Updates the graph's positions
        :param num: The number in the sequence to use
        """
        # Update all the stars
        graph._offsets3d = (
            direct_positions[0][num], direct_positions[1][num], direct_positions[2][num])  # Have to do this for the
        graph1._offsets3d = (
            false_positions[0][num], false_positions[1][num], false_positions[2][num])  # Have to do this for the
        graph2._offsets3d = (
            true_positions[0][num], true_positions[1][num], true_positions[2][num])  # Have to do this for the
        graph3._offsets3d = (
            tree_positions[0][num], tree_positions[1][num], tree_positions[2][num])  # Have to do this for the
        fig.suptitle('DC: {} TC: {} Cut: {} Method: {} \n Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                                   input_args['tree_code'],
                                                                                   input_args['mass_cut'], method,
                                                                                   np.round(num * 0.1, 2)))
        return graph, graph1, graph2, graph3

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    ax.set_zlabel("Z [parsec]")
    ax.set_xlabel("X [parsec]")
    ax.set_ylabel("Y [parsec]")
    ax.set_title("Direct Only")
    ax2.set_zlabel("Z [parsec]")
    ax2.set_ylabel("Y [parsec]")
    ax2.set_xlabel("X [parsec]")
    ax2.set_title("Direct >= Split")
    ax3.set_zlabel("Z [parsec]")
    ax3.set_xlabel("X [parsec]")
    ax3.set_ylabel("Y [parsec]")
    ax3.set_title("Tree >= Split")
    ax4.set_xlabel("X [parsec]")
    ax4.set_ylabel("Y [parsec]")
    ax4.set_zlabel("Z [parsec]")
    ax4.set_title("Tree Only")

    graph = ax.scatter(direct_positions[0][0], direct_positions[1][0], direct_positions[2][0], s=50 * scaling_list,
                       c=direct_colors)
    graph3 = ax4.scatter(tree_positions[0][0], tree_positions[1][0], tree_positions[2][0], s=50 * scaling_list,
                         c=tree_colors)
    graph1 = ax2.scatter(false_positions[0][0], false_positions[1][0], false_positions[2][0], s=50 * scaling_list,
                         c=false_colors)
    graph2 = ax3.scatter(true_positions[0][0], true_positions[1][0], true_positions[2][0], s=50 * scaling_list,
                         c=true_colors)

    red_patch = mpatches.Patch(color='blue', label='Direct Particles')
    blue_patch = mpatches.Patch(color='red', label='Tree Particles')
    fig.legend(handles=[red_patch, blue_patch])

    for ax in fig.get_axes():
        if axlims is not None:
            if isinstance(axlims[0], list):
                ax.set_xlim3d(axlims[0])
                ax.set_ylim3d(axlims[1])
                ax.set_zlim3d(axlims[2])
            else:
                ax.set_xlim3d(axlims)
                ax.set_ylim3d(axlims)
                ax.set_zlim3d(axlims)

    if 'method' in input_args.keys():
        method = input_args['method']
    else:
        method = "mass"
    fig.suptitle('DC: {} TC: {} Cut: {} Method: {} \n Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                               input_args['tree_code'],
                                                                               input_args['mass_cut'], method,
                                                                               0.0))

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, num_timesteps, interval=50, blit=False)
    # plt.show()
    if axlims is None:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}_Combined_4.mp4".format(input_args['direct_code'],
                                                     input_args['tree_code'],
                                                     input_args['virial_radius'],
                                                     input_args['mass_cut'],
                                                     str(input_args['flip_split']),
                                                     input_args['num_bodies'],
                                                     input_args['timestep']), writer=writer)
    else:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}_AxLim_{}_Combined_4.mp4".format(input_args['direct_code'],
                                                              input_args['tree_code'],
                                                              input_args['virial_radius'],
                                                              input_args['mass_cut'],
                                                              str(input_args['flip_split']),
                                                              input_args['num_bodies'],
                                                              input_args['timestep'],
                                                              axlims), writer=writer)
    plt.cla()
    plt.close(fig)


def create_3d_array_radii(direct_positions, tree_positions, virial_positions, core_positions, half_mass_positions,
                          direct_colors, tree_colors,
                          virial_colors, core_colors, half_mass_colors, input_args, num_timesteps, scaling_list,
                          axlims=None):
    """
    This creates a 2 x 3 showing the different effects that the cuts have on the cluster
    :param direct_positions:
    :param tree_positions:
    :param virial_positions:
    :param core_positions:
    :param virial_colors:
    :param core_colors:
    :param input_args:
    :param num_timesteps:
    :param scaling_list:
    :param axlims:
    :return:
    """

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)

    def update_graph(num):
        """
        Updates the graph's positions
        :param num: The number in the sequence to use
        """
        # Update all the stars
        graph._offsets3d = (
            direct_positions[0][num], direct_positions[1][num], direct_positions[2][num])  # Have to do this for the
        graph1._offsets3d = (
            virial_positions[0][num], virial_positions[1][num], virial_positions[2][num])  # Have to do this for the
        graph2._offsets3d = (
            core_positions[0][num], core_positions[1][num], core_positions[2][num])  # Have to do this for the
        graph3._offsets3d = (
            tree_positions[0][num], tree_positions[1][num], tree_positions[2][num])  # Have to do this for the
        graph4._offsets3d = (
            half_mass_positions[0][num], half_mass_positions[1][num], half_mass_positions[2][num])  # Have to do this for the

        fig.suptitle('DC: {} TC: {} Cut: {} \n Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                        input_args['tree_code'],
                                                                        input_args['mass_cut'],
                                                                        np.round(num * 0.1, 2)))
        return graph, graph1, graph2, graph3, graph4

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(232, projection='3d')
    ax3 = fig.add_subplot(233, projection='3d')
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')
    ax.set_zlabel("Z [parsec]")
    ax.set_xlabel("X [parsec]")
    ax.set_ylabel("Y [parsec]")
    ax.set_title("Direct Only")
    ax2.set_zlabel("Z [parsec]")
    ax2.set_ylabel("Y [parsec]")
    ax2.set_xlabel("X [parsec]")
    ax2.set_title("Virial Radius")
    ax3.set_zlabel("Z [parsec]")
    ax3.set_xlabel("X [parsec]")
    ax3.set_ylabel("Y [parsec]")
    ax3.set_title("Core Radius")
    ax4.set_xlabel("X [parsec]")
    ax4.set_ylabel("Y [parsec]")
    ax4.set_zlabel("Z [parsec]")
    ax4.set_title("Tree Only")
    ax5.set_xlabel("X [parsec]")
    ax5.set_ylabel("Y [parsec]")
    ax5.set_zlabel("Z [parsec]")
    ax5.set_title("Half Mass")

    graph = ax.scatter(direct_positions[0][0], direct_positions[1][0], direct_positions[2][0], s=50 * scaling_list,
                       c=direct_colors)
    graph3 = ax4.scatter(tree_positions[0][0], tree_positions[1][0], tree_positions[2][0], s=50 * scaling_list,
                         c=tree_colors)
    graph1 = ax2.scatter(virial_positions[0][0], virial_positions[1][0], virial_positions[2][0], s=50 * scaling_list,
                         c=virial_colors)
    graph2 = ax3.scatter(core_positions[0][0], core_positions[1][0], core_positions[2][0], s=50 * scaling_list,
                         c=core_colors)
    graph4 = ax5.scatter(half_mass_positions[0][0], half_mass_positions[1][0], half_mass_positions[2][0],
                         s=50 * scaling_list,
                         c=half_mass_colors)

    red_patch = mpatches.Patch(color='blue', label='Direct Particles')
    blue_patch = mpatches.Patch(color='red', label='Tree Particles')
    fig.legend(handles=[red_patch, blue_patch])

    for ax in fig.get_axes():
        if axlims is not None:
            if isinstance(axlims[0], list):
                ax.set_xlim3d(axlims[0])
                ax.set_ylim3d(axlims[1])
                ax.set_zlim3d(axlims[2])
            else:
                ax.set_xlim3d(axlims)
                ax.set_ylim3d(axlims)
                ax.set_zlim3d(axlims)

    fig.suptitle('DC: {} TC: {} Multiple: {} \n Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                         input_args['tree_code'],
                                                                         input_args['mass_cut'],
                                                                         0.0))

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, num_timesteps, interval=50, blit=False)
    # plt.show()
    if axlims is None:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}_Combined_Radii.mp4".format(input_args['direct_code'],
                                                         input_args['tree_code'],
                                                         input_args['virial_radius'],
                                                         input_args['mass_cut'],
                                                         str(input_args['flip_split']),
                                                         input_args['num_bodies'],
                                                         input_args['timestep']), writer=writer)
    else:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}_AxLim_{}_Combined_Radii.mp4".format(input_args['direct_code'],
                                                                  input_args['tree_code'],
                                                                  input_args['virial_radius'],
                                                                  input_args['mass_cut'],
                                                                  str(input_args['flip_split']),
                                                                  input_args['num_bodies'],
                                                                  input_args['timestep'],
                                                                  axlims), writer=writer)
    plt.cla()
    plt.close(fig)


def create_3d_animation(xdata, ydata, zdata, colors, input_args, num_timesteps, scaling_list, axlims=None):
    """
    Creates a single 3d aninmation of a single cluster run from multiple angls
    :param xdata:
    :param ydata:
    :param zdata:
    :param colors:
    :param input_args:
    :param num_timesteps:
    :param scaling_list:
    :param axlims: Limits for the axes, in ([X_low, X_high], [Y_low, Y_high], [Z_low,Z_high]) format
    For the virial radius, it would be ([-3.,3.], [-3.,3.], [-3.,3.])
    :return:
    """

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)

    def update_graph(num):
        """
        Updates the graph's positions
        :param num: The number in the sequence to use
        """
        # Update all the stars
        datax = xdata[num]
        datay = ydata[num]
        dataz = zdata[num]
        graph._offsets3d = (datax, datay, dataz)
        graph1._offsets3d = (datax, datay, dataz)
        graph2._offsets3d = (datax, datay, dataz)
        graph3._offsets3d = (datax, datay, dataz)
        fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                             input_args['tree_code'],
                                                             np.round(num * 0.1, 2)))
        return graph, graph1, graph2, graph3

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.view_init(elev=0, azim=0)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.view_init(elev=0, azim=90)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.view_init(elev=90, azim=0)
    ax.set_zlabel("Z [parsec]")
    ax.set_xlabel("X [parsec]")
    ax.set_ylabel("Y [parsec]")
    ax2.set_zlabel("Z [parsec]")
    ax2.set_ylabel("Y [parsec]")
    ax3.set_zlabel("Z [parsec]")
    ax3.set_xlabel("X [parsec]")
    ax4.set_xlabel("X [parsec]")
    ax4.set_ylabel("Y [parsec]")

    datax = xdata[0]
    datay = ydata[0]
    dataz = zdata[0]
    graph = ax.scatter(datax, datay, dataz, s=50 * scaling_list, c=colors)
    graph1 = ax2.scatter(datax, datay, dataz, s=50 * scaling_list, c=colors)
    graph2 = ax3.scatter(datax, datay, dataz, s=50 * scaling_list, c=colors)
    graph3 = ax4.scatter(datax, datay, dataz, s=50 * scaling_list, c=colors)

    red_patch = mpatches.Patch(color='blue', label='Direct Particles')
    blue_patch = mpatches.Patch(color='red', label='Tree Particles')
    fig.legend(handles=[red_patch, blue_patch])

    for ax in fig.get_axes():
        if axlims is not None:
            if isinstance(axlims[0], list):
                ax.set_xlim3d(axlims[0])
                ax.set_ylim3d(axlims[1])
                ax.set_zlim3d(axlims[2])
            else:
                ax.set_xlim3d(axlims)
                ax.set_ylim3d(axlims)
                ax.set_zlim3d(axlims)

    fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                         input_args['tree_code'],
                                                         0.0))

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, num_timesteps, interval=50, blit=False)
    # plt.show()
    if axlims is None:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}_AllOne.mp4".format(input_args['direct_code'],
                                                 input_args['tree_code'],
                                                 input_args['virial_radius'],
                                                 input_args['mass_cut'],
                                                 str(input_args['flip_split']),
                                                 input_args['num_bodies'],
                                                 input_args['timestep']), writer=writer)
    else:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}_AxLim_{}_AllOne.mp4".format(input_args['direct_code'],
                                                          input_args['tree_code'],
                                                          input_args['virial_radius'],
                                                          input_args['mass_cut'],
                                                          str(input_args['flip_split']),
                                                          input_args['num_bodies'],
                                                          input_args['timestep'],
                                                          axlims), writer=writer)
    plt.cla()
    plt.close(fig)


if __name__ in ('__main__', '__plot__'):
    """
    This code plots the animations
    """
    filenames = [
    ]

    datax, datay, dataz, direct_colors, _, num_timesteps, scaling_list = convert_from_pickle(filenames[0])
    direct_positions = (datax, datay, dataz)

    datax, datay, dataz, tree_colors, _, _, _ = convert_from_pickle(filenames[4])
    tree_positions = (datax, datay, dataz)

    datax, datay, dataz, half_mass_colors, input_args, _, _ = convert_from_pickle(filenames[1])
    half_mass_positions = (datax, datay, dataz)

    datax, datay, dataz, virial_colors, _, _, _ = convert_from_pickle(filenames[2])
    virial_positions = (datax, datay, dataz)

    datax, datay, dataz, core_colors, _, _, _ = convert_from_pickle(filenames[3])
    core_positions = (datax, datay, dataz)

    create_3d_array_radii(direct_positions, tree_positions, virial_positions, core_positions, half_mass_positions,
                          direct_colors, tree_colors,
                          virial_colors, core_colors, half_mass_colors, input_args, num_timesteps,
                          scaling_list=scaling_list,
                          axlims=(-5,5))
