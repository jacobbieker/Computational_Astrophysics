from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
from amuse.units import units
from amuse.io import read_set_from_file
import os
from amuse.units import nbody_system
from amuse.datamodel import Particle, Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution

# Picked the Hermite and bhtree 100 stars only file
# no mass seperation yet since it isn't relevant until the bridge is right and the converter is correct

filenames = [
"/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/History_DC_ph4_TC_bhtree_ClusterMass_6958.0653862278095 MSun_Radius_3.0_Cut_0.5_Flip_True_Stars_10000_Timestep_0.1_EndTime_100.0.p",
"/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/History_DC_ph4_TC_bhtree_ClusterMass_6958.0653862278095 MSun_Radius_3.0_Cut_2.0_Flip_False_Stars_10000_Timestep_0.1_EndTime_100.0.p",
"/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/History_DC_ph4_TC_bhtree_ClusterMass_6958.0653862278095 MSun_Radius_3.0_Cut_2.0_Flip_True_Stars_10000_Timestep_0.1_EndTime_100.0.p",
"/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/History_DC_ph4_TC_bhtree_ClusterMass_6958.0653862278095 MSun_Radius_3.0_Cut_4.0_Flip_True_Stars_10000_Timestep_0.1_EndTime_100.0.p",
]

#import glob
#filenames = []
#for file in glob.glob("STRW_Comp/*100.0.p"):
#    filenames.append(file)


def convert_from_pickle(filename):
    pickleFile = pickle.load(open(filename, 'rb'), fix_imports=True, encoding='latin1')

    hdf_filename = os.path.splitext(filename)[0] + ".hdf"

    try:
        raise NotImplementedError
        snapshots = read_set_from_file(hdf_filename, "amuse")
        np.random.seed(5227) # Set for reproducability
        mZAMS = new_powerlaw_mass_distribution(10000, 0.1 | units.MSun, 100 | units.MSun, alpha=-2.0)
        cluster_mass = mZAMS.sum() # (args['num_bodies']) | units.MSun
        converter = nbody_system.nbody_to_si(cluster_mass, 3 | units.parsec)
        center_of_mass = snapshots.history[0].center_of_mass()
        virial_radius = snapshots.history[0].virial_radius()
        _, core_radius, _ = snapshots.history[0].densitycentre_coreradius_coredens(
            unit_converter=converter)
        total_radius = snapshots.history[0].LagrangianRadii(mf=[0.5],
                                                            cm=center_of_mass,
                                                            unit_converter=converter)[0][0]
    except:
        print("Can't Open File")

    dict_data = pickleFile[1]
    input_args = pickleFile[0]

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

    if method == 'virial_radius':
        raise NotImplementedError
    elif method == "half_mass":
        raise NotImplementedError

    # Get the scaling for the points
    scaling_list = []
    for mass in masses:
        scaling_list.append(mass/np.max(masses))

    scaling_list = np.asarray(scaling_list)

    colors = []
    if input_args['tree_code'] is not None and input_args['direct_code'] is not None:
        if method == 'mass':
            for particle in masses:
                if particle >= cut.value_in(units.MSun):
                    if dict_data['flip_split']:
                        colors.append('b')
                    else:
                        colors.append('r')
                else:
                    if dict_data['flip_split']:
                        colors.append('r')
                    else:
                        colors.append('b')
    elif input_args['tree_code'] is None and input_args['direct_code'] is not None:
        colors = np.asarray(['b' for i in range(len(scaling_list))])
    elif input_args['tree_code'] is not None and input_args['direct_code'] is None:
        colors = np.asarray(['r' for i in range(len(scaling_list))])

        '''
        elif method == 'core_radius':
            _, core_radius, _ = particles.densitycentre_coreradius_coredens(
                unit_converter=converter)
            for particle in particles:
                if np.sqrt((particle.x - particles.center_of_mass().x) ** 2 + (
                        particle.y - particles.center_of_mass().y) ** 2 + (
                                   particle.z - particles.center_of_mass().z) ** 2) <= args['mass_cut']*core_radius:
                    if args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        elif method == 'half_mass':
            half_mass_radius = \
                particles.LagrangianRadii(mf=[0.5], cm=particles.center_of_mass(),
                                          unit_converter=converter)[0][0]
            for particle in particles:
                if np.sqrt((particle.x - particles.center_of_mass().x) ** 2 + (
                        particle.y - particles.center_of_mass().y) ** 2 + (
                                   particle.z - particles.center_of_mass().z) ** 2) <= args['mass_cut']*half_mass_radius:
                    if args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        elif method == 'virial_radius':
            virial_radius = particles.virial_radius()
            for particle in particles:
                if np.sqrt((particle.x - particles.center_of_mass().x) ** 2 + (
                        particle.y - particles.center_of_mass().y) ** 2 + (
                                   particle.z - particles.center_of_mass().z) ** 2) <= args['mass_cut']*virial_radius:
                    if args['flip_split']:
                        tree_particles.add_particle(particle)
                    else:
                        direct_particles.add_particle(particle)
                else:
                    if args['flip_split']:
                        direct_particles.add_particle(particle)
                    else:
                        tree_particles.add_particle(particle)
        '''


    return xdata, ydata, zdata, colors, input_args, num_timesteps, scaling_list


def create_3d_animation_array(direct_positions, tree_positions, false_positions, true_positions, direct_colors, tree_colors, false_colors, true_colors, input_args, num_timesteps, scaling_list, title=None, axlims=None):
    """
        Create the animation of the stars over time, optionally with limits
        Designed for seeing how flipping and direct and tree codes differ
        Takes 4 sets
        :param dataframe:
        :param num_timesteps:
        :param axlims: Limits for the axes, in ([X_low, X_high], [Y_low, Y_high], [Z_low,Z_high]) format
        For the virial radius, it would be ([-3.,3.], [-3.,3.], [-3.,3.])
        :return:
        """

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    cdict = {1: 'r', 0: 'b'}
    gdict = {1: 'direct', 0:'tree'}

    def update_graph(num):
        """
        Updates the graph's positions
        :param num: The number in the sequence to use
        """
        # Update all the stars
        for i in range(16):
            if i+1 in [1,2,3,4]:
                ax._offsets3d = (direct_positions[0][num], direct_positions[1][num], direct_positions[2][num])
            elif i+1 in [5,6,7,8]:
                ax._offsets3d = (false_positions[0][num], false_positions[1][num], false_positions[2][num])
            elif i+1 in [9,10,11,12]:
                ax._offsets3d = (true_positions[0][num], true_positions[1][num], true_positions[2][num])
            elif i+1 in [13,14,15,16]:
                ax._offsets3d = (tree_positions[0][num], tree_positions[1][num], tree_positions[2][num])
        fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                 input_args['tree_code'],
                                                                 np.round(num * 0.1, 2)))
        return graphs

    fig = plt.figure(figsize=(20,20))
    for i in range(16):
        if i+1 in [2,6,10,14]:
            ax = fig.add_subplot(4,4,i+1, projection='3d')
            ax.view_init(elev=0, azim=0)
            ax.set_zlabel("Z [parsec]")
            ax.set_ylabel("Y [parsec]")
            if axlims is not None:
                if isinstance(axlims[0], list):
                    ax.set_xlim3d(axlims[0])
                    ax.set_ylim3d(axlims[1])
                    ax.set_zlim3d(axlims[2])
                else:
                    ax.set_xlim3d(axlims)
                    ax.set_ylim3d(axlims)
                    ax.set_zlim3d(axlims)
        elif i+1 in [3,7,11,15]:
            ax = fig.add_subplot(4,4,i+1, projection='3d')
            ax.view_init(elev=0, azim=90)
            ax.set_zlabel("Z [parsec]")
            ax.set_xlabel("X [parsec]")
            if axlims is not None:
                if isinstance(axlims[0], list):
                    ax.set_xlim3d(axlims[0])
                    ax.set_ylim3d(axlims[1])
                    ax.set_zlim3d(axlims[2])
                else:
                    ax.set_xlim3d(axlims)
                    ax.set_ylim3d(axlims)
                    ax.set_zlim3d(axlims)
        elif i+1 in [4,8,12,16]:
            ax = fig.add_subplot(4,4,i+1, projection='3d')
            ax.view_init(elev=90, azim=0)
            ax.set_xlabel("X [parsec]")
            ax.set_ylabel("Y [parsec]")
            if axlims is not None:
                if isinstance(axlims[0], list):
                    ax.set_xlim3d(axlims[0])
                    ax.set_ylim3d(axlims[1])
                    ax.set_zlim3d(axlims[2])
                else:
                    ax.set_xlim3d(axlims)
                    ax.set_ylim3d(axlims)
                    ax.set_zlim3d(axlims)
        elif i+1 in [1,5,9,13]:
            ax = fig.add_subplot(4,4,i+1, projection='3d')
            ax.set_zlabel("Z [parsec]")
            ax.set_xlabel("X [parsec]")
            ax.set_ylabel("Y [parsec]")
            if axlims is not None:
                if isinstance(axlims[0], list):
                    ax.set_xlim3d(axlims[0])
                    ax.set_ylim3d(axlims[1])
                    ax.set_zlim3d(axlims[2])
                else:
                    ax.set_xlim3d(axlims)
                    ax.set_ylim3d(axlims)
                    ax.set_zlim3d(axlims)

    graphs = []
    allaxes = fig.get_axes()
    print(len(allaxes))
    labels = []
    for i in range(16):
        ax = allaxes[i]
        if i+1 in [1,2,3,4]:
            ax.scatter(direct_positions[0][0], direct_positions[1][0], direct_positions[2][0],s=50*scaling_list, c='b')
        elif i+1 in [5,6,7,8]:
            for g in np.unique(false_colors):
                false_ix = np.where(false_colors == g)
                one_label = ax.scatter(false_positions[0][0][false_ix], false_positions[1][0][false_ix], false_positions[2][0][false_ix], s=50*scaling_list, c=cdict[g], label=gdict[g])
                labels.append(one_label)
        elif i+1 in [9,10,11,12]:
            for g in np.unique(true_colors):
                true_ix = np.where(true_colors == g)
                ax.scatter(true_positions[0][0][true_ix], true_positions[1][0][true_ix], true_positions[2][0][true_ix],s=50*scaling_list, c=cdict[g])
        elif i+1 in [13,14,15,16]:
            ax.scatter(tree_positions[0][0], tree_positions[1][0], tree_positions[2][0],s=50*scaling_list, c='r')
    fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                         input_args['tree_code'],
                                                         np.round(0.0, 2)))
    fig.legend((labels[0], labels[1]), ('Direct', 'Tree'), 'upper left')
    #plt.show()
    #exit()
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, num_timesteps, interval=50, blit=False)
    if axlims is None:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}_All4_Titled.mp4".format(input_args['direct_code'],
                                          input_args['tree_code'],
                                          input_args['virial_radius'],
                                          input_args['mass_cut'],
                                          str(input_args['flip_split']),
                                          input_args['num_bodies'],
                                          input_args['timestep']), writer=writer)
    else:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}_AxLim_{}_All4.mp4".format(input_args['direct_code'],
                                                   input_args['tree_code'],
                                                   input_args['virial_radius'],
                                                   input_args['mass_cut'],
                                                   str(input_args['flip_split']),
                                                   input_args['num_bodies'],
                                                   input_args['timestep'],
                                                   axlims), writer=writer)
    plt.cla()
    plt.close(fig)

def create_3d_array(direct_positions, tree_positions, false_positions, true_positions, direct_colors, tree_colors, false_colors, true_colors, input_args, num_timesteps, scaling_list, title=None, axlims=None):
    """
    This creates a 1 x 4 showing the different effects that the cuts have on the animation
    :param direct_positions:
    :param tree_positions:
    :param false_positions:
    :param true_positions:
    :param false_colors:
    :param true_colors:
    :param input_args:
    :param num_timesteps:
    :param scaling_list:
    :param title:
    :param axlims:
    :return:
    """


    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)

    cdict = {1: 'r', 0: 'b'}
    gdict = {1: 'Direct', 0:'Tree'}

    def update_graph(num):
        """
        Updates the graph's positions
        :param num: The number in the sequence to use
        """
        # Update all the stars
        graph._offsets3d = (direct_positions[0][num], direct_positions[1][num], direct_positions[2][num])  # Have to do this for the
        graph1._offsets3d = (false_positions[0][num], false_positions[1][num], false_positions[2][num])  # Have to do this for the
        graph2._offsets3d = (true_positions[0][num], true_positions[1][num], true_positions[2][num])  # Have to do this for the
        graph3._offsets3d = (tree_positions[0][num], tree_positions[1][num], tree_positions[2][num])  # Have to do this for the
        fig.suptitle('DC: {} TC: {} Cut: {} Method: {} \n Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                               input_args['tree_code'], input_args['mass_cut'], method,
                                                             np.round(num * 0.1, 2)))
        return graph, graph1, graph2, graph3

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(141, projection='3d')
    ax2 = fig.add_subplot(142, projection='3d')
    ax3 = fig.add_subplot(143, projection='3d')
    ax4 = fig.add_subplot(144, projection='3d')
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


    graph = ax.scatter(direct_positions[0][0], direct_positions[1][0], direct_positions[2][0], s=50*scaling_list, c=direct_colors)
    graph3 = ax4.scatter(tree_positions[0][0], tree_positions[1][0], tree_positions[2][0], s=50*scaling_list, c=tree_colors)
    graph1 = ax2.scatter(false_positions[0][0], false_positions[1][0], false_positions[2][0], s=50*scaling_list, c=false_colors, label=gdict)
    graph2 = ax3.scatter(true_positions[0][0], true_positions[1][0], true_positions[2][0], s=50*scaling_list, c=true_colors)

    import matplotlib.patches as mpatches
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
                                                         input_args['tree_code'], input_args['mass_cut'], method,
                                                         0.0))

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, num_timesteps, interval=50, blit=False)
    #plt.show()
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


def create_3d_2d_animation_array(direct_positions, tree_positions, false_positions, true_positions, false_colors, true_colors, input_args, num_timesteps, scaling_list, axlims=None):
    """
        Create the animation of the stars over time, optionally with limits
        Designed for seeing how flipping and direct and tree codes differ
        Takes 4 sets
        :param dataframe:
        :param num_timesteps:
        :param axlims: Limits for the axes, in ([X_low, X_high], [Y_low, Y_high], [Z_low,Z_high]) format
        For the virial radius, it would be ([-3.,3.], [-3.,3.], [-3.,3.])
        :return:
        """

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    cdict = {1: 'r', 0: 'b'}
    gdict = {1: 'direct', 0:'tree'}

    def update_graph(num):
        """
        Updates the graph's positions
        :param num: The number in the sequence to use
        """
        # Update all the stars
        for i in range(16):
            if i+1 in [1,2,3,4]:
                ax._offsets3d = (direct_positions[0][num], direct_positions[1][num], direct_positions[2][num])
            elif i+1 in [5,6,7,8]:
                for g in np.unique(colors):
                    true_ix = np.where(true_colors == g)
                    false_ix = np.where(false_colors == g)
                ax._offsets3d = (false_positions[0][num], false_positions[1][num], false_positions[2][num])
            elif i+1 in [9,10,11,12]:
                for g in np.unique(colors):
                    true_ix = np.where(true_colors == g)
                    false_ix = np.where(false_colors == g)
                ax._offsets3d = (true_positions[0][num], true_positions[1][num], true_positions[2][num])
            elif i+1 in [13,14,15,16]:
                ax._offsets3d = (tree_positions[0][num], tree_positions[1][num], tree_positions[2][num])
        fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                             input_args['tree_code'],
                                                             np.round(num * 0.1, 2)))
        return graphs

    fig = plt.figure(figsize=(20,20))
    for i in range(16):
        if i+1 in [2,6,10,14]:
            ax = fig.add_subplot(4,4,i+1)
            ax.set_xlabel("Z [parsec]")
            ax.set_ylabel("Y [parsec]")
        elif i+1 in [3,7,11,15]:
            ax = fig.add_subplot(4,4,i+1)
            ax.set_ylabel("Z [parsec]")
            ax.set_xlabel("X [parsec]")
        elif i+1 in [4,8,12,16]:
            ax = fig.add_subplot(4,4,i+1)
            ax.set_xlabel("X [parsec]")
            ax.set_ylabel("Y [parsec]")
        elif i+1 in [1,5,9,13]:
            ax = fig.add_subplot(4,4,i+1, projection='3d')
            ax.set_zlabel("Z [parsec]")
            ax.set_xlabel("X [parsec]")
            ax.set_ylabel("Y [parsec]")

    if axlims is not None:
        if isinstance(axlims[0], list):
            ax.set_xlim3d(axlims[0])
            ax.set_ylim3d(axlims[1])
            ax.set_zlim3d(axlims[2])
        else:
            ax.set_xlim3d(axlims)
            ax.set_ylim3d(axlims)
            ax.set_zlim3d(axlims)

    graphs = []
    allaxes = fig.get_axes()
    print(len(allaxes))
    for i in range(16):
        if i+1 == 1:
            allaxes[i+1].scatter(direct_positions[0][0], direct_positions[1][0], direct_positions[2][0], c='b')
            # XY
            allaxes[i+2].scatter(direct_positions[0][0], direct_positions[1][0], s=50*scaling_list, c='b')
            # XZ
            allaxes[i+3].scatter(direct_positions[0][0], direct_positions[2][0],s=50*scaling_list, c='b')
            # YZ
            allaxes[i+4].scatter(direct_positions[1][0], direct_positions[2][0],s=50*scaling_list, c='b')
        elif i+1 == 5:
            for g in np.unique(false_colors):
                true_ix = np.where(false_colors == g)
                allaxes[i+1].scatter(true_positions[0][0][true_ix], true_positions[1][0][true_ix], true_positions[2][0][true_ix], c=cdict[g])
                # XY
                allaxes[i+2].scatter(true_positions[0][0][true_ix], true_positions[1][0][true_ix],s=50*scaling_list, c=cdict[g])
                # XZ
                allaxes[i+3].scatter(true_positions[0][0][true_ix], true_positions[2][0][true_ix],s=50*scaling_list, c=cdict[g])
                # YZ
                allaxes[i+4].scatter(true_positions[1][0][true_ix], true_positions[2][0][true_ix],s=50*scaling_list, c=cdict[g])
        elif i+1 == 9:
            for g in np.unique(true_colors):
                true_ix = np.where(true_colors == g)
                allaxes[i+1].scatter(true_positions[0][0][true_ix], true_positions[1][0][true_ix], true_positions[2][0][true_ix], c=cdict[g])
                # XY
                allaxes[i+2].scatter(true_positions[0][0][true_ix], true_positions[1][0][true_ix],s=50*scaling_list, c=cdict[g])
                # XZ
                allaxes[i+3].scatter(true_positions[0][0][true_ix], true_positions[2][0][true_ix],s=50*scaling_list, c=cdict[g])
                # YZ
                allaxes[i+4].scatter(true_positions[1][0][true_ix], true_positions[2][0][true_ix],s=50*scaling_list, c=cdict[g])
        elif i+1 == 12:
            allaxes[i+1].scatter(tree_positions[0][0], tree_positions[1][0], tree_positions[2][0], c='r')
            # XY
            allaxes[i+2].scatter(tree_positions[0][0], tree_positions[1][0],s=50*scaling_list, c='r')
            # XZ
            allaxes[i+3].scatter(tree_positions[0][0], tree_positions[2][0],s=50*scaling_list, c='r')
            # YZ
            allaxes[i+4].scatter(tree_positions[1][0], tree_positions[2][0],s=50*scaling_list, c='r')
    fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                         input_args['tree_code'],
                                                         np.round(0.0, 2)))
    ax.legend(loc='best')
    plt.show()
    exit()
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, num_timesteps, interval=50, blit=False)
    if axlims is None:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}.mp4".format(input_args['direct_code'],
                                          input_args['tree_code'],
                                          input_args['virial_radius'],
                                          input_args['mass_cut'],
                                          str(input_args['flip_split']),
                                          input_args['num_bodies'],
                                          input_args['timestep']), writer=writer)
    else:
        ani.save("History_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}_AxLim_{}.mp4".format(input_args['direct_code'],
                                                   input_args['tree_code'],
                                                   input_args['virial_radius'],
                                                   input_args['mass_cut'],
                                                   str(input_args['flip_split']),
                                                   input_args['num_bodies'],
                                                   input_args['timestep'],
                                                   axlims), writer=writer)
    plt.cla()
    plt.close(fig)

def create_3d_animation(xdata, ydata, zdata, colors, input_args, num_timesteps, scaling_list, title=None, axlims=None):
    """
    Create the animation of the stars over time, optionally with limits
    :param dataframe:
    :param num_timesteps:
    :param axlims: Limits for the axes, in ([X_low, X_high], [Y_low, Y_high], [Z_low,Z_high]) format
    For the virial radius, it would be ([-3.,3.], [-3.,3.], [-3.,3.])
    :return:
    """

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)

    cdict = {1: 'r', 0: 'b'}
    gdict = {1: 'Direct', 0:'Tree'}

    def update_graph(num):
        """
        Updates the graph's positions
        :param num: The number in the sequence to use
        """
        # Update all the stars
        datax = xdata[num]
        datay = ydata[num]
        dataz = zdata[num]
        graph._offsets3d = (datax, datay, dataz)  # Have to do this for the
        graph1._offsets3d = (datax, datay, dataz)  # Have to do this for the
        graph2._offsets3d = (datax, datay, dataz)  # Have to do this for the
        graph3._offsets3d = (datax, datay, dataz)  # Have to do this for the
        fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                  input_args['tree_code'],
                                                                  np.round(num * 0.1, 2)))
        return graph, graph1, graph2, graph3

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(141, projection='3d')
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.view_init(elev=0, azim=0)
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.view_init(elev=0, azim=90)
    ax4 = fig.add_subplot(144, projection='3d')
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
    labels = []
    for g in np.unique(colors):
        ix = np.where(colors == g)
        graph = ax.scatter(datax[ix], datay[ix], dataz[ix], s=50*scaling_list, c=cdict[g], label=gdict[g])
        labels.append(graph)
        graph1 = ax2.scatter(datax[ix], datay[ix], dataz[ix], s=50*scaling_list, c=cdict[g])
        graph2 = ax3.scatter(datax[ix], datay[ix], dataz[ix], s=50*scaling_list, c=cdict[g])
        graph3 = ax4.scatter(datax[ix], datay[ix], dataz[ix], s=50*scaling_list, c=cdict[g])

    if len(labels) >= 2:
        fig.legend((labels[0], labels[1]), ('Direct', 'Tree'), 'upper left')
    else:
        for g in np.unique(colors):
            fig.legend((labels[0],), (gdict[g]), 'upper left')

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
    #plt.show()
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


filenames = [
    "/home/jacob/Development/comp_astro/assignment_three/Base_Test/Checkpoint_DC_None_TC_bhtree_ClusterMass_6958.065386227829_Radius_3.0_Cut_6.0_Flip_False_Stars_10000_Timestep_0.1_EndTime_100.0.p",
    "/home/jacob/Development/comp_astro/assignment_three/Base_Test/Checkpoint_DC_ph4_TC_bhtree_ClusterMass_6958.06538623_Radius_3.0_Cut_2.0_Flip_False_Stars_10000_Timestep_0.1_EndTime_100.0.p",
    "/home/jacob/Development/comp_astro/assignment_three/Base_Test/Checkpoint_DC_ph4_TC_bhtree_ClusterMass_6958.06538623_Radius_3.0_Cut_2.0_Flip_True_Stars_10000_Timestep_0.1_EndTime_100.0.p",
    "/home/jacob/Development/comp_astro/assignment_three/Base_Test/Checkpoint_DC_ph4_TC_None_ClusterMass_6958.065386227829_Radius_3.0_Cut_6.0_Flip_False_Stars_10000_Timestep_0.1_EndTime_100.0.p",
]

datax, datay, dataz, direct_colors, input_args, num_timesteps, scaling_list = convert_from_pickle(filenames[0])
direct_positions = (datax, datay, dataz)
#create_3d_animation(datax, datay, dataz, colors, input_args, num_timesteps, scaling_list, axlims=(-5.,5.))
#create_3d_animation(datax, datay, dataz, colors, input_args, num_timesteps, scaling_list)

datax, datay, dataz, tree_colors, input_args, num_timesteps, scaling_list = convert_from_pickle(filenames[3])
tree_positions = (datax, datay, dataz)
#create_3d_animation(datax, datay, dataz, colors, input_args, num_timesteps, scaling_list)
#create_3d_animation(datax, datay, dataz, colors, input_args, num_timesteps, scaling_list, axlims=(-10.,10.))

datax, datay, dataz, false_colors, input_args, num_timesteps, scaling_list = convert_from_pickle(filenames[1])
false_positions = (datax, datay, dataz)
#create_3d_animation(datax, datay, dataz, colors, input_args, num_timesteps, scaling_list)
#create_3d_animation(datax, datay, dataz, colors, input_args, num_timesteps, scaling_list, axlims=(-10.,10.))

datax, datay, dataz, true_colors, input_args, num_timesteps, scaling_list = convert_from_pickle(filenames[2])
true_positions = (datax, datay, dataz)
#create_3d_animation(datax, datay, dataz, colors, input_args, num_timesteps, scaling_list)
#create_3d_animation(datax, datay, dataz, colors, input_args, num_timesteps, scaling_list, axlims=(-10.,10.))


create_3d_array(direct_positions, tree_positions, false_positions, tree_positions, direct_colors, tree_colors, false_colors, true_colors, input_args, num_timesteps, scaling_list=scaling_list, axlims=(-3.,3.))

