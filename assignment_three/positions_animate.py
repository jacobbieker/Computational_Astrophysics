import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
from amuse.units import units
from amuse.io import read_set_from_file
import os

# Picked the Hermite and bhtree 100 stars only file
# no mass seperation yet since it isn't relevant until the bridge is right and the converter is correct

filenames = [
"/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/History_DC_ph4_TC_bhtree_ClusterMass_6958.0653862278095 MSun_Radius_3.0_Cut_0.5_Flip_True_Stars_10000_Timestep_0.1_EndTime_100.0.p",
"/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/History_DC_ph4_TC_bhtree_ClusterMass_6958.0653862278095 MSun_Radius_3.0_Cut_2.0_Flip_False_Stars_10000_Timestep_0.1_EndTime_100.0.p",
"/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/History_DC_ph4_TC_bhtree_ClusterMass_6958.0653862278095 MSun_Radius_3.0_Cut_2.0_Flip_True_Stars_10000_Timestep_0.1_EndTime_100.0.p",
"/home/jacob/Development/comp_astro/assignment_three/STRW_Comp/History_DC_ph4_TC_bhtree_ClusterMass_6958.0653862278095 MSun_Radius_3.0_Cut_4.0_Flip_True_Stars_10000_Timestep_0.1_EndTime_100.0.p",
]

import glob
filenames = []
for file in glob.glob("STRW_Comp/*100.0.p"):
    filenames.append(file)


def convert_from_pickle(filename):
    pickleFile = pickle.load(open(filename, 'rb'), fix_imports=True, encoding='latin1')

    hdf_filename = os.path.splitext(filename)[0] + ".hdf"

    #snapshots = read_set_from_file(hdf_filename, "amuse")

    #virial_radius = snapshots.history[0].virial_radius()

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


    colors = []
    if input_args['tree_code'] is not None and input_args['direct_code'] is not None:
        if method == 'mass':
            for particle in masses:
                if particle >= cut.value_in(units.MSun) :
                    if dict_data['flip_split']:
                        colors.append(1)
                    else:
                        colors.append(0)
                else:
                    if dict_data['flip_split']:
                        colors.append(0)
                    else:
                        colors.append(1)
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


    return xdata, ydata, zdata, colors, input_args, num_timesteps


def create_3d_animation_array(xdata, ydata, zdata, colors, input_args, num_timesteps, axlims=None):
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
    cdict = {1: 'red', 0: 'blue'}
    gdict = {1: 'direct', 0:'tree'}

    if len(colors) < len(xdata):
        colors = 'b'

    def update_graph(num):
        """
        Updates the graph's positions
        :param num: The number in the sequence to use
        """
        # Update all the stars
        datax = xdata[num]
        datay = ydata[num]
        dataz = zdata[num]
        for g in np.unique(colors):
            ix = np.where(colors == g)
            graph._offsets3d(datax[ix], datay[ix], dataz[ix], c=cdict[g], label=gdict[g])
            graph2._offsets3d(datax[ix], datay[ix], dataz[ix], c=cdict[g])
            graph3._offsets3d(datax[ix], datay[ix], dataz[ix], c=cdict[g])
            graph4._offsets3d(datax[ix], datay[ix], dataz[ix], c=cdict[g])
        fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                 input_args['tree_code'],
                                                                 np.round(num * 0.1, 2)))
        fig.figlegend(loc='best')
        return graph, graph2, graph3, graph4

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

    if axlims is not None:
        if isinstance(axlims[0], list):
            ax.set_xlim3d(axlims[0])
            ax.set_ylim3d(axlims[1])
            ax.set_zlim3d(axlims[2])
        else:
            ax.set_xlim3d(axlims)
            ax.set_ylim3d(axlims)
            ax.set_zlim3d(axlims)

    datax = xdata[0]
    datay = ydata[0]
    dataz = zdata[0]
    for g in np.unique(colors):
        ix = np.where(colors == g)
        graph = ax.scatter(datax[ix], datay[ix], dataz[ix], c=cdict[g], label=gdict[g])
        graph2 = ax2.scatter(datax[ix], datay[ix], dataz[ix], c=cdict[g])
        graph3 = ax3.scatter(datax[ix], datay[ix], dataz[ix], c=cdict[g])
        graph4 = ax4.scatter(datax[ix], datay[ix], dataz[ix], c=cdict[g])
    fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                         input_args['tree_code'],
                                                         np.round(0.0, 2)))
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
def create_3d_animation(xdata, ydata, zdata, colors, input_args, num_timesteps, axlims=None):
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
    if len(colors) < len(xdata):
        colors = 'b'

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
        graph2._offsets3d = (datax, datay, dataz)
        graph3._offsets3d = (datax, datay, dataz)
        graph4._offsets3d = (datax, datay, dataz)
        fig.suptitle('DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                  input_args['tree_code'],
                                                                  np.round(num * 0.1, 2)))
        return graph, graph2, graph3, graph4

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
    #ax2.set_xlabel("X [parsec]")
    ax2.set_ylabel("Y [parsec]")
    ax3.set_zlabel("Z [parsec]")
    ax3.set_xlabel("X [parsec]")
    #ax3.set_ylabel("Y [parsec]")
    #ax4.set_zlabel("Z [parsec]")
    ax4.set_xlabel("X [parsec]")
    ax4.set_ylabel("Y [parsec]")

    if axlims is not None:
        if isinstance(axlims[0], list):
            ax.set_xlim3d(axlims[0])
            ax.set_ylim3d(axlims[1])
            ax.set_zlim3d(axlims[2])
        else:
            ax.set_xlim3d(axlims)
            ax.set_ylim3d(axlims)
            ax.set_zlim3d(axlims)

    datax = xdata[0]
    datay = ydata[0]
    dataz = zdata[0]
    graph = ax.scatter(datax, datay, dataz, c=colors)
    graph2 = ax2.scatter(datax, datay, dataz, c=colors)
    graph3 = ax3.scatter(datax, datay, dataz, c=colors)
    graph4 = ax4.scatter(datax, datay, dataz, c=colors)

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


for filename in filenames:
    datax, datay, dataz, colors, input_args, num_timesteps = convert_from_pickle(filename)
    create_3d_animation(datax, datay, dataz, colors, input_args, num_timesteps)
