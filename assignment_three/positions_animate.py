import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
from amuse.units import units

# Picked the Hermite and bhtree 100 stars only file
# no mass seperation yet since it isn't relevant until the bridge is right and the converter is correct

filenames = [
    "History_DC_None_TC_bhtree_ClusterMass_6958.0653862278095 MSun_Radius_3.0_Cut_6.0_Flip_False_Stars_10000_Timestep_0.1_EndTime_100.0.p"]


def convert_from_pickle(filename):
    pickleFile = pickle.load(open(filename, 'rb'), fix_imports=True, encoding='latin1')

    dict_data = pickleFile[1]
    input_args = pickleFile[0]

    locations = dict_data['combined_particles_locations']
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

    # Create a Pandas dataframe to store the locations accessing by arbitaray variable time so all the x,y,z locations for all stars for the first time step have the t = 0.0  and next time step t=1.0
    t = np.array([np.ones(len(xdata[0])) * i for i in range(len(xdata))]).flatten()

    return xdata, ydata, zdata, input_args, num_timesteps


def create_2d_animation(df, input_args, num_timesteps, axlims=None, add_extra=None):
    """

    :param df:
    :param input_args:
    :param num_timesteps:
    :param axlims:
    :param add_extra:
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
        data = df[df['time'] == num]
        graph._offsets3d = (data.x, data.y, data.z)  # Have to do this for the
        title.set_text('3D DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                  input_args['tree_code'],
                                                                  np.round(num * 0.1, 2)))
        return title, graph

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    # First one is X,Y
    # Second is Y,Z
    # Third is X,Z
    # Final one is Energy or Half Mass or Something
    # All of them

    data = df[df['time'] == 0]
    ax1.scatter(data.x, data.y)

    ax2.scatter(data.y, data.z)
    ax3.scatter(data.x, data.z)

    ax = fig.add_subplot(111, projection='3d', figsize=(30,20))
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
    title = ax.set_title('3D DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                    input_args['tree_code'],
                                                                    0))

    data = df[df['time'] == 0]
    graph = ax.scatter(data.x, data.y, data.z)

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, num_timesteps, interval=50, blit=False)

    plt.show()
    if axlims is None:
        ani.save("History_2d_DC_{}_TC_{}_"
                 "Radius_{}_Cut_{}_Flip_{}_Stars_{}_"
                 "Timestep_{}.mp4".format(input_args['direct_code'],
                                          input_args['tree_code'],
                                          input_args['virial_radius'],
                                          input_args['mass_cut'],
                                          str(input_args['flip_split']),
                                          input_args['num_bodies'],
                                          input_args['timestep']), writer=writer)
    else:
        ani.save("History_2d_DC_{}_TC_{}_"
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


def create_3d_animation(xdata, ydata, zdata, input_args, num_timesteps, axlims=None):
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
        fig.suptitle('3D DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                  input_args['tree_code'],
                                                                  np.round(num * 0.1, 2)))
        return graph, graph2, graph3, graph4

    fig = plt.figure()
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
    ax2.set_xlabel("X [parsec]")
    ax2.set_ylabel("Y [parsec]")
    ax3.set_zlabel("Z [parsec]")
    ax3.set_xlabel("X [parsec]")
    ax3.set_ylabel("Y [parsec]")
    ax4.set_zlabel("Z [parsec]")
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
    graph = ax.scatter(datax, datay, dataz)
    graph2 = ax2.scatter(datax, datay, dataz)
    graph3 = ax3.scatter(datax, datay, dataz)
    graph4 = ax4.scatter(datax, datay, dataz)

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, num_timesteps, interval=50, blit=False)

    plt.show()
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


for filename in filenames:
    datax, datay, dataz, input_args, num_timesteps = convert_from_pickle(filename)
    create_3d_animation(datax, datay, dataz, input_args, num_timesteps)
