import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd

# Picked the Hermite and bhtree 100 stars only file
# no mass seperation yet since it isn't relevant until the bridge is right and the converter is correct

filenames = [
    "History_DC_ph4_TC_None_ClusterMass_132.12625693121714 MSun_Radius_3.0_Cut_6.0_Flip_False_Stars_100_Timestep_0.1.p",
    "History_DC_None_TC_bhtree_ClusterMass_65.21805817083936 MSun_Radius_3.0_Cut_6.0_Flip_False_Stars_100_Timestep_0.1.p",
    "History_DC_None_TC_bhtree_ClusterMass_100000.0 MSun_Radius_3.0_Cut_6.0_Flip_False_Stars_10000_Timestep_0.1.p",
    "/home/jacob/Development/comp_astro/assignment_three/History_DC_hermite_TC_bhtree_ClusterMass_800.0 MSun_Radius_3.0_Cut_6.0_Flip_False_Stars_1000_Timestep_0.1.p"]


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

    core_radii_data = dict_data['core_radius_history']
    half_mass_data = dict_data['half_mass_history']
    num_timesteps = len(xdata)

    # Create a Pandas dataframe to store the locations accessing by arbitaray variable time so all the x,y,z locations for all stars for the first time step have the t = 0.0  and next time step t=1.0
    t = np.array([np.ones(len(xdata[0])) * i for i in range(len(xdata))]).flatten()
    df = pd.DataFrame()
    df["x"] = ""
    df["y"] = ""
    df["z"] = ""
    df['core_radii'] = ""
    df['half_mass'] = ""

    # Definitely not the most effcient way but it is working

    for j in range(len(xdata)):
        xlist = xdata[j]
        ylist = ydata[j]
        zlist = zdata[j]
        core_r_list = core_radii_data[j]
        half_mass_list = half_mass_data[j]
        for i in range(len(xdata[0])):
            df = df.append({'x': xlist[i], 'y': ylist[i], 'z': zlist[i],}, ignore_index=True)
    df["time"] = t

    return df, input_args, num_timesteps


def create_animation(df, input_args, num_timesteps, axlims=None, add_extra=None):
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
        data = df[df['time'] == num]
        graph._offsets3d = (data.x, data.y, data.z)  # Have to do this for the
        title.set_text('3D DC: {} TC: {} Sim Time: {} Myr'.format(input_args['direct_code'],
                                                                  input_args['tree_code'],
                                                                  np.round(num * 0.1, 2)))
        return title, graph

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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
    df, input_args, num_timesteps = convert_from_pickle(filename)
    create_animation(df, input_args, num_timesteps)
    create_animation(df, input_args, num_timesteps, axlims=([-3., 3.], [-3., 3.], [-3., 3.]))
