from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation


def create_3d_animation(positions, colors, **kwargs):
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
