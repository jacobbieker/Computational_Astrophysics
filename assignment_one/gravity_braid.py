from amuse.lab import Particles, units
from amuse.lab import nbody_system
import subprocess
import os
import numpy as np


def bodies():
    """
    Generates the bodies for one of the gravity braids
    :return: The 3 Particles for use in the Nbody simulation
    """
    bodies = Particles(3)
    b1 = bodies[0]
    b1.mass = 1.0 | nbody_system.mass  # Defined by Xiaoming et al.
    b1.position = (-1, -0.00, 0.) | nbody_system.length
    b1.velocity = (0.2869236336, 0.0791847624, 0.0) | (nbody_system.length) / (
        nbody_system.time)  # Defined by Xiaoming et al.as (v1,v2)

    b2 = bodies[2]
    b2.mass = 1.0 | nbody_system.mass  # Defined by Xiaoming et al.
    b2.position = (1, 0.00, 0.) | nbody_system.length
    b2.velocity = (0.2869236336, 0.0791847624, 0.0) | (nbody_system.length) / (
        nbody_system.time)  # Defined by Xiaoming et al. to be (v1,v2)

    b3 = bodies[1]
    b3.mass = 0.5 | nbody_system.mass  # Defined by Xiaoming et al.
    b3.position = (0., 0., 0.) | nbody_system.length
    b3.velocity = (-2 * 0.2869236336 / 0.5, -2 * 0.0791847624 / 0.5, 0.0) | (nbody_system.length) / (
        nbody_system.time)  # Defined by Xiaoming et al. ((2*v1)/mass_3, (2*v2)/m3))

    return bodies


def integrate_bodies(bodies, end_time):
    """
    Integrates the nbody program from 0 to the end_time and saves the positions
    :param bodies: The list of 3 Particles to use
    :param end_time: End time in nbody_system.time units
    :return: The x and y positions of each of the three bodies.
    """
    from amuse.lab import Huayno, nbody_system

    gravity = Huayno()
    gravity.particles.add_particles(bodies)
    b1 = gravity.particles[0]
    b2 = gravity.particles[1]
    b3 = gravity.particles[2]

    x_b1 = [] | nbody_system.length
    y_b1 = [] | nbody_system.length
    x_b3 = [] | nbody_system.length
    y_b3 = [] | nbody_system.length
    x_b2 = [] | nbody_system.length
    y_b2 = [] | nbody_system.length

    while gravity.model_time < end_time: # Integrates the system until the given end time
        gravity.evolve_model(gravity.model_time + (0.01 | nbody_system.time))
        x_b1.append(b1.x)
        y_b1.append(b1.y)
        x_b3.append(b3.x)
        y_b3.append(b3.y)
        x_b2.append(b2.x)
        y_b2.append(b2.y)
    gravity.stop()
    return x_b3, y_b3, x_b2, y_b2, x_b1, y_b1


def plot_track(x_b3, y_b3, x_b2, y_b2, x_b1, y_b1, output_filename):
    """
    Plots the motion of the bodies in both a single image and an animation.
    :param x_b3: X positions of body 3
    :param y_b3: Y positions of body 3
    :param x_b2: X positions of body 2
    :param y_b2: Y positions of body 2
    :param x_b1: X positions of body 1
    :param y_b1: Y positions of body 1
    :param output_filename: Output name of the image and of the MP4 animation
    """
    from matplotlib import pyplot
    figure = pyplot.figure(figsize=(10, 10))
    pyplot.rcParams.update({'font.size': 30})
    plot = figure.add_subplot(1, 1, 1)
    ax = pyplot.gca()
    ax.minorticks_on()
    ax.locator_params(nbins=3)

    x_label = 'x [length]'
    y_label = 'y [length]'
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    plot.plot(x_b1.value_in(nbody_system.length), y_b1.value_in(nbody_system.length), color='g')
    plot.scatter(x_b1.value_in(nbody_system.length)[-1], y_b1.value_in(nbody_system.length)[-1], color='g', s=15)
    plot.plot(x_b3.value_in(nbody_system.length), y_b3.value_in(nbody_system.length), color='b')
    plot.scatter(x_b3.value_in(nbody_system.length)[-1], y_b3.value_in(nbody_system.length)[-1], color='b', s=15)
    plot.plot(x_b2.value_in(nbody_system.length), y_b2.value_in(nbody_system.length), color='r')
    plot.scatter(x_b2.value_in(nbody_system.length)[-1], y_b2.value_in(nbody_system.length)[-1], color='r', s=15)
    plot.set_xlim(-1.5, 1.5)
    plot.set_ylim(-1., 1.)

    pyplot.savefig(output_filename + ".png")
    print('\nSaved figure in file', output_filename, '\n')
    pyplot.show()
    pyplot.cla()

    """
    The code below is from the matplotlib demo : https://matplotlib.org/2.1.1/gallery/animation/movie_demo_sgskip.html
    """

    frames = 120 # number of final frames

    steps_per_time = int(len(x_b1) / frames+1)

    # Get the number of timesteps per frame

    time_per_step = (len(x_b1) * 0.01) / frames

    files = []

    fig, ax = pyplot.subplots(figsize=(10, 10))

    for i in range(frames):
        pyplot.cla()
        pyplot.plot(x_b1.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time],
                    y_b1.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time], color='g')
        pyplot.scatter(x_b1.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time][-1],
                       y_b1.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time][-1], color='g',
                       s=15)
        pyplot.plot(x_b1.value_in(nbody_system.length)[:(i + 1) * steps_per_time],
                    y_b1.value_in(nbody_system.length)[:(i + 1) * steps_per_time], color='g', alpha=0.2)
        pyplot.plot(x_b3.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time],
                    y_b3.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time], color='b')
        pyplot.scatter(x_b3.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time][-1],
                       y_b3.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time][-1], color='b',
                       s=15)
        pyplot.plot(x_b3.value_in(nbody_system.length)[:(i + 1) * steps_per_time],
                    y_b3.value_in(nbody_system.length)[:(i + 1) * steps_per_time], color='b', alpha=0.2)
        pyplot.plot(x_b2.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time],
                    y_b2.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time], color='r')
        pyplot.scatter(x_b2.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time][-1],
                       y_b2.value_in(nbody_system.length)[i * steps_per_time:(i + 1) * steps_per_time][-1], color='r',
                       s=15)
        pyplot.plot(x_b2.value_in(nbody_system.length)[:(i + 1) * steps_per_time],
                    y_b2.value_in(nbody_system.length)[:(i + 1) * steps_per_time], color='r', alpha=0.2)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1., 1.)
        ax.set_xlabel("x [length]")
        ax.set_ylabel("y [length]")
        ax.set_title("Timestep: {}".format(np.round(i*time_per_step, 3)))
        fname = '_tmp%04d.png' % i
        pyplot.savefig(fname)
        files.append(fname)

    subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc "
                    "-lavcopts vcodec=wmv2 -oac copy -o {}.mpg".format(output_filename + "_animation"), shell=True)

    # cleanup
    for fname in files:
        os.remove(fname)


def new_option_parser():
    """
    Parses the name of the output file
    :return: The output file name if given, else the default
    """
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-o",
                      dest="output_filename", default="ThreeBodyProblem",
                      help="output filename [%default]")
    return result


if __name__ in ('__main__', '__plot__'):
    o, arguments = new_option_parser().parse_args()

    bodies = bodies()
    # Use the T* from the paper 4.538 for the base time period
    x_b3, y_b3, x_b2, y_b2, x_b1, y_b1 = integrate_bodies(bodies, 4.538*5 | nbody_system.time)
    plot_track(x_b3, y_b3, x_b2, y_b2, x_b1, y_b1, o.output_filename)
