from amuse.lab import Particles, units
from amuse.lab import nbody_system
import subprocess
import os


def bodies():
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

    while gravity.model_time < end_time:
        gravity.evolve_model(gravity.model_time + (0.01 | nbody_system.time))
        x_b1.append(b1.x)
        y_b1.append(b1.y)
        x_b3.append(b3.x)
        y_b3.append(b3.y)
        x_b2.append(b2.x)
        y_b2.append(b2.y)
    gravity.stop()
    return x_b3, y_b3, x_b2, y_b2, x_b1, y_b1


###BOOKLISTSTOP2###
###BOOKLISTSTART3###
def plot_track(x_b3, y_b3, x_b2, y_b2, x_b1, y_b1, output_filename):
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

    save_file = 'Three_body_problem.png'
    pyplot.savefig(save_file)
    print('\nSaved figure in file', save_file, '\n')
    pyplot.show()
    pyplot.cla()

    steps_per_time = int(len(x_b1) / 121)

    files = []

    fig, ax = pyplot.subplots(figsize=(10, 10))
    for i in range(120):  # 120 frames
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
        fname = '_tmp%03d.png' % i
        print('Saving frame', fname)
        pyplot.savefig(fname)
        files.append(fname)

    print('Making movie animation.mpg - this may take a while')
    subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc "
                    "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg", shell=True)

    # cleanup
    for fname in files:
        os.remove(fname)


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-o",
                      dest="output_filename", default="ThreeBodyProblem",
                      help="output filename [%default]")
    return result


if __name__ in ('__main__', '__plot__'):
    o, arguments = new_option_parser().parse_args()

    bodies = bodies()
    x_b3, y_b3, x_b2, y_b2, x_b1, y_b1 = integrate_bodies(bodies, 4.538*5 | nbody_system.time)
    plot_track(x_b3, y_b3, x_b2, y_b2, x_b1, y_b1, o.output_filename)
