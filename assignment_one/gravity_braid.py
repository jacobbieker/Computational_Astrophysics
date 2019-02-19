###BOOKLISTSTART1###
from amuse.lab import Particles, units
from amuse.lab import nbody_system

def sun_venus_and_earth():
    bodies = Particles(3)
    b1 = bodies[0]
    b1.mass = 1.0 | nbody_system.mass
    #sun.radius = 1.0 |
    b1.position = (855251, -804836, -3186) | nbody_system.length
    b1.velocity = (7.893, 11.894, 0.20642) | (nbody_system.length)/(nbody_system.time)

    b2 = bodies[1]
    b2.mass = 0.5 | nbody_system.mass
    #venus.radius = 3026.0 |
    b2.position = (-0.3767, 0.60159, 0.03930) | nbody_system.length
    b2.velocity = (-29.7725, -18.849, 0.795) | (nbody_system.length)/(nbody_system.time)

    b3 = bodies[2]
    b3.mass = 1.0 | nbody_system.mass
    #earth.radius = 1.0 | units.REarth
    b3.position = (-0.98561, 0.0762, -7.847e-5) | nbody_system.length
    b3.velocity = (-2.927, -29.803, -0.0005327) | (nbody_system.length)/(nbody_system.time)

    bodies.move_to_center()
    return bodies
###BOOKLISTSTOP1###
###BOOKLISTSTART2###
def integrate_solar_system(particles, end_time):
    from amuse.lab import Huayno, nbody_system
    #convert_nbody = nbody_system.nbody_to_si(particles.mass.sum(),particles[1].position.length())
   
    gravity = Huayno()
    gravity.particles.add_particles(bodies)
    b2 = gravity.particles[1]
    b3 = gravity.particles[2]
    
    x_b3 = [] 
    y_b3 = []
    x_b2 = []
    y_b2 = []

    while gravity.model_time < end_time:
        gravity.evolve_model(gravity.model_time + (1 | nbody_system.time))
        x_b3.append(b3.x)
        y_b3.append(b3.y)
        x_b2.append(b2.x)
        y_b2.append(b2.y)
    gravity.stop()
    return x_b3, y_b3, x_b2, y_b2
###BOOKLISTSTOP2###
###BOOKLISTSTART3###
def plot_track(x_b3,y_b3,x_b2,y_b2, output_filename):

    from matplotlib import pyplot
    figure = pyplot.figure(figsize=(10, 10))
    pyplot.rcParams.update({'font.size': 30})
    plot = figure.add_subplot(1,1,1)
    ax = pyplot.gca()
    ax.minorticks_on() 
    ax.locator_params(nbins=3)

    x_label = 'x [length]'
    y_label = 'y [length]'
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    plot.scatter([0.0], [0.0], color='y', lw=8)
    plot.plot(x_b3.value_in(nbody_system.length), y_b3.value_in(nbody_system.length), color = 'b')
    plot.plot(x_b2.value_in(nbody_system.length), y_b2.value_in(nbody_system.length), color = 'r')
    plot.set_xlim(-1.3, 1.3)
    plot.set_ylim(-1.3, 1.3)

    save_file = 'sun_venus_earth.png'
    pyplot.savefig(save_file)
    print '\nSaved figure in file', save_file,'\n'
    pyplot.show()
###BOOKLISTSTOP3###

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-o", 
                      dest="output_filename", default ="SunVenusEarth",
                      help="output filename [%default]")
    return result
    
if __name__ in ('__main__','__plot__'):
    o, arguments  = new_option_parser().parse_args()

    bodies = sun_venus_and_earth()
    x_b3,y_b3, x_b2,y_b2 = integrate_solar_system(bodies, 2 | nbody_system.time)
    plot_track(x_b3, y_b3, x_b2, y_b2, o.output_filename)
    
