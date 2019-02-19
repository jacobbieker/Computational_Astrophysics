from amuse.lab import Particles, units
from amuse.lab import nbody_system

def bodies():
  
    
    bodies = Particles(3)
    b1 = bodies[0]
    b1.mass = 1.0 | nbody_system.mass #Defined by Xiaoming et al.
    b1.position = (-1,0.001,0) | nbody_system.length
    b1.velocity = (0.3592003644,0.1424958332, 0.0) | (nbody_system.length)/(nbody_system.time)  #Defined by Xiaoming et al.as (v1,v2)

    b2 = bodies[2]
    b2.mass = 1.0 | nbody_system.mass  #Defined by Xiaoming et al.
    b2.position = (1,0.001,0)  | nbody_system.length 
    b2.velocity = (0.3592003644, 0.1424958332, 0.0) | (nbody_system.length)/(nbody_system.time)  #Defined by Xiaoming et al. to be (v1,v2)
    
    b3 = bodies[1]
    b3.mass = 0.75 | nbody_system.mass  #Defined by Xiaoming et al.
    b3.position = (0, 0, 0) | nbody_system.length
    b3.velocity = (-0.9578676384,-0.3799888885, 0.0) | (nbody_system.length)/(nbody_system.time)  #Defined by Xiaoming et al. ((2*v1)/mass_3, (2*v2)/m3))
  
    return bodies

def integrate_bodies(particles, end_time):
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
def plot_track(x_b3,y_b3,x_b2,y_b2,x_b1,y_b1, output_filename):

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
  
    plot.plot(x_b1.value_in(nbody_system.length), y_b1.value_in(nbody_system.length), color = 'g')
    plot.plot(x_b3.value_in(nbody_system.length), y_b3.value_in(nbody_system.length), color = 'b')
    plot.plot(x_b2.value_in(nbody_system.length), y_b2.value_in(nbody_system.length), color = 'r')
    plot.set_xlim(-2, 2)
    plot.set_ylim(-2, 2)

    save_file = 'Three_body_problem.png'
    pyplot.savefig(save_file)
    print '\nSaved figure in file', save_file,'\n'
    pyplot.show()


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-o", 
                      dest="output_filename", default ="ThreeBodyProblem",
                      help="output filename [%default]")
    return result
    
if __name__ in ('__main__','__plot__'):
    o, arguments  = new_option_parser().parse_args()

    bodies = bodies()
    x_b3,y_b3, x_b2,y_b2, x_b1,y_b1 = integrate_bodies(bodies, 1000 | nbody_system.time)
    plot_track(x_b3, y_b3, x_b2, y_b2, x_b1, y_b1, o.output_filename)
    
