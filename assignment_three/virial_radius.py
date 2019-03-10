from amuse.units import nbody_system
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.datamodel import Particle, Particles
from amuse.units import units
from amuse.community.ph4.interface import ph4
from amuse.community.hermite0.interface import Hermite
from amuse.ext.bridge import bridge
from amuse.units import quantities
from amuse.community.bhtree.interface import BHTree
import matplotlib.pyplot as pyplot

# Create the required mass distribution
cluster_mass = 1e5 | units.MSun
mZAMS = new_powerlaw_mass_distribution(1000, 0.1|units.MSun, 100|units.MSun, alpha=-2.0)
cluster_mass = mZAMS.sum()
# Create a converter for the whole system
converter = nbody_system.nbody_to_si(cluster_mass, 3. |units.parsec)

particles = new_plummer_model(1000, convert_nbody=converter)
particles.mass = mZAMS
particles.scale_to_standard(convert_nbody=converter) # Scale the converter to the correct virial radius and mass

# Now get the masses in each one for the different converters
direct_particles = Particles()
tree_particles = Particles()
tree_converter = None
direct_converter = None
for particle in particles:
    if particle.mass >= 30. | units.MSun: # Mass cutoff is high to make it work
        direct_particles.add_particle(particle)
    else:
        tree_particles.add_particle(particle)

# Create converters for the tree particles and the direct particles
tree_converter = nbody_system.nbody_to_si(tree_particles.mass.sum(), tree_particles.virial_radius())
direct_converter = nbody_system.nbody_to_si(direct_particles.mass.sum(), direct_particles.virial_radius())

# Copied from the gravity_gravity example code
galaxy_code = BHTree(tree_converter, number_of_workers=1)
setattr(galaxy_code.parameters, "epsilon_squared", (0.001 | units.parsec)**2)
channe_to_galaxy = galaxy_code.particles.new_channel_to(tree_particles)
galaxy_code.particles.add_particles(tree_particles)

# copied from the gravity_gravity example code
cluster_code=ph4(direct_converter, number_of_workers=1)
cluster_code.particles.add_particles(direct_particles)
channel_to_stars=cluster_code.particles.new_channel_to(direct_particles)

system=bridge(verbose=False)
system.add_system(cluster_code, (galaxy_code,))
system.add_system(galaxy_code, (cluster_code,))
system.timestep = 0.1*0.1 | units.Myr

initial_energy = system.potential_energy + system.kinetic_energy
timestep_history = []

energy_history = []

no_smooth_hist = []

some_smooth_hist = []

times = quantities.arange(0|units.Myr, 10. | units.Myr, 0.1 | units.Myr)
for i,t in enumerate(times):
    #print "Time=", t.in_(units.Myr)
    channe_to_galaxy.copy()
    channel_to_stars.copy()

    new_energy = system.potential_energy + system.kinetic_energy

    energy_history.append((initial_energy - new_energy)/initial_energy)
    timestep_history.append(i)

    #inner_stars =  galaxy.select(lambda r: r.length()<Rinit,["position"])
    #print "Minner=", inner_stars.mass.sum().in_(units.MSun)

    system.evolve_model(t)
pyplot.plot(timestep_history, energy_history)
pyplot.title("Energy Loss Over Time Smoothed")
pyplot.xlabel("Timestep")
pyplot.ylabel("(E_intial - E_curr)/E_initial")
pyplot.show()
galaxy_code.stop()
cluster_code.stop()
