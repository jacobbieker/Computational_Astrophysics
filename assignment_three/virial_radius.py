from amuse.units import nbody_system
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.units import units

# Create the required mass distribution
mZAMS = new_powerlaw_mass_distribution(10000, 0.1|units.MSun, 100|units.MSun, alpha=-2.0)

# The convertor, so that the nbody units are scaled to the entire mass of all the stars
# as well as the virial radius of 3 parsecs
converter = nbody_system.nbody_to_si(mZAMS.sum(), 3 |units.parsec)

# Creates the plummer model, scaled to the converter, so it should have a virial radius, or core radius, of 3 parsecs
particles = new_plummer_model(10000, convert_nbody=converter)
_, core_radius, _ = particles.densitycentre_coreradius_coredens(unit_converter=converter)
print(core_radius.value_in(units.parsec)) # Roughly 1.2 parsecs, so not what should be 3 parsecs

particles.mass = mZAMS # Add the masses from the distribution
_, core_radius, _ = particles.densitycentre_coreradius_coredens(unit_converter=converter)
print(core_radius.value_in(units.parsec)) # Roughly 1.2 parsecs

particles.scale_to_standard(convert_nbody=converter) # Should scale it to 3 parsecs
_, core_radius, _ = particles.densitycentre_coreradius_coredens(unit_converter=converter)
print(core_radius.value_in(units.parsec)) # Still roughly 1.2 parsecs

converter=nbody_system.nbody_to_si(Mcluster,Rcluster)
stars=new_king_model(N,W0,convert_nbody=converter)
masses = new_powerlaw_mass_distribution(N, 0.1|units.MSun, 100|units.MSun, -2.35)
stars.mass = masses
stars.scale_to_standard(converter)

