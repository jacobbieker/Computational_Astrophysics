import numpy as np
from amuse.lab import *
from amuse.units import nbody_system
from amuse.datamodel import Particle, Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.units import units
from .HybridGravity import HybridGravity



if __name__ in ('__main__', '__plot__'):
    # TODO Add parsing arguments here
    n = 1e4
    particles = new_plummer_model(n)
    mZAMS = new_powerlaw_mass_distribution(n, 0.1|units.MSun, 100|units.MSun, alpha=-2.0)
    particles.mass = mZAMS

    gravity = HybridGravity()
    gravity.add_particles(particles)
    gravity.evolve_model(10 | units.Myr)
