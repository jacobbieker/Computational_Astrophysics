import numpy as np
from amuse.lab import *
from amuse.units import nbody_system
from amuse.datamodel import Particle, Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.units import units
from .HybridGravity import HybridGravity


if __name__ in ('__main__', '__plot__'):
    print("Hi")
    particles = new_plummer_model(1000)
    mZAMS = new_powerlaw_mass_distribution(1000, 0.1|units.MSun, 100|units.MSun, alpha=-2.0)
    particles = Particles(mass=mZAMS)

    gravity = HybridGravity()
    gravity.add_particles(particles)

