import numpy as np
from amuse.lab import *
from amuse.units import nbody_system
from amuse.datamodel import Particle, Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.units import units
from .HybridGravity import HybridGravity
import argparse

def get_args():
    """
    Obtains and returns a dictionary of the command line arguments for this program

    :return: Command line arguments and their values in a dictionary
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_bodies", required=False, default=1e4, type=int, help="Number of bodies to simulate")
    ap.add_argument("-dc", "--direct_code", required=False, default="ph4", type=str, help="Direct Code Integrator (ph4, Huayno, Hermite, or SmallN) or None")
    ap.add_argument("-tc", '--tree_code', required=False, default='bhtree', type=str, help="Tree Code Integrator (BHTree, etc.) or None")
    ap.add_argument("-mc", '--mass_cut', required=False, default=6., type=float, help="Mass Cutoff for splitting bodies, in units MSun (default = 6.)")
    ap.add_argument("-f", "--flip_split", required=False, default=False, type=bool, help="Flip the splitting procedure, if True, all particles above mass_cut are sent to the tree code"
                                                                                         " if False (default), all particles above mass_cut are sent to the direct code")
    ap.add_argument("-t", "--timestep", required=False, default=0.1, type=float, help="Timestep to save out information in Myr, default is 0.1 Myr")
    ap.add_argument("-end", "--end_time", required=False, default=10., type=float, help="End time of simulation in Myr, defaults to 10. Myr")
    ap.add_argument("-r", "--virial_radius", required=False, default=3., type=float, help="Virial Radius in kpc, defaults to 3.")

    args = vars(ap.parse_args())
    return args

if __name__ in ('__main__', '__plot__'):
    args = get_args()
    n = args['num_bodies']
    particles = new_plummer_model(n)
    mZAMS = new_powerlaw_mass_distribution(n, 0.1|units.MSun, 100|units.MSun, alpha=-2.0)
    particles.mass = mZAMS

    gravity = HybridGravity()
    gravity.add_particles(particles)
    gravity.evolve_model(10 | units.Myr)
