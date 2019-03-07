import numpy as np
from amuse.lab import *
from amuse.units import nbody_system
from amuse.datamodel import Particle, Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.units import units
from HybridGravity import HybridGravity
from amuse.community.huayno.interface import Huayno
import amuse.datamodel.particle_attributes
import argparse

import matplotlib.pyplot as plt

def get_args():
    """
    Obtains and returns a dictionary of the command line arguments for this program

    :return: Command line arguments and their values in a dictionary
    """

    def str2bool(v):
        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_bodies", required=False, default=10000, type=int, help="Number of bodies to simulate")
    ap.add_argument("-dc", "--direct_code", required=False, default="ph4", type=str, help="Direct Code Integrator (ph4, Huayno, Hermite, or SmallN) or None")
    ap.add_argument("-tc", '--tree_code', required=False, default='bhtree', type=str, help="Tree Code Integrator (BHTree, Bonsai, Octgrav etc.) or None")
    ap.add_argument("-mc", '--mass_cut', required=False, default=6., type=float, help="Mass Cutoff for splitting bodies, in units MSun (default = 6.)")
    ap.add_argument("-f", "--flip_split", required=False, default=False, type=str2bool, help="Flip the splitting procedure, if True, all particles above mass_cut are sent to the tree code"
                                                                                         " if False (default), all particles above mass_cut are sent to the direct code")
    ap.add_argument("-t", "--timestep", required=False, default=0.1, type=float, help="Timestep to save out information in Myr, default is 0.1 Myr")
    ap.add_argument("-end", "--end_time", required=False, default=10., type=float, help="End time of simulation in Myr, defaults to 10. Myr")
    ap.add_argument("-r", "--virial_radius", required=False, default=3., type=float, help="Virial Radius in kpc, defaults to 3.")
    ap.add_argument("-c", "--use_converter", required=False, default=True, type=str2bool, help="Whether to use the converter from nbody units to physical units, with units 1 MSun, 1 kpc, defaults to True")

    args_dict = vars(ap.parse_args())

    if args_dict['direct_code'].lower() == "none":
        args_dict['direct_code'] = None

    if args_dict['tree_code'].lower() == "none":
        args_dict['tree_code'] = None

    return args_dict


if __name__ in ('__main__', '__plot__'):
    args = get_args()
    print(args)
    cluster_mass = 100000 | units.MSun
    mZAMS = new_powerlaw_mass_distribution(args['num_bodies'], 0.1|units.MSun, 100|units.MSun, alpha=-2.0)
    if args['use_converter']:
        converter = nbody_system.nbody_to_si(cluster_mass, args['virial_radius'] |units.parsec)
    else:
        converter = None

    particles = new_plummer_model(args['num_bodies'], convert_nbody=converter)
    particles.mass = mZAMS
    particles.scale_to_standard(convert_nbody=converter)
    # set_standard scale to rescale it

    gravity = HybridGravity(direct_code=args['direct_code'],
                            tree_code=args['tree_code'],
                            mass_cut=args['mass_cut'] | units.MSun,
                            timestep=args['timestep'],
                            flip_split=args['flip_split'],
                            convert_nbody=converter)
    gravity.add_particles(particles)

    timestep_history, mass_history, energy_history, half_mass_history, core_radii_history = gravity.evolve_model(args['end_time'] | units.Myr)

    print("Timestep length: {}".format(len(timestep_history)))

    plt.plot(timestep_history, mass_history, label="Mass")
    plt.plot(timestep_history, energy_history, label="Energy")
    plt.plot(timestep_history, half_mass_history, label="Half-Mass")
    plt.plot(timestep_history, core_radii_history, label="Core Radii")
    plt.title("Histories")
    plt.legend(loc='best')
    plt.show()
    plt.cla()

    print(max(energy_history))
    print(min(energy_history))


