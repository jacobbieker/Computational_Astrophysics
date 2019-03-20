import numpy as np
from amuse.units import nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_powerlaw_mass_distribution
from amuse.units import units
from HybridGravity import HybridGravity
from amuse.units.optparse import OptionParser
import matplotlib.pyplot as plt
# TODO: Rename any imports like plt and np to the full name

def new_option_parser():
    result = OptionParser()
    result.add_option("-N", dest="N", type="int",
                      default = 10000,
                      help="Number of particles [%default]")
    result.add_option("--theta", dest="theta", type="float",
                      default = 0.3,
                      help="Opening Angle [%default]")
    result.add_option("--M_min", unit = units.MSun, dest="M_min", type="float",
                      default = 0.1 | units.MSun,
                      help="Min. star mass [%default]")
    result.add_option("--M_max", unit = units.MSun, dest="M_max", type="float",
                      default = 100 | units.MSun,
                      help="Max. star mass [%default]")
    result.add_option("-r", unit = units.parsec, dest="r", type="float",
                      default = 3 | units.parsec,
                      help="Size of star cluster [%default]")
    result.add_option("--t_end", unit = units.Myr, dest="t_end", type="float",
                      default = 10 | units.Myr,
                      help="End time of simulation [%default]")
    result.add_option("--tg_time_step_frac", dest="tg_time_step_frac", type="float",
                      default = 0.1,
                      help="Fraction of gravity timestep for SE timestep[%default]")
    result.add_option("--tse_time_step_frac", dest="tse_time_step_frac", type="float",
                      default = 0.5,
                      help="Fraction of t_dyn for gravity timestep [%default]")
    result.add_option("--bridge_time_step_frac", dest="bridge_time_step_frac", type="float",
                      default = 1/20.,
                      help="Fraction of bridge timestep [%default]")
    result.add_option("--imf", dest="imf", type="string",
                      default = 'salpeter',
                      help="Choose IMF: salpeter, miller_scalo, otherexp [%default]")
    result.add_option("--scheme", dest="code", type="int",
                      default = 0,
                      help = "Gravity code: (0) hybrid, (1) all N-body, (2) all tree-code [%default]")
    result.add_option("--m_cut", unit = units.MSun, dest="m_cut", type="float",
                      default = 10 | units.MSun,
                      help="Mass splitting parameter [%default]")

    return result


if __name__ in ('__main__'):
    o, arguments = new_option_parser().parse_args()
    # Just to check if everything is given properly
    # print result
    print(o.__dict__)
    main(**o.__dict__)
