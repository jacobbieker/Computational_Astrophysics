from amuse.couple import bridge
from amuse.units import units
from amuse.datamodel import Particles
from amuse.community.huayno.interface import Huayno
from amuse.community.hermite0.interface import Hermite
from amuse.community.smalln.interface import SmallN
from amuse.community.ph4.interface import ph4
from amuse.community.bhtree.interface import BHTree
from amuse.community.octgrav.interface import Octgrav
from amuse.community.seba.interface import SeBa
from amuse.community.bonsai.interface import Bonsai
from amuse.io import write_set_to_file
import numpy as np
from amuse.ic.salpeter import new_salpeter_mass_distribution
# TODO Rename all 'as np' kind of ones to the actual name
import time


class Galaxy(object):

    def __init__(self, gravity_code=Huayno, stellar_code=SeBa, hydro_code=None, radiative_code=None,
                 mass=1e9 | units.MSun, dark_matter_mass=1e10 | units.MSun, radius=10 | units.kpc,
                 stellar_distribution=new_salpeter_mass_distribution):
        """

        This class includes everything needed to instantiate and evolve a simulated galaxy, with optional gas,
        radiative transport, stellar evolution, "dark matter", etc.

        """
        raise NotImplementedError
