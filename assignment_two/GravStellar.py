from amuse.lab import *
import numpy as np


class GravStellar(object):

    def __init__(self, timestep, num_bodies, masses, velocities, ):

        self.timestep = timestep
        self.num_bodies = num_bodies
        self.masses = masses
        self.velocities = velocities
