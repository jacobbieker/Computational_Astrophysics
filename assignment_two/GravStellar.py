from amuse.lab import *
import numpy as np
import matplotlib.pyplot as plt
from amuse.ext.solarsystem import get_position

from amuse.units import units, constants, nbody_system
from amuse.units.quantities import zero
from amuse.datamodel import Particle, Particles
from amuse.support.console import set_printing_strategy

from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary

from amuse.community.huayno.interface import Huayno
from amuse.community.smalln.interface import SmallN
from amuse.community.hermite0.interface import Hermite
from amuse.community.seba.interface import SeBa
from amuse.community.sse.interface import SSE


class GravStellar(object):
    """
    Class to do the different timestep updates for the stellar and gravitational updates

    Should be given bodies used, and the run the
    """

    def __init__(self, timestep, end_time, integration_scheme, stellar_mass_loss_frac, stars, stellar_start_time,
                 gravity_model=SeBa, stellar_model=Hermite, verbose=False):

        self.timestep = timestep
        self.num_bodies = len(stars)
        self.verbose = verbose
        self.gravity_model = gravity_model
        self.stellar_model = stellar_model
        self.stars = stars
        self.end_time = end_time
        self.integration_scheme = integration_scheme
        self.stellar_mass_loss_time_step_fraction = stellar_mass_loss_frac
        self.stellar_start_time = stellar_start_time

        self.gravity = None
        self.stellar = None

        self.time = 0.0 | end_time.unit


    def advance_stellar(self, ts, dt):
        E0 = self.gravity.kinetic_energy + self.gravity.potential_energy
        ts += dt
        self.stellar.evolve_model(ts)
        channel_from_stellar.copy_attributes(["mass"])
        channel_from_framework_to_gd.copy_attributes(["mass"])
        return ts, gravity.kinetic_energy + gravity.potential_energy - E0

    def advance_gravity(self, tg, dt):
        tg += dt
        self.gravity.evolve_model(tg)
        channel_from_gd_to_framework.copy()
        return tg
    def get_orbital_period(self):
        return 2*np.pi*(a**3/(constants.G*Mtot)).sqrt()

    def get_semi_major_axis(self):
        return (constants.G*Mtot*P**2/(4*np.pi**2))**(1./3)

    def get_mass_loss(self):
        return NotImplementedError

    def get_orbital_elements_of_triple(self):
        inner_binary = self.stars[0]+self.stars[1]
        outer_binary = Particles(1)
        outer_binary[0].mass = inner_binary.mass.sum()
        outer_binary[0].position = inner_binary.center_of_mass()
        outer_binary[0].velocity = inner_binary.center_of_mass_velocity()
        outer_binary.add_particle(self.stars[2])
        M1, M2, ain, ein, ta_in, inc_in, lan_in, aop_in \
            = orbital_elements_from_binary(inner_binary, G=constants.G)
        M12, M3, aout, eout, ta_out, outc_out, lan_out, aop_out \
            = orbital_elements_from_binary(outer_binary, G=constants.G)
        return ain, ein, aout, eout

    def evolve(self):

        while self.time < self.end_time:

            # TODO: Add determining timestep

            if self.integration_scheme == 1:

                ts, dE_se = self.advance_stellar(ts, dt)
                time = self.advance_gravity(time, dt)

            elif self.integration_scheme == 2:

                time = self.advance_gravity(time, dt)
                ts, dE_se = self.advance_stellar(ts, dt)

            else:

                dE_se = zero
                #ts, dE_se = advance_stellar(ts, dt/2)
                time = self.advance_gravity(time, dt)
                #ts, dE = advance_stellar(ts, dt/2)
                #dE_se += dE

            if self.time >= self.t_diag:

                t_diag = time + dt_diag

                Ekin = gravity.kinetic_energy
                Epot = gravity.potential_energy
                Etot = Ekin + Epot
                dE = Etot_prev - Etot
                Mtot = triple.mass.sum()
                print("T=", time, end=' ')
                print("M=", Mtot, "(dM[SE]=", Mtot / Mtriple, ")", end=' ')
                print("E= ", Etot, "Q= ", Ekin / Epot, end=' ')
                print("dE=", (Etot_init - Etot) / Etot, "ddE=", (Etot_prev - Etot) / Etot, end=' ')
                print("(dE[SE]=", dE_se / Etot, ")")
                Etot_init -= dE
                Etot_prev = Etot
                ain, ein, aout, eout = get_orbital_elements_of_triple(triple)
                print("Triple elements t=", (4 | units.Myr) + time,
                      "inner:", triple[0].mass, triple[1].mass, ain, ein,
                      "outer:", triple[2].mass, aout, eout)

                t.append(time.value_in(units.Myr))
                smai.append(ain/ain_0)
                ecci.append(ein/ein_0)
                smao.append(aout/aout_0)
                ecco.append(eout/eout_0)

                if eout > 1.0 or aout <= zero:
                    print("Binary ionized or merged")
                    break

        self.gravity.stop()
        self.stellar.stop()

        return t, smai, ecci, smao, ecco

    def set_gravity(self, stars):
        converter = nbody_system.nbody_to_si(stars.mass.sum(), aout_0)
        self.gravity = self.gravity_model(converter)
        self.gravity.particles.add_particles(stars)

        channel_from_framework_to_gd = stars.new_channel_to(self.gravity.particles)
        channel_from_gd_to_framework = self.gravity.particles.new_channel_to(stars)

        total_initial_energy = self.gravity.kinetic_energy + self.gravity.potential_energy
        total_previous_energy = total_initial_energy

        self.gravity.particles.move_to_center()



    def age_stars(self):
        mult_stars = Particles(self.num_bodies)
        for i in range(self.num_bodies):
            mult_stars[i].mass = self.stars[i].mass

        # Start Stellar Evolution

        stellar = self.stellar_model()
        stellar.particles.add_particles(mult_stars)
        channel_from_stellar = stellar.particles.new_channel_to(mult_stars)
        stellar.evolve_model(self.stellar_start_time)
        channel_from_stellar.copy_attributes(['mass'])
        for i in range(self.num_bodies):
            self.stars[i].mass = mult_stars[i].mass



