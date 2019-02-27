from GravStellar import GravitationalStellar

import matplotlib.pyplot as plt
import numpy as np

from amuse.units import units, constants, nbody_system
from amuse.units.quantities import zero
from amuse.datamodel import Particle, Particles

from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary

from amuse.community.huayno.interface import Huayno
from amuse.community.smalln.interface import SmallN
from amuse.community.hermite0.interface import Hermite
from amuse.community.seba.interface import SeBa
from amuse.community.sse.interface import SSE

from amuse.ext.solarsystem import get_position

from prepare_figure import single_frame
from distinct_colours import get_distinct


def plot_results(computer_time, eccentricity_out, eccentricity_in, semimajor_axis_in, semimajor_axis_out,
                 stellar_mass_fraction, grav_stellar):
    """
    Plot outputs from a single simulation
    :param computer_time: list of times from simulation
    :param eccentricity_out: List of eccentricities of outer binary
    :param eccentricity_in: List of eccentricites of the inner binary
    :param semimajor_axis_in: List of semimajor axis of inner binary
    :param semimajor_axis_out: List of semimajor axis of outer binary
    :param stellar_mass_fraction: Stellar mass fraction to determine timestep
    :param grav_stellar: GravitationalStellar object
    :return:
    """
    x_label = "$a/a_{0}$"
    y_label = "$e/e_{0}$"
    fig = single_frame(x_label, y_label, logx=False, logy=False,
                       xsize=10, ysize=8)
    color = get_distinct(4)
    plt.cla()

    time, ai, ei, ao, eo = computer_time, semimajor_axis_in, eccentricity_in, semimajor_axis_out, eccentricity_out
    plt.plot(ai, ei, c=color[0], label='inner')
    plt.plot(ao, eo, c=color[1], label='outer')

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.tight_layout()
    save_file = 'semi_and_eccen' \
                + '_dtse={:.7f}'.format(stellar_mass_fraction) \
                + '.png'
    plt.savefig(save_file)
    print('\nSaved figure in file', save_file, '\n')

    grav_models = ['hermite', 'huayno', 'smalln']
    if grav_stellar.gravity_model == Hermite:
        grav_model = grav_models[0]
    elif grav_stellar.gravity_model == Huayno:
        grav_model = grav_models[1]
    else:
        grav_model = grav_models[2]

    # Now plot the orbital params as function of timestep

    # First Eccentricity vs time

    plt.cla()
    plt.plot(computer_time, eccentricity_out, label='Outer')
    plt.plot(computer_time, eccentricity_in, label='Inner')
    plt.legend(loc='best')
    plt.ylabel("Eccentricity/(Initial Eccentricity)")
    plt.xlabel("Time (years)")
    plt.title("Wall Clock Time: {} s\n Sim Time: {} s".format(np.round(grav_stellar.elapsed_total_time, 3),
                                                              np.round(grav_stellar.elapsed_sim_time, 3)))
    plt.tight_layout()
    plt.savefig(
        "eccentricity_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method={}".format(stellar_mass_fraction, grav_stellar.inclination, grav_model,
                                                                      grav_stellar.integration_scheme) + ".png")

    # then Semimajor axis vs time

    plt.cla()
    plt.plot(computer_time, semimajor_axis_in, label="Outer")
    plt.plot(computer_time, semimajor_axis_out, label='Inner')
    plt.legend(loc='best')
    plt.ylabel("Semimajor Axis/(Initial Semimajor Axis)")
    plt.xlabel("Time (years)")
    plt.title("Wall Clock Time: {} s\n Sim Time: {} s".format(np.round(grav_stellar.elapsed_total_time, 3),
                                                              np.round(grav_stellar.elapsed_sim_time, 3)))
    plt.tight_layout()
    plt.savefig(
        "semimajor_axis_vs_time_dts={:.8f}_inc={:.3f}_grav={}_method={}".format(stellar_mass_fraction, grav_stellar.inclination, grav_model,
                                                                        grav_stellar.integration_scheme) + ".png")
    plt.cla()


def plot_all_results(computer_times, eccentricity_outs, eccentricity_ins, semimajor_axis_ins, semimajor_axis_outs,
                     stellar_mass_fraction, grav_stellars, inclinations):
    """
    Plots results combining multiple outputs into one. Specifically, this makes a plot of all 4 different integration
    methods in one. Saves out all files.
    :param computer_times: List of time lists
    :param eccentricity_outs: List of lists of the outer binary eccentricities
    :param eccentricity_ins:  List of lists of the inner binary eccentricities
    :param semimajor_axis_ins: List of lists of the inner binary semimajor axis
    :param semimajor_axis_outs: List of lists of the outer binary semimajor axis
    :param stellar_mass_fraction: The stellar mass fraction used to determine the timestep
    :param grav_stellars: List of GravitationalStellar objects
    :param inclinations: List of lists of the inclination between the binaries
    :return:
    """
    schemes = ["interlaced", "stellar_first", "gravity_first", "diagnostic"]
    grav_models = ['hermite', 'huayno', 'smalln']
    if grav_stellars[0].gravity_model == Hermite:
        grav_model = grav_models[0]
    elif grav_stellars[0].gravity_model == Huayno:
        grav_model = grav_models[1]
    else:
        grav_model = grav_models[2]
    avg_wall_time = 0.0
    avg_sim_time = 0.0
    plt.cla()
    for j in range(len(computer_times)):
        plt.plot(computer_times[j], semimajor_axis_ins[j], label="inner, {}".format(schemes[j]))
        plt.plot(computer_times[j], semimajor_axis_outs[j], label="outer, {}".format(schemes[j]))
        avg_wall_time += grav_stellars[j].elapsed_total_time
        avg_sim_time += grav_stellars[j].elapsed_sim_time

    avg_wall_time /= len(computer_times)
    avg_sim_time /= len(computer_times)

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Semimajor Axis/(Initial Semimajor Axis)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("ALL_semimajor_axis_vs_time_inc={:.3f}_dts={:.8f}_grav={}_method=all".format(grav_stellars[0].inclination, stellar_mass_fraction,
                                                                                        grav_model) + ".png")
    plt.cla()

    # Now plot the orbital params as function of timestep
    for j in range(len(computer_times)):
        plt.plot(computer_times[j], eccentricity_ins[j], label="inner, {}".format(schemes[j]))
        plt.plot(computer_times[j], eccentricity_outs[j], label="outer, {}".format(schemes[j]))

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Eccentricity/(Initial Eccentricity)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("ALL_eccentricity_vs_time_inc={:.3f}_dts={:.8f}_grav={}_method=all".format(grav_stellars[0].inclination, stellar_mass_fraction,
                                                                                grav_model) + ".png")
    plt.cla()


    for j in range(len(computer_times)):
        plt.plot(computer_times[j], inclinations[j], label="{}".format(schemes[j]))

    plt.legend(loc='best', ncol=1, shadow=False, fontsize=20)
    plt.title(
        "Avg Wall Clock Time: {} s\n Avg Sim Time: {} s".format(np.round(avg_wall_time, 3), np.round(avg_sim_time, 3)))
    plt.ylabel("Inclination/(Initial Inclination)")
    plt.xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig("ALL_inclination_vs_time_inc={:.3f}_dts={:.8f}_grav={}_method=all".format(grav_stellars[0].inclination, stellar_mass_fraction,
                                                                                         grav_model) + ".png")
    plt.cla()


def get_orbital_period(orbital_separation, total_mass):
    return 2 * np.pi * (orbital_separation ** 3 / (constants.G * total_mass)).sqrt()


def get_semi_major_axis(orbital_period, total_mass):
    return (constants.G * total_mass * orbital_period ** 2 / (4 * np.pi ** 2)) ** (1. / 3)


def generate_initial_muscea(integration_method='interlaced', stellar_mass_loss_fraction=0.000001,
                            inclination=60, gravity_model=Huayno):
    # Initial Conditions
    eccentricity_init = 0.2
    eccentricity_out_init = 0.6
    semimajor_axis_out_init = 100 | units.AU
    mean_anomaly = 180
    argument_of_perigee = 180
    longitude_of_the_ascending_node = 0

    M1 = 60 | units.MSun
    M2 = 30 | units.MSun
    M3 = 20 | units.MSun
    period_init = 19 | units.day
    semimajor_axis_init = 0.63 | units.AU

    period_or_semimajor = 1  # 1: Get semimajor_axis from period, Otherwise: get period from semimjaor_axis

    stellar_start_time = 4.0 | units.Myr
    end_time = 0.001 | units.Myr
    triple = Particles(3)
    triple[0].mass = M1
    triple[1].mass = M2
    triple[2].mass = M3

    grav_stellar = GravitationalStellar(stellar_mass_loss_timestep_fraction=stellar_mass_loss_fraction,
                                        gravity_model=gravity_model,
                                        integration_scheme=integration_method, inclination=inclination)
    grav_stellar.add_particles(triple)
    triple = grav_stellar.age_stars(stellar_start_time)
    # Inner binary
    tmp_stars = Particles(2)
    tmp_stars[0].mass = triple[0].mass
    tmp_stars[1].mass = triple[1].mass

    if period_or_semimajor == 1:
        semimajor_axis_init = get_semi_major_axis(period_init, triple[0].mass + triple[1].mass)
    else:
        period_init = get_orbital_period(semimajor_axis_init, triple[0].mass + triple[1].mass)

    delta_time = 0.1 * period_init

    # Inner binary

    r, v = get_position(triple[0].mass, triple[1].mass, eccentricity_init, semimajor_axis_init, mean_anomaly,
                        inclination,
                        argument_of_perigee, longitude_of_the_ascending_node, delta_time)
    tmp_stars[1].position = r
    tmp_stars[1].velocity = v
    tmp_stars.move_to_center()

    # Outer binary

    r, v = get_position(triple[0].mass + triple[1].mass, triple[2].mass, eccentricity_out_init, semimajor_axis_out_init,
                        0,
                        0, 0, 0, delta_time)
    tertiary = Particle()
    tertiary.mass = triple[2].mass
    tertiary.position = r
    tertiary.velocity = v
    tmp_stars.add_particle(tertiary)
    tmp_stars.move_to_center()

    triple.position = tmp_stars.position
    triple.velocity = tmp_stars.velocity

    grav_stellar.add_particles(triple)

    grav_stellar.set_initial_parameters(semimajor_axis_init, eccentricity_init,
                                        semimajor_axis_out_init, eccentricity_out_init, period_init)

    grav_stellar.set_gravity(semimajor_axis_out_init)

    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, inclination_history = grav_stellar.evolve_model(end_time)
    print("Number of Timesteps: ", len(timestep_history))
    plot_results(timestep_history, eccentricity_out_history, eccentricity_in_history, semimajor_axis_in_history,
                 semimajor_axis_out_history, stellar_mass_loss_fraction, grav_stellar)

    return timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
           semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history


def run_simulation(gravity_model=Huayno, stellar_mass_loss_fraction=0.001):
    """
    Runs the simulations for roughly 0, 30, 60, and 90 degree inclination
    :param gravity_model: Gravity model to use, of Huayno, Hermite, and SmallN
    :param stellar_mass_loss_fraction: Stellar Mass Loss fraction to use to determine timestep
    """
    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                      stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                      inclination=0.01, gravity_model=gravity_model)

    stimestep_history, smass_history, ssemimajor_axis_in_history, seccentricity_in_history, \
    ssemimajor_axis_out_history, seccentricity_out_history, sgrav_stellar, sinclination_history = generate_initial_muscea("stellar_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=0.01, gravity_model=gravity_model)

    gtimestep_history, gmass_history, gsemimajor_axis_in_history, geccentricity_in_history, \
    gsemimajor_axis_out_history, geccentricity_out_history, ggrav_stellar, ginclination_history = generate_initial_muscea("gravity_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=0.01, gravity_model=gravity_model)

    dtimestep_history, dmass_history, dsemimajor_axis_in_history, deccentricity_in_history, \
    dsemimajor_axis_out_history, deccentricity_out_history, dgrav_stellar, dinclination_history = generate_initial_muscea("diagnostic",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=0.01, gravity_model=gravity_model)

    plot_all_results([timestep_history, stimestep_history, gtimestep_history, dtimestep_history],
                     [eccentricity_out_history, seccentricity_out_history, geccentricity_out_history,
                      deccentricity_out_history],
                     [eccentricity_in_history, seccentricity_in_history, geccentricity_in_history,
                      deccentricity_in_history],
                     [semimajor_axis_in_history, ssemimajor_axis_in_history, gsemimajor_axis_in_history,
                      dsemimajor_axis_in_history],
                     [semimajor_axis_out_history, ssemimajor_axis_out_history, gsemimajor_axis_out_history,
                      dsemimajor_axis_out_history],
                     stellar_mass_loss_fraction,
                     [grav_stellar, sgrav_stellar, ggrav_stellar, dgrav_stellar],
                     [inclination_history, sinclination_history, ginclination_history, dinclination_history])

    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                      stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                      inclination=1., gravity_model=gravity_model)

    stimestep_history, smass_history, ssemimajor_axis_in_history, seccentricity_in_history, \
    ssemimajor_axis_out_history, seccentricity_out_history, sgrav_stellar, sinclination_history = generate_initial_muscea("stellar_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=1., gravity_model=gravity_model)

    gtimestep_history, gmass_history, gsemimajor_axis_in_history, geccentricity_in_history, \
    gsemimajor_axis_out_history, geccentricity_out_history, ggrav_stellar, ginclination_history = generate_initial_muscea("gravity_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=1., gravity_model=gravity_model)

    dtimestep_history, dmass_history, dsemimajor_axis_in_history, deccentricity_in_history, \
    dsemimajor_axis_out_history, deccentricity_out_history, dgrav_stellar, dinclination_history = generate_initial_muscea("diagnostic",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=1., gravity_model=gravity_model)

    plot_all_results([timestep_history, stimestep_history, gtimestep_history, dtimestep_history],
                     [eccentricity_out_history, seccentricity_out_history, geccentricity_out_history,
                      deccentricity_out_history],
                     [eccentricity_in_history, seccentricity_in_history, geccentricity_in_history,
                      deccentricity_in_history],
                     [semimajor_axis_in_history, ssemimajor_axis_in_history, gsemimajor_axis_in_history,
                      dsemimajor_axis_in_history],
                     [semimajor_axis_out_history, ssemimajor_axis_out_history, gsemimajor_axis_out_history,
                      dsemimajor_axis_out_history],
                     stellar_mass_loss_fraction,
                     [grav_stellar, sgrav_stellar, ggrav_stellar, dgrav_stellar],
                     [inclination_history, sinclination_history, ginclination_history, dinclination_history])
    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                      stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                      inclination=0.5, gravity_model=gravity_model)

    stimestep_history, smass_history, ssemimajor_axis_in_history, seccentricity_in_history, \
    ssemimajor_axis_out_history, seccentricity_out_history, sgrav_stellar, sinclination_history = generate_initial_muscea("stellar_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=0.5, gravity_model=gravity_model)

    gtimestep_history, gmass_history, gsemimajor_axis_in_history, geccentricity_in_history, \
    gsemimajor_axis_out_history, geccentricity_out_history, ggrav_stellar, ginclination_history = generate_initial_muscea("gravity_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=0.5, gravity_model=gravity_model)

    dtimestep_history, dmass_history, dsemimajor_axis_in_history, deccentricity_in_history, \
    dsemimajor_axis_out_history, deccentricity_out_history, dgrav_stellar, dinclination_history = generate_initial_muscea("diagnostic",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=0.5, gravity_model=gravity_model)

    plot_all_results([timestep_history, stimestep_history, gtimestep_history, dtimestep_history],
                     [eccentricity_out_history, seccentricity_out_history, geccentricity_out_history,
                      deccentricity_out_history],
                     [eccentricity_in_history, seccentricity_in_history, geccentricity_in_history,
                      deccentricity_in_history],
                     [semimajor_axis_in_history, ssemimajor_axis_in_history, gsemimajor_axis_in_history,
                      dsemimajor_axis_in_history],
                     [semimajor_axis_out_history, ssemimajor_axis_out_history, gsemimajor_axis_out_history,
                      dsemimajor_axis_out_history],
                     stellar_mass_loss_fraction,
                     [grav_stellar, sgrav_stellar, ggrav_stellar, dgrav_stellar],
                     [inclination_history, sinclination_history, ginclination_history, dinclination_history])

    timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
    semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                      stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                      inclination=1.5, gravity_model=gravity_model)

    stimestep_history, smass_history, ssemimajor_axis_in_history, seccentricity_in_history, \
    ssemimajor_axis_out_history, seccentricity_out_history, sgrav_stellar, sinclination_history = generate_initial_muscea("stellar_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=1.5, gravity_model=gravity_model)

    gtimestep_history, gmass_history, gsemimajor_axis_in_history, geccentricity_in_history, \
    gsemimajor_axis_out_history, geccentricity_out_history, ggrav_stellar, ginclination_history = generate_initial_muscea("gravity_first",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=1.5, gravity_model=gravity_model)

    dtimestep_history, dmass_history, dsemimajor_axis_in_history, deccentricity_in_history, \
    dsemimajor_axis_out_history, deccentricity_out_history, dgrav_stellar, dinclination_history = generate_initial_muscea("diagnostic",
                                                                                                                          stellar_mass_loss_fraction=stellar_mass_loss_fraction,
                                                                                                                          inclination=1.5, gravity_model=gravity_model)

    plot_all_results([timestep_history, stimestep_history, gtimestep_history, dtimestep_history],
                     [eccentricity_out_history, seccentricity_out_history, geccentricity_out_history,
                      deccentricity_out_history],
                     [eccentricity_in_history, seccentricity_in_history, geccentricity_in_history,
                      deccentricity_in_history],
                     [semimajor_axis_in_history, ssemimajor_axis_in_history, gsemimajor_axis_in_history,
                      dsemimajor_axis_in_history],
                     [semimajor_axis_out_history, ssemimajor_axis_out_history, gsemimajor_axis_out_history,
                      dsemimajor_axis_out_history],
                     stellar_mass_loss_fraction,
                     [grav_stellar, sgrav_stellar, ggrav_stellar, dgrav_stellar],
                     [inclination_history, sinclination_history, ginclination_history, dinclination_history])


def get_timestep_walltime_differences(gravity_model):
    """
    Produces the dependence of the timestep on the wall time for a given gravity model
    :param gravity_model: Gravity model to use, Hermite, Huayno, or SmallN
    """
    timesteps = np.linspace(0.01, 0.0000001, 25)
    wall_times = []
    sim_times = []
    amuse_times = []
    grav_models = ['hermite', 'huayno', 'smalln']
    if gravity_model == Hermite:
        grav_model = grav_models[0]
    elif gravity_model == Huayno:
        grav_model = grav_models[1]
    else:
        grav_model = grav_models[2]
    for timestep in timesteps:
        timestep_history, mass_history, semimajor_axis_in_history, eccentricity_in_history, \
        semimajor_axis_out_history, eccentricity_out_history, grav_stellar, inclination_history = generate_initial_muscea("interlaced",
                                                                                                                          stellar_mass_loss_fraction=timestep,
                                                                                                                          inclination=1, gravity_model=gravity_model)
        wall_times.append(grav_stellar.elapsed_total_time)
        sim_times.append(grav_stellar.elapsed_sim_time)
        amuse_times.append(grav_stellar.elapsed_total_time - grav_stellar.elapsed_sim_time)

    plt.cla()
    plt.plot(timesteps, wall_times, label='Wall')
    plt.plot(timesteps, sim_times, label='Sim')
    plt.plot(timesteps, amuse_times, label='AMUSE')
    plt.title("Timestep vs Wall Time")
    plt.xlabel("Timestep (Stellar Mass Loss Fraction)")
    plt.ylabel("Wall Time (s)")
    plt.legend(loc='best')
    plt.savefig("timestep_vs_walltime_grav={}".format(grav_model) + ".png")
    plt.cla()


if __name__ in ('__main__', '__plot__'):
    gravity_model = Huayno # Gravity model to use

    do_timestep = False # Whether to calculate the timestep dependence or not
    if do_timestep:
        get_timestep_walltime_differences(gravity_model)

    stellar_mass_loss_fractions = [0.001, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001] # Timesteps to run in the simulation
    # Runs all the timesteps for all the integration schemes
    for fraction in stellar_mass_loss_fractions:
        run_simulation(gravity_model, fraction)
