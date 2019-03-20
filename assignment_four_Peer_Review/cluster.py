#from amuse.lab import *
from __future__ import division
import numpy as np
import time as whatever
from amuse.lab import new_plummer_model
from amuse.lab import Particles, units, nbody_system, constants
from amuse.ic.salpeter import new_salpeter_mass_distribution
from amuse.ic.brokenimf import new_miller_scalo_mass_distribution
from amuse.lab import new_powerlaw_mass_distribution
from amuse.lab import ph4, BHTree, SeBa
from amuse.couple import bridge
from matplotlib import pyplot as plt
from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.io import write_set_to_file
import csv
import os

"""
    EDITS
Changing the np and dplt to numpy and matplotlib.pyplot
    
"""

# Create a directory to store the results, if it exists, we pass
if os.path.exists('./results'):
	pass
else:
    os.makedirs('./results')


initial_run_time = whatever.time()

# Creates a star cluster given the initial conditions N, M_min, M_Max, imf:
# Plummer model = a self-consistent equilibrium solution to Poisson's eq-n,
# which creates stars of equal mass. We want our N stars to obey an IMF such
# that M_min = 0.1 MSun and M_max = 100 MSun (see 1-24 in AMUSE book).
# Several IMF choices implemented, interchangeable via imf parameter in option parser.

# Three different options for an IMF, can be changedin the option parser
def cluster(N, M_min, M_max, r, imf):
	if imf == 'salpeter':
		mZAMS = new_salpeter_mass_distribution(N, M_min.in_(units.MSun), M_max.in_(units.MSun))
	elif imf == 'miller_scalo':
		mZAMS = new_miller_scalo_mass_distribution(N, M_max.in_(units.MSun))
	elif imf == 'otherexp':
		print '\nExponential IMF with alpha = -2. Change in cluster() if needed.\n'
		mZAMS = new_powerlaw_mass_distribution(N, M_min.in_(units.MSun), M_max.in_(units.MSun), alpha = -2.0)
	else:
		print 'Wrong syntax on <imf>'

	Mtot = mZAMS.sum()
	# Converting acoording to our system, its total mass and radius
	convert = nbody_system.nbody_to_si(Mtot, r)
	# Using the plummer model to create the stars
	cluster = new_plummer_model(N, convert_nbody = convert)
	# Assigned new masses according to the chosen imf,
	# however this changes M_tot and hence the energy
	# => Need to revirialise: done with scale_to_standard() function.
	cluster.mass = mZAMS
	virial_init = cluster.kinetic_energy() / cluster.potential_energy()
	cluster.scale_to_standard(convert_nbody = convert)
	virial_end = cluster.kinetic_energy() / cluster.potential_energy()
	# Check that the system is virialized before and after the conversion
	print 'Virial equilibrium', '\nInitially', virial_init, '\nAfter scaling', virial_end, '\n'
	return cluster


# Function to calculate the half-mass radius of the cluster
def halfmass_radius(starcluster):
	# from amuse.ext.LagrangianRadii import LagrangianRadii
	MassFraction = [0.5, 1.0]
	R_halfmass = LagrangianRadii(starcluster, massf=MassFraction,verbose=True)[0]
	# print R_halfmass.in_(units.parsec)
	return R_halfmass

# Function to calculate the dynamical timescale of a class of particles
def dynamical_timescale(R_halfmass, starcluster):
	return (R_halfmass**3/(constants.G*starcluster.mass.sum())).sqrt()


# Function to evolve a star cluster gravitationally, taking the stars split
# in two: 'light' for stars of mass below than a specified mass-cut and 'heavy'
# for stars heavier than the mass-cut. The ligher masses are evolved using a
# tree-code, whereas the heavier - using a direct N-body code.
# Additionally, the code includes stellar evolution for each star in the cluster.
def hybrid_gravity(N, starcluster, theta, m_cut, r,
				   t_end, tg_time_step_frac, tse_time_step_frac, bridge_time_step_frac,
				   imf, method):

	# Objects of class Particles() to hold the stars ligher and heavier than the specified m_cut.
	light = Particles()
	heavy = Particles()

	# Put particles with mass below m_cut in light(), and heavier in heavy().
	light = starcluster[starcluster.mass<m_cut]
	heavy = starcluster-light

	M_tot = light.mass.sum() + heavy.mass.sum()
	convert = nbody_system.nbody_to_si(M_tot, r)


	# Introducing a flag system so the code can run in three different schemes
	# that can be changed in the option parser by changing the "scheme" option.
	# (0)hybrid, (1)nbody, (2)tree

	light_flag = True
	heavy_flag = True
	if len(light) == 0: light_flag = False
	if len(heavy) == 0: heavy_flag = False

	#  Stellar evolution model introduced and used in all three schemes
	stellar_evolution = SeBa()

	#  In hybrid and tree schemes, the tree model for the 'light' stars is introduced
	if light_flag:
		# Barnes-Hut tree model for the "light" stars
		light_gravity = BHTree(convert)
		# print light_gravity.parameters
		light_gravity.parameters.opening_angle = theta
		light_gravity.parameters.timestep = 0.5*light_gravity.parameters.timestep
		light_gravity.particles.add_particles(light)
		# Adding the 'light' particles to SE
		stellar_evolution.particles.add_particle(light)
	else:	pass

	#  In hybrid and nbody schemes, the nbody model for the 'heavy' stars is introduced
	if heavy_flag:
		# Nbody model for "heavy" stars
		heavy_gravity = ph4(convert)
		heavy_gravity.particles.add_particles(heavy)
		# Adding the 'heavy' particles to SE
		stellar_evolution.particles.add_particle(heavy)
	else:	pass

	# In nbody or tree only codes, the corresponding model is the one used
	if not heavy_flag:	combined_gravity = light_gravity
	elif not light_flag:	combined_gravity = heavy_gravity
	# In hybrid model, the tree and nbody model must be bridged
	else:
		# Bridge between the two gravity models
		combined_gravity = bridge.Bridge()
		combined_gravity.add_system(light_gravity, (heavy_gravity,))
		combined_gravity.add_system(heavy_gravity, (light_gravity,))

	# Channel gravity info to framework so we can plot the star's positions in each point in time we want
	channel_from_combined_gravity_to_framework = combined_gravity.particles.new_channel_to(starcluster)
	# Channel from stellar_evolution to combined_gravity so the mass of each stars gets updated in each cycle
	channel_from_stellar_to_framework = stellar_evolution.particles.new_channel_to(starcluster)
	channel_from_framework_to_combined_gravity = starcluster.new_channel_to(combined_gravity.particles)

	# Calculating the initial energy of the system
	E_comb_gr_init = combined_gravity.kinetic_energy + combined_gravity.potential_energy
	E_comb_gr = E_comb_gr_init
	# print 'Initial virial ratio:', combined_gravity.kinetic_energy / combined_gravity.potential_energy



	time_list = [] | units.Myr
	dE_list = []
	ddE_list = []
	Rhm = [] | units.parsec
	time = 0 | units.Myr
	time_se = 0 | units.Myr
	cr = [] | units.AU
	x_cluster = [] | units.AU
	y_cluster = [] | units.AU
	z_cluster = [] | units.AU
	# The iteration starts here
	while time < t_end:
		light = starcluster[starcluster.mass<m_cut]
		heavy = starcluster-light

		MassFraction = [0.5, 1.0]
		# Halfmass radius calculation
		R_halfmass = LagrangianRadii(combined_gravity.particles, massf=MassFraction,verbose=True)[0]      # Both the Halfmass radius and the dynamical time scale of the cluster
		# Calculating the dynamical time scale of the cluster												# change over time due to the position and mass of the stars changing
		# t_dyn_cluster = (R_halfmass**3/(constants.G*starcluster.mass.sum())).sqrt()
		t_dyn_cluster = dynamical_timescale(R_halfmass, starcluster)

#
#
#
#
#
# -----------------Careful at these lines-----------------
# The fractions can be changed in the option parser


		# Combined_gravity timestep
		tg_time_step = t_dyn_cluster * tg_time_step_frac
		# Stellar evolution timestep
		tse_time_step = tg_time_step * tse_time_step_frac
		# Bridge timestep
		time_step_bridge = tg_time_step*bridge_time_step_frac
		combined_gravity.timestep = time_step_bridge


#-------------------------------------------------------
#
#
#
#
#
#





		# print '\nVirial equilibrium', combined_gravity.kinetic_energy / combined_gravity.potential_energy
		# Advancing the gravity and stellar evolution by a time step
		time += tg_time_step
		time_se += tse_time_step
		# Half step for stellar evolution
		stellar_evolution.evolve_model(time_se)
		channel_from_stellar_to_framework.copy()
		channel_from_framework_to_combined_gravity.copy()

		combined_gravity.evolve_model(time)
		channel_from_combined_gravity_to_framework.copy()
		# Another half step for stellar evolution
		time_se += tse_time_step
		stellar_evolution.evolve_model(time_se)

		# Calculate new energy
		E_comb_gr_prev = E_comb_gr
		E_comb_gr = combined_gravity.kinetic_energy + combined_gravity.potential_energy
		# Calculate the total dE
		dE = abs((E_comb_gr - E_comb_gr_init) / E_comb_gr_init)
		# Calculate the dE in each cycle
		ddE = abs((E_comb_gr - E_comb_gr_prev) / E_comb_gr)
		# print 'Time=', time.in_(units.Myr), 'dE=', dE, 'ddE=', ddE


		# Calculate the core radius
		pos,core_radius,coredens = combined_gravity.particles.densitycentre_coreradius_coredens(convert)

		# Append everything into lists to plot
		cr.append(core_radius)
		Rhm.append(R_halfmass)
		time_list.append(time)
		dE_list.append(dE)
		ddE_list.append(ddE)
		x_cluster.append(starcluster.x)
		y_cluster.append(starcluster.y)
		z_cluster.append(starcluster.z)


		# Just a percentage to indicate the time needed to finish the iteration
		percentage = int(round(100*time/t_end))
		box = int(round(10*time/t_end))
		print percentage, '%'
		print '| ---------- |\n|', '-' * box, '|\n| ---------- |\n'


	# Plot the dE, ddE, core radius and halfmass radius in one plot with two Y-axis, the left for the energies and the right for the radii
	fig, ax1 = matplotlib.pyplot.subplots()

	ax2 = ax1.twinx()

	p1, =ax1.plot(time_list.value_in(units.Myr), dE_list, ls='-', label='dE(t)', color ='r')
	p2, =ax1.plot(time_list.value_in(units.Myr), ddE_list, ls=':', label='ddE(t)', color ='r')
	p3, =ax2.plot(time_list.value_in(units.Myr), cr.value_in(units.parsec), ls='-', label='Core Radius (t)', color ='g')
	p4, =ax2.plot(time_list.value_in(units.Myr), Rhm.value_in(units.parsec), ls='-.', label='Halfmass Radius (t)', color ='g')
	ax1.semilogy()
	ax1.set_xlabel('$time(Myr)$')
	ax1.set_ylabel('$dE$')
	ax2.set_ylabel('Radius(parsec)')

	lines = [p1, p2, p3, p4]
	matplotlib.pyplot.legend(lines, [l.get_label() for l in lines], loc='best')
	matplotlib.pyplot.title('Combined Gravity, IMF=%s, Gravity type=%s\nN=%i, Mcut= %.1f, Opening angle=%.1f'%(imf,method, N, m_cut.value_in(units.MSun), theta))
	fig.tight_layout()
	matplotlib.pyplot.savefig('./results/cluster_%s_%s_N=%i_mcut=%.1f_theta=%.1f.pdf'%(imf, method, N, m_cut.value_in(units.MSun), theta))

	matplotlib.pyplot.clf()
	matplotlib.pyplot.close()

	# Stop the evolution model that is running depending on the scheme that is used
	if light_flag:	light_gravity.stop()
	if heavy_flag:	heavy_gravity.stop()

	dE_final = dE_list[-1]

	return N, dE_final, theta, x_cluster, y_cluster, z_cluster


# The main funtcion wrapping up everything
def main(N, theta, M_min, M_max, r,
		 t_end, tg_time_step_frac, tse_time_step_frac, bridge_time_step_frac,
		 imf, code, m_cut):
    # Splitting parameter:
    # code = 0 uses specified m_cut for the particles to go to the hybrid code;
    # code = 1 sets m_cut = M_min for all particles to go to the N-body code;
    # code = 2 sets m_cut = M_max for all particles to go to the tree code.
	# scheme = code | can be changed in the option parser
	if code == 0:
		method = 'hybrid'

	if code == 1:
		m_cut = M_min
		method = 'nbody'

	if code == 2:
		m_cut = M_max
		method = 'tree'


	# The important part
	# ------------------------------------------------------------------------------------------------------------------------------------------------
	# ------------------------------------------------------------------------------------------------------------------------------------------------
	starcluster = cluster(N, M_min, M_max, r, imf)
	N, final_energy_error, theta, x_cluster, y_cluster, z_cluster = hybrid_gravity(N, starcluster,theta, m_cut, r,
																				   t_end, tg_time_step_frac, tse_time_step_frac, bridge_time_step_frac,
																				   imf, method)
	# ------------------------------------------------------------------------------------------------------------------------------------------------
	# ------------------------------------------------------------------------------------------------------------------------------------------------



	# Plotting a histogram to have a picture of the IMF
	fig_hist, ax_hist = matplotlib.pyplot.subplots()
	ax_hist.hist(numpy.log(starcluster.mass.value_in(units.MSun)))
	ax_hist.set_xlabel('$log{mass}$')
	matplotlib.pyplot.title('Mass Distribution')
	matplotlib.pyplot.savefig('./results/cluster_%s_N=%i.pdf'%(imf, N))
	matplotlib.pyplot.clf()
	matplotlib.pyplot.close()

	# Plotting a scatter plot of the cluster before evolving
	from mpl_toolkits.mplot3d import Axes3D
	fig_scatter = matplotlib.pyplot.figure()
	ax_scatter = fig_scatter.add_subplot(111, projection='3d')

	xcluster = x_cluster.value_in(units.AU)
	ycluster = y_cluster.value_in(units.AU)
	zcluster = z_cluster.value_in(units.AU)
	# The '1' corresponds to the second time step of the evolution, it can be change to plot a scatter plot at any time
	ax_scatter.scatter(xcluster[1,:], ycluster[1,:], zcluster[1,:])
	matplotlib.pyplot.savefig('./results/cluster_scatter_%s_%s_N=%i_mcut=%.1f_theta=%.1f.pdf'%(imf, method, N, m_cut.value_in(units.MSun), theta))
	matplotlib..clf()
	matplotlib.pyplot.close()

	# ------------------------------------------------------------------------------------------------------------------------------------------------
	# ------------------------------------------------------------------------------------------------------------------------------------------------

	# An unsuccessful attempt to create an animation of the cluster
	#
	# x_anim = []
	# y_anim = []
	# z_anim = []
	# for i in range(len(xcluster)):
	# 	x_anim.append(xcluster[i,:])
	# 	y_anim.append(ycluster[i,:])
	# 	z_anim.append(zcluster[i,:])
	#
	# x_anim = np.array(x_anim)
	# y_anim = np.array(y_anim)
	# z_anim = np.array(z_anim)
	#
	# fig_anim = plt.figure()
	# ax_anim = fig_anim.add_subplot(111, projection='3d')
	#
	#
	# def update(num):
	# 	# graph._offsets3d = (x_anim, y_anim, z_anim)
	# 	ax_anim.clear()
	# 	ax_anim.scatter(x_anim, y_anim, z_anim)
	#
	#
	#
	# # graph = ax_anim.plot(x_anim[0], y_anim[0], z_anim[0])
	#
	#
	# anim = animation.FuncAnimation(fig_anim, update, interval=100, frames=6, repeat=True, blit=False)
	# plt.show()
	#
	# anim.save('./results/name.gif', writer='imagemagick')

	# ------------------------------------------------------------------------------------------------------------------------------------------------
	# ------------------------------------------------------------------------------------------------------------------------------------------------



	end_run_time = whatever.time()
	# Calculate the wall clock time
	total_run_time = end_run_time - initial_run_time
	print '\n\n\nTotal time elapsed', total_run_time, 'seconds'


	things_to_print = {'Run Time': total_run_time | units.s,'dE': final_energy_error,'Mass Cut': m_cut}
	# Print the Wall clock time, the total energy error and the mass cut to a file for later use
	with open('./results/cluster_%s_%s_N=%i_mcut=%.1f_theta=%.1f.csv'%(imf, method, N, m_cut.value_in(units.MSun), theta), 'wb') as csvfile:
		writer = csv.DictWriter(csvfile, things_to_print.keys())
		writer.writeheader()
		writer.writerow(things_to_print)


# Introducing the option parser so every parameter can be altered through the terminal
def new_option_parser():
	from amuse.units.optparse import OptionParser
	result = OptionParser()
	result.add_option("-N", dest="N", type="int",
					  default = 10000,
					  help="no. of particles [%default]")
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
# Running the whole code with the parameters declared in the option parser
if __name__ in ("__main__"):
    o, arguments = new_option_parser().parse_args()
	# Just to check if everything is given properly
    # print result
    print o.__dict__
    main(**o.__dict__)