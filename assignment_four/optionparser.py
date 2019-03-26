from amuse.units.optparse import OptionParser

def new_option_parser():
    result = OptionParser()
    result.add_option("-N", dest="N", type="int", default = 10000, help="no. of particles [%default]")
    result.add_option("-M_min", unit = units.MSun, dest="M_min", type="float", default = 0.1 | units.MSun, help="Min. star mass [%default]")
    result.add_option("-M_max", unit = units.MSun, dest="M_max", type="float", default = 100 | units.MSun, help="Max. star mass [%default]")
    result.add_option("-r", unit = units.parsec, dest="r", type="float", default = 3 | units.parsec, help="Size of star cluster [%default]")
    result.add_option("-t_end", unit = units.Myr, dest="t_end", type="float", default = 10 | units.Myr, help="End time of simulation [%default]")
    # Trial for IMF:
    result.add_option("-IMF", dest="imf", type="string", default = 'salpeter', help="Choose IMF: salpeter, miller_scalo, otherexp [%default]")
    # or, if this doesn't work, involve numbers, not strings:     REMEMBER to change cluster()
    # result.add_option("-IMF", dest="imf", type="int", default = 1, help="Choose IMF: salpeter (1), miller_scalo (2), otherexp (3) [%default]")
    result.add_option("-scheme", dest="code", type="int", default = 0, help = "Gravity code: (0) hybrid, (1) all N-body, (2) all tree-code [%default]")
    # make conditional statement to ask for m_cut if code == 0 (hybrid)?
    result.add_option("-m_cut", unit = units.MSun, dest="m_cut", type="float", default = 10 | units.MSun, help="Mass splitting parameter [%default]")
    
    return result
    
if __name__ in ("__main__"):
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)
    
    
    

def main(N, M_min, M_max, r, t_end, imf, code, m_cut):
    # Splitting parameter:
    # code = 0 uses specified m_cut for the particles to go to the hybrid code;
    # code = 1 sets m_cut = M_min for all particles to go to the N-body code;
    # code = 2 sets m_cut = M_max for all particles to go to the tree code.
    if code == 1:	m_cut = M_min
    elif code == 2:	m_cut = M_max    
    
    starcluster = cluster(N, M_min, M_max, r)	# create star cluster using cluster()
    N, theta, final_energy_error = hybrid_gravity(starcluster, m_cut, r, t_end)
    end_run_time = whatever.time()
    total_run_time = end_run_time - initial_run_time
    print '\n\n\nTotal time elapsed', total_run_time, 'seconds'


    things_to_print = {'Run Time': total_run_time | units.s,'dE': final_energy_error,'Mass Cut': m_cut}

    with open('./results/cluster_N=%i_mcut=%.1f_theta=%.1f.csv'%(N, m_cut.value_in(units.MSun), theta), 'wb') as csvfile:
	writer = csv.DictWriter(csvfile, things_to_print.keys())
	writer.writeheader()
	writer.writerow(things_to_print)