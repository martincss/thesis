# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
from absorber_analysis import covering_fraction_by_rays_one_to_map, \
                              covering_fraction_by_rays
from multiprocessing import Pool
from tools import usable_cores, handle_in_subsample
import glob

pool = Pool(usable_cores())

abs_directory = './absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter/'

N_threshs = np.logspace(12, 15, 10)
vel_threshs = np.arange(0, 120, 15)

abs_lines = ['Si III 1206', 'C II 1335', 'Si II 1193', 'C IV 1548']

observations = {line: np.loadtxt('../../RN_fig.3/{}.txt'.format(line),
                skiprows = 1, delimiter=',') for line in abs_lines}


def covering_fractions_parallel(absorbers_directory, N_thresh_array, line_list,
                                vel_thresh_list, r_min=0, r_max=250,
                                pool=pool):


    tasks = [(handle, N_thresh_array, line_list, vel_thresh_list, r_min, r_max)\
             for handle in glob.glob(absorbers_directory + 'abs*') if \
             handle_in_subsample(handle, amplitude_polar=1)]

    number_of_sightlines = len(tasks)

    results = pool.map(covering_fraction_by_rays_one_to_map, tasks)
    keys = results[0].keys()

    covfs = {key:sum([res[key] for res in results]) for key in keys}


    for key in covfs.keys():

        covfs[key] = covfs[key]/number_of_sightlines

    return covfs


covfs = covering_fractions_parallel(abs_directory, N_threshs, abs_lines,
                                vel_threshs)


fig, axs = plt.subplots(2,2, sharex=True, sharey=True, squeeze=True)
# fig.suptitle('Covering fraction by $N_{thresh}$', fontsize = 20)
axs = axs.flatten()

axs[0].set_ylabel('$f_c$', fontsize= 15)
axs[2].set_xlabel('$N_{thresh}$ [cm$^{-2}$]', fontsize= 15)
axs[2].set_ylabel('$f_c$', fontsize= 15)
axs[3].set_xlabel('$N_{thresh}$ [cm$^{-2}$]', fontsize= 15)

for i, line in enumerate(abs_lines):

    axs[i].errorbar(10**observations[line][:,0], observations[line][:,1],
                    observations[line][:,2], label = 'RN17')

    for vel in vel_threshs:

        axs[i].semilogx(N_threshs, covfs[(line, vel)],
                        label = '$v\\, > {} $ km/s'.format(vel))
        #lw = '.')

    axs[i].set_xscale('log')
    # axs[i].legend()
    axs[i].grid()
    axs[i].set_title(line, fontsize = 15)
    #plt.xlabel('$\\Log N_{thresh}$', fontsize = 15)
    #plt.ylabel('')

axs[3].legend()
