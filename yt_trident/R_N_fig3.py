# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
from absorber_analysis import covering_fraction_by_rays_one_to_map, \
                              covering_fraction_by_rays
from multiprocessing import Pool
from tools import usable_cores
import glob

pool = Pool(usable_cores())

abs_directory = './absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter/'

N_threshs = np.logspace(12, 15, 10)

abs_lines = ['Si III 1206', 'C II 1335', 'Si II 1193', 'C IV 1548']

observations = {line: np.loadtxt('../../RN_fig.3/{}.txt'.format(line),
                skiprows = 1, delimiter=',') for line in abs_lines}



def covering_fractions_parallel(absorbers_directory, N_thresh_list, line_list,
                                pool):

    sightlines_over_thresh = {(N_thresh, line):0 for N_thresh in N_thresh_list \
                              for line in line_list}

    tasks = [(handle, N_thresh_list, line_list, sightlines_over_thresh) for \
             handle in glob.glob(absorbers_directory + 'abs*')]

    number_of_sightlines = len(tasks)

    pool.map(covering_fraction_by_rays_one_to_map, tasks)

    covfs = {(N_thresh, line): \
            sightlines_over_thresh[(N_thresh, line)]/number_of_sightlines for \
            N_thresh, line in sightlines_over_thresh.keys()}

    return covfs

#covfs = covering_fractions_parallel(abs_directory, N_threshs, abs_lines, pool)



covfs = {line:covering_fraction_by_rays(abs_directory, line, N_threshs) for
         line in abs_lines}


fig, axs = plt.subplots(2,2, sharex=True, squeeze=True)
axs = axs.flatten()

for i, line in enumerate(abs_lines):

    axs[i].errorbar(10**observations[line][:,0], observations[line][:,1],
                    observations[line][:,2])
    axs[i].semilogx(N_threshs, np.array(list(covfs[line].values())), label = line)
                 #lw = '.')
    axs[i].set_xscale('log')
    axs[i].legend()
    axs[i].grid()
    #plt.xlabel('$\\Log N_{thresh}$', fontsize = 15)
    #plt.ylabel('')
