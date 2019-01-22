# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
from absorber_analysis import covering_fraction_by_rays

abs_directory = './absorbers_2Mpc_LG_to_mw_2000/'

N_threshs = np.logspace(12, 15, 10)

abs_lines = ['Si III 1206', 'C II 1335', 'Si II 1193', 'C IV 1548']

covfs = {line:covering_fraction_by_rays(abs_directory, line, N_threshs) for
         line in abs_lines}


fig, axs = plt.subplots(2,2, sharex=True, squeeze=True)
axs = axs.flatten()

for i, line in enumerate(abs_lines):

    axs[i].semilogx(N_threshs, np.array(list(covfs[line].values())), label = line)
                 #lw = '.')
    axs[i].legend()
    axs[i].grid()
    #plt.xlabel('$\\Log N_{thresh}$', fontsize = 15)
    #plt.ylabel('')
