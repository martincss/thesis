# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
from tools import retrieve_all_hvcs
from multiprocessing import Pool
p= Pool(2)

absorbers_directory = 'absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter_sun/'
abs_lines = ['Si III 1206', 'C II 1335', 'Si II 1193', 'C IV 1548']

number_of_bins = 100

hvcs = retrieve_all_hvcs(absorbers_directory, p)

hvcs_each = {line:hvcs[hvcs['Line']==line] for line in abs_lines}
vel_each =  {line: np.array(hvcs_each[line]['vel_spectrum'], dtype=float) \
             for line in abs_lines}

counts_edges_each = {line: np.histogram(vel_each[line],bins=number_of_bins) for \
                     line in abs_lines}
def centers(edges): return edges[:-1] + np.diff(edges)/2

plt.figure()

for line in abs_lines:

    #plt.hist(vel_each[line], bins = number_of_bins, label=line, alpha = 0.3)
    plt.plot(centers(counts_edges_each[line][1]), counts_edges_each[line][0]/500, label = line)


plt.legend()
plt.grid()
plt.xlabel('Velocity [km/s]')
