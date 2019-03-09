# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
from multiprocessing import Pool
from tools import usable_cores, handle_in_subsample
from absorber_analysis import get_attribute_by_distance
import glob
import gc

abs_directory = './absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter_sun/'


def v_los_all_cells(abs_directory, subsampling=True):

    if subsampling:
        files = [handle for handle in glob.glob(abs_directory + 'abs_*') if \
                 handle_in_subsample(handle, amplitude_polar=1)]

    else:
        files = [handle for handle in glob.glob(abs_directory + 'abs_*')]

    v_los_list = []

    for handle in files:

        df = pd.read_csv(handle, skiprows=1)
        linekey = df['Line'].iloc[0]

        v_los = df[df['Line']==linekey]['v_los']
        v_los_list.append(v_los)

    return np.concatenate(v_los_list)


def v_los_all_cells_by_distance(abs_directory, r_max, delta_r,
                                subsampling=True):

    if subsampling:
        files = [handle for handle in glob.glob(abs_directory + 'abs_*') if \
                 handle_in_subsample(handle, amplitude_polar=1)]

    else:
        files = [handle for handle in glob.glob(abs_directory + 'abs_*')]

    v_los_list = []
    r_bins = np.arange(0, r_max+delta_r, delta_r)

    for handle in files:

        df = pd.read_csv(handle, skiprows=1)
        r, v_los = get_attribute_by_distance(df, df['Line'].iloc[0], 'v_los')

        v_los_binned = np.histogram(r, r_bins, weights=v_los)[0]

        v_los_list.append(v_los_binned)

    v_by_dist = {r:np.vstack(v_los_list)[:,i] for i,r in enumerate(r_bins[:-1])}

    return v_by_dist


vlos_selected = v_los_all_cells(abs_directory)
vlos_all = v_los_all_cells(abs_directory, False)

v_dist_selected = v_los_all_cells_by_distance(abs_directory, 250, 10)
v_dist_all = v_los_all_cells_by_distance(abs_directory, 250, 10, False)


plt.figure()
plt.hist(vlos_selected, bins=100, edgecolor='black', color = 'magenta', log=True)
plt.xlabel('$V_{los} [km\\slash s]$', fontsize=15)
plt.title('Selected sightlines', fontsize=15)
plt.grid()

plt.figure()
plt.hist(vlos_all, bins=100, edgecolor='black', color = 'crimson', log=True)
plt.xlabel('$V_{los} [km\\slash s]$', fontsize=15)
plt.title('All sightlines', fontsize=15)
plt.grid()


plt.figure()
plt.hist(v_dist_selected, bins=100, edgecolor='black', color = 'crimson', log=True)
plt.xlabel('$V_{los} [km\\slash s]$', fontsize=15)
plt.title('All sightlines', fontsize=15)
plt.grid()
