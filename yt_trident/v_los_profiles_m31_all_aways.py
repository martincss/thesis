#!/usr/bin/env python
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import glob
import numpy as np
from numpy.linalg import norm
from tools import HUBBLE_2Mpc_LG, K_BOLTZMANN, ray_away_from_m31
from absorber_analysis import get_attribute_by_distance, get_ratio_by_distance
from ray_profiles_m31_and_away import m31_vlos, df_m31


absorbers_directory = './absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter_sun/'


def v_los_away_binned_all(delta_r, r_max = 2000):

    r_bins = np.arange(0, r_max+delta_r, delta_r)
    counts_per_bin = np.zeros(len(r_bins)-1)

    v_los_binned = []

    for handle in glob.glob(absorbers_directory + 'abs*'):

        if ray_away_from_m31(handle, 5):

            df = pd.read_csv(handle, skiprows=1)
            r, vlos = get_attribute_by_distance(df, df['Line'].iloc[0], 'v_los')

            v_los_binned.append(np.histogram(r, r_bins, weights=vlos)[0])

    return r_bins, np.vstack(v_los_binned)


if __name__ == '__main__':

    r_away, v_los_all_away = v_los_away_binned_all(25)
    mean_away = np.mean(v_los_all_away, axis=0)
    sigma_away = np.std(v_los_all_away, axis=0)

    plt.figure()

    for i in range(v_los_all_away.shape[0]):

        plt.plot(r_away[:-1]/HUBBLE_2Mpc_LG, v_los_all_away[i,:], color = 'plum')

    plt.plot(r_away[:-1]/HUBBLE_2Mpc_LG, mean_away, color = 'magenta', label = 'mean away')
    r_m31, vlos_m31 = get_attribute_by_distance(df_m31, df_m31['Line'].iloc[0], 'v_los')
    plt.plot(r_m31/HUBBLE_2Mpc_LG, vlos_m31, label = 'from m31', color = 'purple')
    plt.hlines(m31_vlos, 0, 3000, color = 'crimson', linestyles='dashed', label = 'M31 V LOS')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('v LOS [km/s]', fontsize = 15)
    plt.title('vlos profile', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    plt.plot(r_away[:-1]/HUBBLE_2Mpc_LG, mean_away, color = 'purple', label = 'mean away')
    plt.fill_between(r_away[:-1]/HUBBLE_2Mpc_LG, mean_away - sigma_away,
     mean_away + sigma_away, color = 'plum', label = 'mean away $\\pm \\sigma$')
    plt.plot(r_m31/HUBBLE_2Mpc_LG, vlos_m31, label = 'from m31', color = 'magenta')
    plt.hlines(m31_vlos, 0, 3000, color = 'crimson', linestyles='dashed', label = 'M31 V LOS')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$v_{LOS}$ [km/s]', fontsize = 15)
    plt.title('Velocity in LOS profiles by direction', fontsize = 20)
    plt.grid()
    plt.legend()
