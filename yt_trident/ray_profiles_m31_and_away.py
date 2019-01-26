# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import glob
from tools import HUBBLE_2Mpc_LG, K_BOLTZMANN
from absorber_analysis import get_attribute_by_distance, get_ratio_by_distance


ray_m31_filename = './rays_2Mpc_LG_to_mw_2000/ray_2000.000_0.698_4.115.h5'
ray_away_filename = './rays_2Mpc_LG_to_mw_2000/ray_2000.000_1.036_1.314.h5'

abs_m31_filename = './absorbers_2Mpc_LG_to_mw_2000/abs_2000.000_0.698_4.115.txt'
abs_away_filename = './absorbers_2Mpc_LG_to_mw_2000/abs_2000.000_1.036_1.314.txt'

df_m31 = pd.read_csv(abs_m31_filename, skiprows=1)
df_away = pd.read_csv(abs_away_filename, skiprows=1)



if __name__=='__main__':

    plt.figure()
    r_m31, ratio_m31 = get_ratio_by_distance(df_m31, 'Si II 1260', 'Si III 1206')
    r_away, ratio_away = get_ratio_by_distance(df_away, 'Si II 1260', 'Si III 1206')
    plt.semilogy(r_m31/HUBBLE_2Mpc_LG, ratio_m31, label = 'from m31')
    plt.semilogy(r_away/HUBBLE_2Mpc_LG, ratio_away, label = 'from away')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('Column density ratio', fontsize = 15)
    plt.title('Si II / Si III', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r_m31, T_m31 = get_attribute_by_distance(df_m31, 'C II 1036', 'T')
    r_away, T_away = get_attribute_by_distance(df_away, 'C II 1036', 'T')
    plt.semilogy(r_m31/HUBBLE_2Mpc_LG, T_m31, label = 'from m31', color = 'red')
    plt.semilogy(r_away/HUBBLE_2Mpc_LG, T_away, label = 'from away', color = 'crimson', ls = '--')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('Temperature [K]', fontsize = 15)
    plt.title('Temperature profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r_m31, p_m31 = get_attribute_by_distance(df_m31, 'C II 1036', 'p')
    r_away, p_away = get_attribute_by_distance(df_away, 'C II 1036', 'p')
    plt.semilogy(r_m31/HUBBLE_2Mpc_LG, p_m31/K_BOLTZMANN, label = 'from m31', color = 'blue')
    plt.semilogy(r_away/HUBBLE_2Mpc_LG, p_away/K_BOLTZMANN, label = 'from away', color = 'cyan')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('pressure/k [K/cm^3]', fontsize = 15)
    plt.title('Pressure profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r_m31, rho_m31 = get_attribute_by_distance(df_m31, 'C II 1036', 'rho')
    r_away,rho_away = get_attribute_by_distance(df_away, 'C II 1036', 'rho')
    plt.semilogy(r_m31/HUBBLE_2Mpc_LG, rho_m31, label = 'from m31', color = 'magenta')
    plt.semilogy(r_away/HUBBLE_2Mpc_LG, rho_away, label = 'from away', color = 'purple')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('density [g/cm^3]', fontsize = 15)
    plt.title('Mass density profile', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r_m31, vlos_m31 = get_attribute_by_distance(df_m31, 'C II 1036', 'v_los')
    r_away,vlos_away = get_attribute_by_distance(df_away, 'C II 1036', 'v_los')
    plt.semilogy(r_m31/HUBBLE_2Mpc_LG, vlos_m31, label = 'from m31', color = 'magenta')
    plt.semilogy(r_away/HUBBLE_2Mpc_LG, vlos_away, label = 'from away', color = 'purple')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('v LOS [km/s]', fontsize = 15)
    plt.title('vlos profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r_m31, NS3_m31 = get_attribute_by_distance(df_m31, 'Si III 1206', 'N')
    r_away, NS3_away = get_attribute_by_distance(df_away, 'Si III 1206', 'N')
    r_m31, NS2_m31 = get_attribute_by_distance(df_m31, 'Si II 1260', 'N')
    r_away, NS2_away = get_attribute_by_distance(df_away, 'Si II 1260', 'N')
    plt.subplot(121)
    plt.semilogy(r_m31/HUBBLE_2Mpc_LG, NS3_m31+1, label = 'Si III 1206\n from m31', color = 'purple')
    plt.semilogy(r_away/HUBBLE_2Mpc_LG, NS3_away+1, label = 'Si III 1206\n from away', color = 'violet', ls = '--')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$N$ [cm$^{-2}$]', fontsize = 15)
    plt.title('Column density profiles', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.semilogy(r_m31/HUBBLE_2Mpc_LG, NS2_m31+1, label = 'Si II 1260\n from m31', color = 'red')
    plt.semilogy(r_away/HUBBLE_2Mpc_LG, NS2_away+1, label = 'Si II 1260\n from away', color = 'coral', ls = '--')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$N$ [cm$^{-2}$]', fontsize = 15)
    plt.title('Column density profiles', fontsize = 20)
    plt.grid()
    plt.legend()
