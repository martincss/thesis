# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import glob
from tools import HUBBLE_2Mpc_LG, K_BOLTZMANN

from absorber_analysis import radial_profile_various, \
                              weighted_radial_profile_various, \
                              covering_fraction_various, \
                              column_density_ratio_profile

R_MAX = 300 * HUBBLE_2Mpc_LG
R_BIN_LENGTH = 3 # kpc/h


def radial_profiles_2k(line, attributes_list, delta_r, weighted):
    """
    Radial profile wrapper for 2Mpc_LG to mw at 2000kpc/h
    """

    if weighted:

        return weighted_radial_profile_various(
                        absorbers_directory='./absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter_sun/',
                        line=line, attributes_list=attributes_list, r_max=R_MAX,
                        delta_r=delta_r)
    else:

        return radial_profile_various(
                        absorbers_directory='./absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter_sun/',
                        line=line, attributes_list=attributes_list, r_max=R_MAX,
                        delta_r=delta_r)

def column_density_profile_2k(line, delta_r, weighted=False):

    return radial_profiles_2k(line, ['N'], delta_r, weighted)

#
# def density_profile_2k(delta_r, weighted=False):
#
#     return radial_profile_2k('C II 1036', 'rho', delta_r, weighted)
#
#
# def pressure_profile_2k(delta_r, weighted=False):
#
#     return radial_profile_2k('C II 1036', 'p', delta_r, weighted)
#
#
# def temperature_profile_2k(delta_r, weighted=False):
#
#     return radial_profile_2k('C II 1036', 'T', delta_r, weighted)


def covering_fractions_2k(line_list, N_thresh, delta_r):
    """
    Covering_fraction wrapper for 2Mpc_LG to mw at 2000kpc/h
    """

    return covering_fraction_various('./absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter_sun/',
                            line_list=line_list, N_thresh=N_thresh, r_max=R_MAX,
                            delta_r=delta_r)


def cold_ratio_profile_2k(line1, line2, delta_r):

    return column_density_ratio_profile('./absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter_sun/',
                                        line1=line1, line2=line2, r_max=R_MAX,
                                        delta_r=delta_r)



if __name__=='__main__':
    # print('first profiles')
    # r, profiles = radial_profiles_2k('Si II 1260', ['T', 'p', 'rho', 'N'], R_BIN_LENGTH, False)
    #
    # print('now density weighted profiles')
    # _, profiles_w = radial_profiles_2k('Si II 1260', ['T', 'p', 'rho', 'N'], R_BIN_LENGTH,
    #                                    True)

    print('next covering fractions > 1e9')
    r, covfs_9 = covering_fractions_2k(['Si III 1206', 'Si II 1193', \
                                        'C II 1335', 'C IV 1548'],
                                        1e9, R_BIN_LENGTH)

    print('finally covering fractions > 1e12')
    _, covfs_12 = covering_fractions_2k(['Si III 1206', 'Si II 1193', \
                                        'C II 1335', 'C IV 1548'],
                                        1e12, R_BIN_LENGTH)
    r /= HUBBLE_2Mpc_LG

    #
    # plt.figure()
    # plt.semilogy(r[:-1], profiles['T'], label = 'bin size = {}'.format(R_BIN_LENGTH), color = 'red')
    # plt.semilogy(r[:-1], profiles_w['T'], label = 'denisty weighted', color = 'crimson', ls = '--')
    # plt.xlabel('Distance [kpc]', fontsize = 15)
    # plt.ylabel('Temperature [K]', fontsize = 15)
    # plt.title('Temperature profile MW', fontsize = 20)
    # plt.grid()
    # plt.legend()
    #
    # plt.figure()
    # plt.semilogy(r[:-1], profiles['p']/K_BOLTZMANN, label = 'bin size = {}'.format(R_BIN_LENGTH), color = 'blue')
    # plt.semilogy(r[:-1], profiles_w['p']/K_BOLTZMANN, label = 'density weighted', color = 'cyan')
    # plt.xlabel('Distance [kpc]', fontsize = 15)
    # plt.ylabel('pressure/k [K/cm^3]', fontsize = 15)
    # plt.title('Pressure profile MW', fontsize = 20)
    # plt.grid()
    # plt.legend()
    #
    # plt.figure()
    # plt.semilogy(r[:-1], profiles['rho'], label = 'bin size = {}'.format(R_BIN_LENGTH), color = 'magenta')
    # plt.semilogy(r[:-1], profiles_w['rho'], label = 'density weighted', color = 'purple')
    # plt.xlabel('Distance [kpc]', fontsize = 15)
    # plt.ylabel('density [g/cm^3]', fontsize = 15)
    # plt.title('Mass density profile MW', fontsize = 20)
    # plt.grid()
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(r[:-1], covfs_9['C II 1335'], label = '$N > 10^9 cm^{-2}$', color = 'olive')
    # plt.plot(r[:-1], covfs_12['C II 1335'], label = '$N > 10^{12} cm^{-2}$', color = 'green')
    # plt.xlabel('Distance [kpc]', fontsize = 15)
    # plt.ylabel('$f$', fontsize = 15)
    # plt.title('Covering fraction for C II 1335 MW', fontsize = 20)
    # plt.grid()
    # plt.legend()
    #
    #
    # plt.figure()
    # plt.plot(r[:-1], covfs_9['Si III 1206'], label = '$N > 10^9 cm^{-2}$', color = 'olive')
    # plt.plot(r[:-1], covfs_12['Si III 1206'], label = '$N > 10^{12} cm^{-2}$', color = 'green')
    # plt.xlabel('Distance [kpc]', fontsize = 15)
    # plt.ylabel('$f$', fontsize = 15)
    # plt.title('Covering fraction for Si III 1206 MW', fontsize = 20)
    # plt.grid()
    # plt.legend()
    #
    #
    # plt.figure()
    # print('also some column densities')
    # _, NS3 = column_density_profile_2k('Si III 1206', R_BIN_LENGTH)
    # _, NS3w = column_density_profile_2k('Si III 1206', R_BIN_LENGTH, weighted=True)
    # plt.semilogy(r[:-1], NS3['N'], label = 'Si III 1206', color = 'purple')
    # plt.semilogy(r[:-1], NS3w['N'], label = 'density weighted\n Si III 1206', color = 'violet', ls = '--')
    # plt.semilogy(r[:-1], profiles['N'], label = 'Si II 1260', color = 'red')
    # plt.semilogy(r[:-1], profiles_w['N'], label = 'density weighted\n Si II 1260', color = 'coral', ls = '--')
    # plt.xlabel('Distance [kpc]', fontsize = 15)
    # plt.ylabel('$N$ [cm$^{-2}$]', fontsize = 15)
    # plt.title('Column density profiles', fontsize = 20)
    # plt.grid()
    # plt.legend()


    plt.figure()
    for line, covf in covfs_9.items():
        plt.plot(r[:-1], covf, label = line)
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for $N > 10^{9}$ cm$^{-2}$ MW', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    for line, covf in covfs_12.items():
        plt.plot(r[:-1], covf, label = line)
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for $N > 10^{12}$ cm$^{-2}$ MW', fontsize = 20)
    plt.grid()
    plt.legend()


    #
    # plt.figure()
    # print('and the ratio last of all')
    # _, ratio = cold_ratio_profile_2k('Si II 1260', 'Si III 1206', R_BIN_LENGTH)
    # plt.semilogy(r[:-1], ratio, label = 'bin size = 10kpc')
    # plt.xlabel('Distance [kpc]', fontsize = 15)
    # plt.ylabel('Column density ratio', fontsize = 15)
    # plt.title('Si II / Si III MW', fontsize = 20)
    # plt.grid()
    # plt.legend()
