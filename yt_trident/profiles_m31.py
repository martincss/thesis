# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import glob
from tools import HUBBLE_2Mpc_LG, K_BOLTZMANN

from absorber_analysis import radial_profile, weighted_radial_profile,
                              covering_fraction


def radial_profile_m31(line, attribute, delta_r, weighted):
    """
    Radial profile wrapper for 2Mpc_LG to mw at 2000kpc
    """

    if weighted:

        return weighted_radial_profile(
                        absorbers_directory='./absorbers_2Mpc_LG_to_m31_210/',
                        line=line, attribute=attribute, r_max=210,
                        delta_r=delta_r)
    else:

        return radial_profile(
                        absorbers_directory='./absorbers_2Mpc_LG_to_m31_210/',
                        line=line, attribute=attribute, r_max=210,
                        delta_r=delta_r)


def column_density_profile_m31(line, delta_r, weighted=False):

    return radial_profile_m31(line, 'N', delta_r, weighted)


def density_profile_m31(delta_r, weighted=False):

    return radial_profile_m31('C II 1036', 'rho', delta_r, weighted)


def pressure_profile_m31(delta_r, weighted=False):

    return radial_profile_m31('C II 1036', 'p', delta_r, weighted)


def temperature_profile_m31(delta_r, weighted=False):

    return radial_profile_m31('C II 1036', 'T', delta_r, weighted)


def covering_fraction_m31(line, N_thresh, delta_r):
    """
    Covering_fraction wrapper for 2Mpc_LG to m31 at 210kpc/h
    """

    return covering_fraction('./absorbers_2Mpc_LG_to_m31_210/', line=line,
                             N_thresh=N_thresh, r_max=210, delta_r=delta_r)


def cold_ratio_profile_m31(line1, line2, delta_r):

    return column_density_ratio_profile('./absorbers_2Mpc_LG_to_m31_210/',
                                        line1=line1, line2=line2, r_max=210,
                                        delta_r=delta_r)


if __name__=='__main__':

    plt.figure()
    r, ratio = cold_ratio_profile_m31('Si II 1260', 'Si III 1206', 20)
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, ratio, label = 'bin size = 20kpc/h')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('Column density ratio', fontsize = 15)
    plt.title('Si II / Si III', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, T = temperature_profile_m31(20)
    _, Tw = temperature_profile_m31(20, weighted=True)
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, T, label = 'bin size = 20kpc/h', color = 'red')
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, Tw, label = 'denisty weighted', color = 'crimson', ls = '--')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('Temperature [K]', fontsize = 15)
    plt.title('Temperature profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r, p = pressure_profile_m31(20)
    _, pw = pressure_profile_m31(20, weighted=True)
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, p/K_BOLTZMANN, label = 'bin size = 20kpc/h', color = 'blue')
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, pw/K_BOLTZMANN, label = 'density weighted', color = 'cyan')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('pressure/k [erg/cm^3]', fontsize = 15)
    plt.title('Pressure profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r, rho = density_profile_m31(20)
    _, rhow = density_profile_m31(20, weighted=True)
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, rho, label = 'bin size = 20kpc/h', color = 'magenta')
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, rhow, label = 'density weighted', color = 'purple')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('density [g/cm^3]', fontsize = 15)
    plt.title('Mass density profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r, covf_C1 = covering_fraction_m31('C II 1335', 1e9, 20)
    r, covf_C2 = covering_fraction_m31('C II 1335', 1e12, 20)
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_C1, label = 'bin size = 20kpc/h\n N > 1e9', color = 'olive')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_C2, label = 'bin size = 20kpc/h\n N > 1e12', color = 'green')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for C II 1036', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, covf_S1 = covering_fraction_m31('Si III 1206', 1e9, 20)
    r, covf_S2 = covering_fraction_m31('Si III 1206', 1e12, 20)
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S1, label = 'bin size = 20kpc/h\n N > 1e9', color = 'pink')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S2, label = 'bin size = 20kpc/h\n N > 1e12', color = 'violet')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for Si III 1206', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, NS3 = column_density_profile_m31('Si III 1206', 50)
    r, NS3w = column_density_profile_m31('Si III 1206', 50, weighted=True)
    r, NS2 = column_density_profile_m31('Si II 1260', 50)
    r, NS2w = column_density_profile_m31('Si II 1260', 50, weighted=True)
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, NS3, label = 'Si III 1206', color = 'purple')
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, NS3w, label = 'density weighted\n Si III 1206', color = 'violet', ls = '--')
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, NS2, label = 'Si II 1260', color = 'red')
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, NS2w, label = 'density weighted\n Si II 1260', color = 'coral', ls = '--')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$N$ [cm$^{-2}$]', fontsize = 15)
    plt.title('Column density profiles', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, covf_S3 = covering_fraction_m31('Si III 1206', 1e9, 10)
    r, covf_S2 = covering_fraction_m31('Si II 1260', 1e9, 10)
    r, covf_S4 = covering_fraction_m31('Si IV 1403', 1e9, 10)
    r, covf_C2 = covering_fraction_m31('C II 1335', 1e9, 10)
    r, covf_C4 = covering_fraction_m31('C IV 1551', 1e9, 10)
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S3, label = 'Si III 1206')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S2, label = 'Si II 1260')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S4, label = 'Si IV 1403')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_C2, label = 'C II 1335')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_C4, label = 'C IV 1551')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for $N > 10^{12}$ cm$^{-2}$, bin size = 10kpc/h', fontsize = 20)
    plt.grid()
    plt.legend()
