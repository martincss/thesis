# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import glob
from tools import HUBBLE_2Mpc_LG, K_BOLTZMANN


def get_distances(df, line):
    """
    Returns radial distance along sightline for a given absorber dataframe and
    absorption line.
    Distance measured from ray_end to ray_start (i.e. from mw to emission point)

    Parameters
    ----------
    df: absorber dataframe
    line: line string (e.g. 'C II 1036')

    Returns
    -------
    r: radial distances from ray_end
    """

    dfc = df[ df['Line']== line ]

    r = np.cumsum(dfc['dl'][::-1])

    return r

def get_attribute_by_distance(df, line, attribute):
    """
    Retrieves absorber attribute (e.g. N) orderded by radial distance along
    sightline, for a given absorber dataframe (df) and absorption line (line)

    Parameters
    ----------
    df: absorber dataframe
    line: line string (e.g. 'C II 1036')

    Returns
    -------
    r: radial distances from ray_end
    att: selected attribute from dataframe
    """

    dfc =  df[ df['Line']== line ]

    r = get_distances(df, line)
    att = dfc[attribute][::-1]

    return r, att

def get_ratio_by_distance(df, line1, line2):
    """
    Retrieves absorber column density ratio between line1 and line2, orderded
    by radial distance along sightline, for a given absorber dataframe (df).

    Parameters
    ----------
    df: absorber dataframe
    line1: numerator line string e.g. 'C II 1036'
    line2: denominator line string


    Returns
    -------
    r: radial distances from ray_end
    ratio: selected column density ratio from dataframe columns
    """

    df1 =  df[ df['Line']== line1 ]
    df2 =  df[ df['Line']== line2 ]

    r = get_distances(df, line1)
    N1 = np.asarray(df1['N'][::-1])
    N2 = np.asarray(df2['N'][::-1])

    ratio = N1/N2

    return r, ratio


def radial_profile(absorbers_directory, line, attribute, r_max, delta_r):
    """
    Calculates radial profile of absorber attribute (e.g. N) for a given
    absorption line

    Parameters
    ----------
    absorbers_directory: string
        path to directory
    line: string
        line string e.g. 'C II 1036'
    attribute: string
        column name from absorber datafile
    r_max: float
        maximum radial distance
    delta_r: float
        distance bin size


    Returns
    -------
    r_bins: array
        bin edges for radial distances
    profile: array

    """

    r_bins = np.arange(0, r_max+delta_r, delta_r)
    counts_per_bin = np.zeros(len(r_bins)-1)
    profile = np.zeros(len(r_bins)-1)

    for handle in glob.glob(absorbers_directory + 'abs*'):

        df = pd.read_csv(handle, skiprows=1)
        r, att = get_attribute_by_distance(df, line, attribute)

        counts_per_bin += np.histogram(r, r_bins)[0]
        profile += np.histogram(r, r_bins, weights=att)[0]

    return r_bins, profile/counts_per_bin


def weighted_radial_profile(absorbers_directory, line, attribute, r_max,
                            delta_r):
    """
    Calculates density weighted radial profile of absorber attribute (e.g. N)
    for a given absorption line

    Parameters
    ----------
    absorbers_directory: string
        path to directory
    line: string
        line string e.g. 'C II 1036'
    attribute: string
        column name from absorber datafile
    r_max: float
        maximum radial distance
    delta_r: float
        distance bin size


    Returns
    -------
    r_bins: array
        bin edges for radial distances
    profile: array

    """

    r_bins = np.arange(0, r_max+delta_r, delta_r)
    density_per_bin = np.zeros(len(r_bins)-1)
    profile = np.zeros(len(r_bins)-1)

    for handle in glob.glob(absorbers_directory + 'abs*'):

        df = pd.read_csv(handle, skiprows=1)
        r, att = get_attribute_by_distance(df, line, attribute)
        _, dens = get_attribute_by_distance(df, line, 'rho')

        density_per_bin += np.histogram(r, r_bins, weights=dens)[0]
        profile += np.histogram(r, r_bins, weights=att*dens)[0]

    return r_bins, profile/density_per_bin


def radial_profile_2k(line, attribute, delta_r, weighted):
    """
    Radial profile wrapper for 2Mpc_LG to mw at 2000kpc/h
    """

    if weighted:

        return weighted_radial_profile(
                        absorbers_directory='./absorbers_2Mpc_LG_to_mw_2000/',
                        line=line, attribute=attribute, r_max=2000,
                        delta_r=delta_r)
    else:

        return radial_profile(
                        absorbers_directory='./absorbers_2Mpc_LG_to_mw_2000/',
                        line=line, attribute=attribute, r_max=2000,
                        delta_r=delta_r)

def column_density_profile_2k(line, delta_r, weighted=False):

    return radial_profile_2k(line, 'N', delta_r, weighted)


def density_profile_2k(delta_r, weighted=False):

    return radial_profile_2k('C II 1036', 'rho', delta_r, weighted)


def pressure_profile_2k(delta_r, weighted=False):

    return radial_profile_2k('C II 1036', 'p', delta_r, weighted)


def temperature_profile_2k(delta_r, weighted=False):

    return radial_profile_2k('C II 1036', 'T', delta_r, weighted)



def covering_fraction(absorbers_directory, line, N_thresh, r_max, delta_r):
    """
    Calculates radial covering_fraction of absorbers with column_density
    exceeding N_thresh for a given absorption line

    Parameters
    ----------
    absorbers_directory: string
        path to directory
    line: string
        line string e.g. 'C II 1036'
    N_thresh: float
        min column density value for absorber to be counted
    r_max: float
        maximum radial distance
    delta_r: float
        distance bin size


    Returns
    -------
    r_bins: array
        bin edges for radial distances
    covf: array


    """

    r_bins = np.arange(0, r_max+delta_r, delta_r)
    counts_per_bin = np.zeros(len(r_bins)-1)
    absorbers_per_bin = np.zeros(len(r_bins)-1)

    for handle in glob.glob(absorbers_directory + 'abs*'):

        df = pd.read_csv(handle, skiprows=1)
        r, N = get_attribute_by_distance(df, line, 'N')

        counts_per_bin += np.histogram(r, r_bins)[0]
        absorbers_per_bin += np.histogram(r, r_bins,
                                weights=np.asarray(N > N_thresh,dtype='int'))[0]

    covf = absorbers_per_bin/counts_per_bin

    return r_bins, covf


def covering_fraction_2k(line, N_thresh, delta_r):
    """
    Covering_fraction wrapper for 2Mpc_LG to mw at 2000kpc/h
    """

    return covering_fraction('./absorbers_2Mpc_LG_to_mw_2000/', line=line,
                             N_thresh=N_thresh, r_max=2000, delta_r=delta_r)


def column_density_ratio_profile(absorbers_directory,
                                 line1, line2, r_max, delta_r):
    """
    Calculates radial profile of column_density ratio for given absorption lines

    Parameters
    ----------
    absorbers_directory: string
        path to directory
    line1: string
        numerator line string e.g. 'C II 1036'
    line2: string
        denominator line string
    r_max: float
        maximum radial distance
    delta_r: float
        distance bin size


    Returns
    -------
    r_bins: array
        bin edges for radial distances
    profile: array

    """

    r_bins = np.arange(0, r_max+delta_r, delta_r)
    counts_per_bin = np.zeros(len(r_bins)-1)
    profile = np.zeros(len(r_bins)-1)

    for handle in glob.glob(absorbers_directory + 'abs*'):

        df = pd.read_csv(handle, skiprows=1)
        r, ratio = get_ratio_by_distance(df, line1, line2)
        ratio = np.nan_to_num(ratio)

        counts_per_bin += np.histogram(r, r_bins)[0]
        profile += np.histogram(r, r_bins, weights=ratio)[0]

    return r_bins, profile/counts_per_bin


def cold_ratio_profile_2k(line1, line2, delta_r):

    return column_density_ratio_profile('./absorbers_2Mpc_LG_to_mw_2000/',
                                        line1=line1, line2=line2, r_max=2000,
                                        delta_r=delta_r)


if __name__=='__main__':

    plt.figure()
    r, ratio = cold_ratio_profile_2k('Si II 1260', 'Si III 1206', 20)
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, ratio, label = 'bin size = 20kpc')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('Column density ratio', fontsize = 15)
    plt.title('Si II / Si III', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, T = temperature_profile_2k(20)
    _, Tw = temperature_profile_2k(20, weighted=True)
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, T, label = 'bin size = 20kpc', color = 'red')
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, Tw, label = 'denisty weighted', color = 'crimson', ls = '--')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('Temperature [K]', fontsize = 15)
    plt.title('Temperature profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r, p = pressure_profile_2k(20)
    _, pw = pressure_profile_2k(20, weighted=True)
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, p/K_BOLTZMANN, label = 'bin size = 20kpc', color = 'blue')
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, pw/K_BOLTZMANN, label = 'density weighted', color = 'cyan')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('pressure/k [K/cm^3]', fontsize = 15)
    plt.title('Pressure profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r, rho = density_profile_2k(20)
    _, rhow = density_profile_2k(20, weighted=True)
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, rho, label = 'bin size = 20kpc', color = 'magenta')
    plt.semilogy(r[:-1]/HUBBLE_2Mpc_LG, rhow, label = 'density weighted', color = 'purple')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('density [g/cm^3]', fontsize = 15)
    plt.title('Mass density profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r, covf_C1 = covering_fraction_2k('C II 1335', 1e9, 20)
    r, covf_C2 = covering_fraction_2k('C II 1335', 1e12, 20)
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_C1, label = 'bin size = 20kpc\n N > 1e9', color = 'olive')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_C2, label = 'bin size = 20kpc\n N > 1e12', color = 'green')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for C II 1036', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, covf_S1 = covering_fraction_2k('Si III 1206', 1e9, 20)
    r, covf_S2 = covering_fraction_2k('Si III 1206', 1e12, 20)
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S1, label = 'bin size = 20kpc\n N > 1e9', color = 'pink')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S2, label = 'bin size = 20kpc\n N > 1e12', color = 'violet')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for Si III 1206', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, NS3 = column_density_profile_2k('Si III 1206', 50)
    r, NS3w = column_density_profile_2k('Si III 1206', 50, weighted=True)
    r, NS2 = column_density_profile_2k('Si II 1260', 50)
    r, NS2w = column_density_profile_2k('Si II 1260', 50, weighted=True)
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
    r, covf_S3 = covering_fraction_2k('Si III 1206', 1e9, 10)
    r, covf_S2 = covering_fraction_2k('Si II 1260', 1e9, 10)
    r, covf_S4 = covering_fraction_2k('Si IV 1403', 1e9, 10)
    r, covf_C2 = covering_fraction_2k('C II 1335', 1e9, 10)
    r, covf_C4 = covering_fraction_2k('C IV 1551', 1e9, 10)
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S3, label = 'Si III 1260')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S2, label = 'Si II 1206')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_S4, label = 'Si IV 1403')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_C2, label = 'C II 1335')
    plt.plot(r[:-1]/HUBBLE_2Mpc_LG, covf_C4, label = 'C IV 1551')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for $N > 10^{12}$ cm$^{-2}$, bin size = 10kpc', fontsize = 20)
    plt.grid()
    plt.legend()
