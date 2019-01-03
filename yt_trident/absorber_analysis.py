# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import glob


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

def radial_profile_2k(line, attribute, delta_r):
    """
    Radial profile wrapper for 2Mpc_LG to mw at 2000kpc
    """

    return radial_profile(absorbers_directory='./absorbers_2Mpc_LG_to_mw_2000/',
            line=line, attribute=attribute, r_max=2000, delta_r=delta_r)

def column_density_profile_2k(line, delta_r):

    return radial_profile_2k(line, 'N', delta_r)


def density_profile_2k(delta_r):

    return radial_profile_2k('C II 1036', 'rho', delta_r)


def pressure_profile_2k(delta_r):

    return radial_profile_2k('C II 1036', 'p', delta_r)


def temperature_profile_2k(delta_r):

    return radial_profile_2k('C II 1036', 'T', delta_r)



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
        absorbers_per_bin += np.histogram(r, r_bins, weights=np.asarray(N > N_thresh,dtype='int'))[0]

    covf = absorbers_per_bin/counts_per_bin

    return r_bins, covf


def covering_fraction_2k(line, N_thresh, delta_r):
    """
    Covering_fraction wrapper for 2Mpc_LG to mw at 2000kpc
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
    plt.semilogy(r[:-1], ratio, label = 'bin size = 20kpc')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('Column density ratio', fontsize = 15)
    plt.title('Si II / Si III', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, T = temperature_profile_2k(20)
    plt.semilogy(r[:-1], T, label = 'bin size = 20kpc')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('Temperature [K]', fontsize = 15)
    plt.title('Temperature profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r, p = pressure_profile_2k(20)
    plt.semilogy(r[:-1], p, label = 'bin size = 20kpc', color = 'blue')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('pressure [dyne/cm^2]', fontsize = 15)
    plt.title('Pressure profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r, rho = density_profile_2k(20)
    plt.semilogy(r[:-1], rho, label = 'bin size = 20kpc', color = 'magenta')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('density [g/cm^3]', fontsize = 15)
    plt.title('Mass density profile', fontsize = 20)
    plt.grid()
    plt.legend()

    plt.figure()
    r, covf_C1 = covering_fraction_2k('C II 1335', 1e9, 20)
    r, covf_C2 = covering_fraction_2k('C II 1335', 1e12, 20)
    plt.plot(r[:-1], covf_C1, label = 'bin size = 20kpc\n N > 1e9', color = 'olive')
    plt.plot(r[:-1], covf_C2, label = 'bin size = 20kpc\n N > 1e12', color = 'green')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for C II 1036', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, covf_S1 = covering_fraction_2k('Si III 1206', 1e9, 20)
    r, covf_S2 = covering_fraction_2k('Si III 1206', 1e12, 20)
    plt.plot(r[:-1], covf_S1, label = 'bin size = 20kpc\n N > 1e9', color = 'pink')
    plt.plot(r[:-1], covf_S2, label = 'bin size = 20kpc\n N > 1e12', color = 'violet')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$f$', fontsize = 15)
    plt.title('Covering fraction for Si III 1206', fontsize = 20)
    plt.grid()
    plt.legend()


    plt.figure()
    r, NC = column_density_profile_2k('Si III 1206', 20)
    r, NS = column_density_profile_2k('C II 1335', 20)
    plt.semilogy(r[:-1], NS, label = 'bin size = 20kpc\n Si III 1206', color = 'purple')
    plt.semilogy(r[:-1], NC, label = 'bin size = 20kpc\n C II 1335', color = 'red')
    plt.xlabel('Distance [kpc]', fontsize = 15)
    plt.ylabel('$N$ [cm$^{-2}$]', fontsize = 15)
    plt.title('Column density profiles', fontsize = 20)
    plt.grid()
    plt.legend()
