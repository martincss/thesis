# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob
from tools import HUBBLE_2Mpc_LG, K_BOLTZMANN, handle_in_subsample, line_table,\
                  lambda_to_velocity
import pdb

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

def get_binned_attribute_all_rays(absorbers_directory, line, attribute, r_max,
                                  delta_r):
    """
    Calculates radial binned array of absorber attribute (e.g. N) for a given
    absorption line, for each absorber file (i.e. ray) and returns all of them
    in a list

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
    radial_attribute_list: list of arrays

    """

    r_bins = np.arange(0, r_max+delta_r, delta_r)
    radial_attribute_list = []

    for handle in glob.glob(absorbers_directory + 'abs*'):

        df = pd.read_csv(handle, skiprows=1)
        r, att = get_attribute_by_distance(df, line, attribute)

        # get attribute array into binned radial array
        att_binned = np.histogram(r, r_bins, weights=att)[0]

        radial_attribute_list.append(att_binned)

    return r_bins, radial_attribute_list

def radial_profile_median(absorbers_directory, line, attribute, r_max,
                                  delta_r):

    r, values_binned = get_binned_attribute_all_rays(absorbers_directory, line,
                                                     attribute, r_max, delta_r)

    profile = np.median(np.vstack(values_binned), axis=0)

    return r, profile



def radial_profile_various(absorbers_directory, line, attributes_list, r_max,
                           delta_r):
    """
    Calculates radial profile of absorber attributes (e.g. N) in attributes_list
    for a given absorption line

    Parameters
    ----------
    absorbers_directory: string
        path to directory
    line: string
        line string e.g. 'C II 1036'
    attribute: list of strings
        column names from absorber datafile
    r_max: float
        maximum radial distance
    delta_r: float
        distance bin size


    Returns
    -------
    r_bins: ndarray
        bin edges for radial distances
    profiles: dict of ndarrays
        radial profile for each selected attribute, with the latter as the key

    """

    r_bins = np.arange(0, r_max+delta_r, delta_r)
    counts_per_bin = np.zeros(len(r_bins)-1)
    profiles = {att: np.zeros(len(r_bins)-1) for att in attributes_list}

    for handle in glob.glob(absorbers_directory + 'abs*'):

        df = pd.read_csv(handle, skiprows=1)

        for att in attributes_list:

            r, att_values = get_attribute_by_distance(df, line, att)
            profiles[att] += np.histogram(r, r_bins, weights=att_values)[0]

        counts_per_bin += np.histogram(r, r_bins)[0]


    return r_bins, {att: profiles[att]/counts_per_bin for att in attributes_list}


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



def weighted_radial_profile_various(absorbers_directory, line, attributes_list,
                                    r_max, delta_r):
    """
    Calculates radial profile of absorber attributes (e.g. N) in attributes_list
    for a given absorption line

    Parameters
    ----------
    absorbers_directory: string
        path to directory
    line: string
        line string e.g. 'C II 1036'
    attribute: list of strings
        column names from absorber datafile
    r_max: float
        maximum radial distance
    delta_r: float
        distance bin size


    Returns
    -------
    r_bins: ndarray
        bin edges for radial distances
    profiles: dict of ndarrays
        radial profile for each selected attribute, with the latter as the key

    """

    r_bins = np.arange(0, r_max+delta_r, delta_r)
    density_per_bin = np.zeros(len(r_bins)-1)
    profiles = {att: np.zeros(len(r_bins)-1) for att in attributes_list}

    for handle in glob.glob(absorbers_directory + 'abs*'):

        df = pd.read_csv(handle, skiprows=1)

        r, dens = get_attribute_by_distance(df, line, 'rho')
        density_per_bin += np.histogram(r, r_bins, weights=dens)[0]

        for att in attributes_list:

            _, att_values = get_attribute_by_distance(df, line, att)
            profiles[att] += np.histogram(r, r_bins, weights=att_values*dens)[0]

    return r_bins, {att:profiles[att]/density_per_bin for att in attributes_list}


def density_quotient_radial_profile(absorbers_directory, line, r_max, delta_r):
    """
    Calculates total gas density radial profile by dividing total mass in
    spherical shell by total volume in shell

    Parameters
    ----------
    absorbers_directory: string
        path to directory
    r_max: float
        maximum radial distance
    delta_r: float
        distance bin size


    Returns
    -------
    r_bins: array
        bin edges for radial distances
    density: array

    """
    # NOT YET FINISHED ########################################################

    # r_bins = np.arange(0, r_max+delta_r, delta_r)
    # counts_per_bin = np.zeros(len(r_bins)-1)
    # profile = np.zeros(len(r_bins)-1)
    #
    # for handle in glob.glob(absorbers_directory + 'abs*'):
    #
    #     df = pd.read_csv(handle, skiprows=1)
    #     r, att = get_attribute_by_distance(df, line, attribute)
    #
    #     counts_per_bin += np.histogram(r, r_bins)[0]
    #     profile += np.histogram(r, r_bins, weights=att)[0]
    #
    # return r_bins, profile/counts_per_bin
    pass


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

def covering_fraction_various(absorbers_directory, line_list, N_thresh, r_max,
                              delta_r):
    """
    Calculates radial covering_fraction of absorbers with column_density
    exceeding N_thresh for a given set of absorption lines

    Parameters
    ----------
    absorbers_directory: string
        path to directory
    line: list of strings
        list of line string e.g. ['C II 1036', 'Si III 1206']
    N_thresh: float
        min column density value for absorber to be counted
    r_max: float
        maximum radial distance
    delta_r: float
        distance bin size


    Returns
    -------
    r_bins: ndarray
        bin edges for radial distances
    covf: dict of ndarrays
        covering fraction by distance for each absorption line given, with such
        line as keyz


    """

    r_bins = np.arange(0, r_max+delta_r, delta_r)
    counts_per_bin = {line: np.zeros(len(r_bins)-1) for line in line_list}
    absorbers_per_bin = {line: np.zeros(len(r_bins)-1) for line in line_list}

    for handle in glob.glob(absorbers_directory + 'abs*'):

        df = pd.read_csv(handle, skiprows=1)

        for line in line_list:

            r, N = get_attribute_by_distance(df, line, 'N')
            # pdb.set_trace()
            counts_per_bin[line] += np.histogram(r, r_bins)[0]
            absorbers_per_bin[line] += np.histogram(r, r_bins,
                                weights=np.asarray(N > N_thresh,dtype='int'))[0]

    covf = {line:absorbers_per_bin[line]/counts_per_bin[line] for line in \
            line_list}

    return r_bins, covf




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



def covering_fraction_by_rays(absorbers_directory, line, N_thresh_list):
    """
    Calculates total covering_fraction of absorbers with column_density
    exceeding N_thresh for a given absorption line.
    Counts all sightlines with total (i.e. summed) N greater than the threshold
    and divides it by total number of sightlines; for each N_thresh in the list.

    Parameters
    ----------
    absorbers_directory: string
        path to directory
    line: string
        line string e.g. 'C II 1036'
    N_thresh_list: list of floats
        min column density values for absorber to be counted

    Returns
    -------
    covf: float


    """

    number_of_sightlines = 0
    sightlines_over_thresh = {N_thresh:0 for N_thresh in N_thresh_list}

    for handle in [handle for handle in glob.glob(absorbers_directory + 'abs*')\
                   if handle_in_subsample(handle, amplitude_polar=1)]:

        df = pd.read_csv(handle, skiprows=1)
        r, N = get_attribute_by_distance(df, line, 'N')

        for N_thresh in N_thresh_list:

            if N[(r > 10*HUBBLE_2Mpc_LG) & (r < 300*HUBBLE_2Mpc_LG)].sum() > N_thresh:
            # if N[(r > 10*HUBBLE_2Mpc_LG)].sum() > N_thresh:

                sightlines_over_thresh[N_thresh] += 1

        number_of_sightlines += 1

    covf = {N_thresh: sightlines_over_thresh[N_thresh]/number_of_sightlines
            for N_thresh in N_thresh_list}

    return covf


def covering_fraction_by_rays_one_to_map(args):

    handle, N_thresh_array, line_list, vel_thresh_list, r_min, r_max = args

    counts = {}

    df = pd.read_csv(handle, skiprows=1)

    for line in line_list:

        r, N = get_attribute_by_distance(df, line, 'N')
        r, wavelength = get_attribute_by_distance(df, line, 'lambda')
        vel_spec = lambda_to_velocity(wavelength, line_table[line])

        for vel_thresh in vel_thresh_list:

            counts[(line, vel_thresh)] = \
            1*(N[(r > r_min) & (r < r_max) & \
              (np.abs(vel_spec) > vel_thresh)].sum() > N_thresh_array)

    return counts
