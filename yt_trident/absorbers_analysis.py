# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
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

    r = get_distances(df, line)
    N1 = df1['N'][::-1]
    N2 = df2['N'][::-1]

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
