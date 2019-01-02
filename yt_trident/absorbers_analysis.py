# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_distances(df, line):
    """
    Returns radial distance along sightline for a given absorber dataframe (df)
    and absorption line (line).
    Distance measured from ray_end to ray_start (i.e. from mw to emission point)
    """

    dfc = df[ df['Line']== line ]

    r = np.cumsum(dfc['dl'][::-1])

    return r

def get_attribute_by_distance(df, line, attribute):
    """
    Retrieves absorber attribute (e.g. N) orderded by radial distance along
    sightline, for a given absorber dataframe (df) and absorption line (line)

    Returns
    -------
    r: radial distances from ray_end
    att: selected attribute from dataframe
    """

    dfc =  df[ df['Line']== line ]

    r = get_distances(df, line)
    att = dfc[attribute][::-1]

    return r, att
