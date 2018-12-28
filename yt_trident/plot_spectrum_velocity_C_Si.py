# -*- coding: utf-8 -*-
"""
Script to plot C and Si absorption lines as a function of velocity.
Recieves spectrum generated from 'C II', 'C IV', 'Si III', 'Si II' lines.
"""

import numpy as np
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
import yt
from R_N_fig1 import plot_line, load_or_make_spectrum
=======
import pandas as pd
>>>>>>> Stashed changes
plt.ion()

<<<<<<< Updated upstream
rays_directory = 'rays_2Mpc_LG_from_mw/'
ray_filename = 'ray_1000_0.70_4.19.h5'
spectra_directory = 'spectra_C_Si_2Mpc_LG_from_mw'
=======
fname = 'spectra_C_Si_2Mpc_LG_from_mw/spec_990_0.70_4.19.txt'
absorber_filename = 'abs_990_0.70_4.19.txt'
>>>>>>> Stashed changes


<<<<<<< Updated upstream
line_keys = ['Si III 1206', 'Si II 1190', 'Si II 1260','C II 1335', 'C IV 1548']

bandwidth = 16
=======
df = pd.read_csv(absorber_filename, skiprows=1)
line_df = df[df['Line']=='C II 1335']
index = np.argmax(line_df['N'])
lambda_obs = line_df['lambda'][index]

lambda_obs = 1335.09

line_table = {'Si III':1206, 'Si IIa':1190, 'Si IIb':1260, 'C II': lambda_obs,
              'C IV':1548}


bandwidth = 10
>>>>>>> Stashed changes

ray = yt.load(rays_directory + ray_filename)
wavelength, flux = load_or_make_spectrum(ray, ray_filename, spectra_directory)


def plot_single_line(line):
    """
    Line: any key from line_keys
    """

    fig, ax = plt.subplots()

    plot_line(ax, line, wavelength, flux, bandwidth, ray)
