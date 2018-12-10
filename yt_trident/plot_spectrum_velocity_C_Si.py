# -*- coding: utf-8 -*-
"""
Script to plot C and Si absorption lines as a function of velocity.
Recieves spectrum generated from 'C II', 'C IV', 'Si III', 'Si II' lines.
"""

import numpy as np
import matplotlib.pyplot as plt
import yt
from R_N_fig1 import plot_line, load_or_make_spectrum
plt.ion()

rays_directory = 'rays_2Mpc_LG_from_mw/'
ray_filename = 'ray_1000_0.70_4.19.h5'
spectra_directory = 'spectra_C_Si_2Mpc_LG_from_mw'


line_keys = ['Si III 1206', 'Si II 1190', 'Si II 1260','C II 1335', 'C IV 1548']

bandwidth = 16

ray = yt.load(rays_directory + ray_filename)
wavelength, flux = load_or_make_spectrum(ray, ray_filename, spectra_directory)


def plot_single_line(line):
    """
    Line: any key from line_keys
    """

    fig, ax = plt.subplots()

    plot_line(ax, line, wavelength, flux, bandwidth, ray)
