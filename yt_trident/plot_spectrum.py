# -*- coding: utf-8 -*-
"""
Script to recieve absorption spectrum, find lines and plot each one as a
function of velocity
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy.signal import find_peaks

fname = './spectra_C_Si_2Mpc_LG/spec_0.020_0.70_4.19.txt'

wavelength, tau, flux, flux_error = np.loadtxt(fname=fname, delimiter=' ',
                                    skiprows=1, unpack=True)

def plot_all():

    fig = plt.figure()

    plt.plot(wavelength, flux, 'crimson')
    plt.xlabel('Wavelength [A]', fontsize=15)
    plt.ylabel('Relative flux',  fontsize=15)
    plt.title('Transmittance',  fontsize=20)
    plt.grid(True)
#
# def plot_peaks():
#
#     plt.scatter(wavelength[peak_indices], flux[peak_indices], c='red')
#
# peak_indices, _ = find_peaks(flux, height=(None, 0), distance=50, threshold=0.1)


plot_all()
# plot_peaks()
