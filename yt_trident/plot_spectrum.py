# -*- coding: utf-8 -*-
"""
Script to recieve absorption spectrum, find lines and plot each one as a
function of velocity
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy.signal import find_peaks

<<<<<<< Updated upstream
fname = './spectra_C_Si_2Mpc_LG/spec_990_0.70_4.19.txt'
=======
fname = './spec_990_0.70_4.19.txt'
>>>>>>> Stashed changes

wavelength, tau, flux, flux_error = np.loadtxt(fname=fname, delimiter=' ',
                                    skiprows=1, unpack=True)

def plot_all():

    fig = plt.figure()

    plt.plot(wavelength, flux)
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Relative flux')
    plt.grid(True)

def plot_peaks():

    plt.scatter(wavelength[peak_indices], flux[peak_indices], c='red')

peak_indices, _ = find_peaks(flux, height=(None, 0), distance=50, threshold=0.1)


plot_all()
plot_peaks()
