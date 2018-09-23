# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

fname = 'spec_raw.txt'

wavelength, tau, flux, flux_error = np.loadtxt(fname=fname, delimiter=' ',
                                    skiprows=1, unpack=True)

def plot_all():

    fig = plt.figure()

    plt.plot(wavelength, flux)
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Relative flux')
    plt.grid(True)
