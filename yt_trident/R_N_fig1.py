# -*- coding: utf-8 -*-
import numpy as np
import glob
import matplotlib.pyplot as plt
plt.ion()
import yt
from tools import get_line, line_table
import pandas as pd


line_list = ['C II', 'C IV', 'Si III', 'Si II']
bandwidth = 4

def make_spectrum(ray, filename):

    sg = trident.SpectrumGenerator(lambda_min = 1150, lambda_max = 1600,
        dlambda = 0.01)

    sg.make_spectrum(ray, lines=line_list)
    sg.save_spectrum(filename + '.txt')
    sg.plot_spectrum(filename + '.png')


def plot_line(ax, line, wavelength, flux):
    """
    Given a line from line_table (i.e. 'C II'), plots the relative flux as a
    function of LSR velocity.
    """

    velocity, flux = get_line(line, wavelength=wavelength, flux=flux,
                    wavelength_interval=bandwidth)

    ax.plot(velocity, flux, label = '$\\lambda = ${}'.format(line_table[line]))

    ax.set_xlabel('Velocity [km/s]', fontsize = 15)
    ax.set_ylabel('Relative Flux', fontsize = 15)
    ax.set_title('{}'.format(line), fontsize = 15)
    ax.legend()
    ax.grid(True)
