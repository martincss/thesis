# -*- coding: utf-8 -*-
"""
Script to plot C and Si absorption lines as a function of velocity.
Recieves spectrum generated from 'C II', 'C IV', 'Si III', 'Si II' lines.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
LIGHTSPEED = 299792 # in km/s

fname = 'spectra_C_Si_2Mpc_LG_from_mw/spec_990_0.70_4.19.txt'

wavelength, tau, flux, flux_error = np.loadtxt(fname=fname, delimiter=' ',
                                    skiprows=1, unpack=True)

line_table = {'Si III':1206, 'Si IIa':1190, 'Si IIb':1260, 'C II': 1334.53,
              'C IV':1548}

bandwidth = 16

def get_line(line, wavelength_interval):
    """
    Given a line from line_table (i.e. 'C II') and a wavelength bandwidth,
    isolates the line's wavelengths and flux, and converts wavelength to
    velocity in LSR
    Return velocity and flux arrays for the selected line

    wavelength_interval in angstroms
    """

    lambda_0 = line_table[line]
    # quedó re cabeza ese indexeado en la salida del where, ver como se arregla
    try:
        central_index = np.where(wavelength == lambda_0)[0][0]
        right_index = np.where(wavelength == int(lambda_0 + wavelength_interval/2))[0][0]
        left_index = np.where(wavelength == int(lambda_0 - wavelength_interval/2))[0][0]

        wavelength_section = wavelength[left_index:right_index]
        delta_lambda = wavelength_section - lambda_0
        flux_section = flux[left_index:right_index]
        velocity = LIGHTSPEED * delta_lambda/lambda_0

    except:
        velocity = np.array([])
        flux_section = np.array([])
        print('Wavelength not in spectrum')

    return velocity, flux_section

def plot_line(line):
    """
    Given a line from line_table (i.e. 'C II'), plots the relative flux as a
    function of LSR velocity.
    """

    velocity, flux = get_line(line, wavelength_interval=bandwidth)

    plt.figure()
    plt.plot(velocity, flux, label = '$\\lambda = ${}'.format(line_table[line]))

    plt.xlabel('Velocity [km/s]', fontsize = 15)
    plt.ylabel('Relative Flux', fontsize = 15)
    plt.title('{}'.format(line), fontsize = 15)
    plt.legend()
    plt.grid(True)
