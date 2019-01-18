# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()
import yt
yt.enable_parallelism()
import trident
from tools import get_line, get_absorber_chars, get_absorber_chars_from_file, \
                  absorber_region_2Mpc_LG, HUBBLE_2Mpc_LG
from R_N_fig1 import make_spectrum, load_or_make_spectrum
import pandas as pd


b_values_phys = np.array([20, 50, 100, 200, 500])
b_values_code = b_values_phys * HUBBLE_2Mpc_LG

sightlines_list = ['ray_away_b_{:.2f}.h5'.format(b) for b in b_values_code]

line_list = ['C II', 'C IV', 'Si III', 'Si II', 'O VI']
line_keys = ['Si III 1206', 'Si II 1260', 'C II 1335', 'C IV 1548', 'O VI 1038']

#line_list = ['C II', 'C IV', 'Si III', 'Si II']
#line_keys = ['Si III 1206', 'Si II 1190', 'Si II 1260','C II 1335', 'C IV 1548']
bandwidth = 8

rays_directory = './rays_2Mpc_LG_to_mw_impact/'
spectra_directory = './spectra_C_Si_2Mpc_LG_to_mw_impact/'
absorbers_directory = './absorbers_2Mpc_LG_to_mw_impact/'



def plot_line(ax, line, wavelength, flux, bandwidth, ray):
    """
    Given a line from line_table (i.e. 'C II'), plots the relative flux as a
    function of LSR velocity.
    """

    ray_filename = str(ray)
    abs_filename = 'abs_' + ray_filename[4:-3] + '.txt'

    if not os.path.exists(absorbers_directory + abs_filename):

        lambda_0, N, T, absorber_position = get_absorber_chars(ray, line,
                                                               line_list)

    else:

        lambda_0, N, T, absorber_position = get_absorber_chars_from_file(
                                            absorbers_directory + abs_filename,
                                            line)

    absorber_region = absorber_region_2Mpc_LG(absorber_position)

    velocity, flux = get_line(lambda_0, wavelength=wavelength, flux=flux,
                    wavelength_interval=bandwidth)

    ax.plot(velocity, flux, label = 'N = {:.2e}\nT = {:.2e}\n{}'.format(N, T,
                                                               absorber_region))

    #ax.set_xlabel('Velocity [km/s]', fontsize = 15)
    #ax.set_ylabel('Relative Flux', fontsize = 15)
    ax.set_title('{}'.format(line), fontsize = 15)
    ax.set_xlim(-430,430)
    ax.set_ylim(0,1.1)
    ax.legend(loc='best')
    ax.grid(True)


def plot_labels(sightlines_list, axarr):

    for axes in axarr[-1,:]:

        axes.set_xlabel('Velocity [km/s]', fontsize = 15)

    axarr[2,0].set_ylabel('Relative Flux', fontsize = 15)

    for i, axes in enumerate(axarr[0,:]):

        axes.set_title('b = {} \n {}'.format(sightlines_list[i], 'Si III 1206'), fontsize = 15)

    #fig.suptitle('Absorption lines along sightlines', fontsize = 15)


if __name__ == '__main__':

    fig, axarr = plt.subplots(len(line_keys), len(sightlines_list))

    for col_number, ray_filename in enumerate(sightlines_list):

        ray = yt.load(rays_directory + ray_filename)

        wavelength, flux = load_or_make_spectrum(ray, ray_filename, spectra_directory)

        for row_number, line in enumerate(line_keys):

            plot_line(axarr[row_number, col_number], line,
                      wavelength, flux, bandwidth, ray)

    plot_labels(b_values_phys, axarr)
