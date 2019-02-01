# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
#plt.ion()
import yt
import trident
from tools import get_line, get_absorber_chars, get_absorber_chars_from_file,\
                  absorber_region_2Mpc_LG, line_table
import pandas as pd

# np.mean(mean_denisities) = 2.528e-6
# ray_1000_1.0_5.6.h5 has mean_density of 1.41e-6
# ray_1000_0.3_2.8.h5 has mean_density of 2.54e-6
# ray_1000_1.7_1.4.h5 has mean_density of 4.80e-6


#sightlines_list = ['ray_1000_1.0_5.6.h5', 'ray_1000_0.3_2.8.h5', 'ray_1000_1.7_1.4.h5']
#sightlines_list = ['ray_1000_0.70_4.19.h5', 'ray_1000_1.05_1.40.h5', 'ray_1000_2.8_2.1.h5']

#sightlines_list = ['ray_2000.000_0.698_4.115.h5', 'ray_2000.000_1.059_1.881.h5', 'ray_2000.000_2.837_2.574.h5']
sightlines_list = ['ray_1000.000_0.698_4.189.h5', 'ray_2000.000_0.698_4.189.h5', 'ray_1000.000_1.047_1.396.h5', 'ray_2000.000_1.047_1.396.h5']

line_list = ['C II', 'C IV', 'Si III', 'Si II']
line_keys = ['Si III 1206', 'Si II 1193','C II 1335', 'C IV 1548']
bandwidth = 8

# rays_directory = './rays_2Mpc_LG_from_mw/'
# spectra_directory = './spectra_C_Si_2Mpc_LG_from_mw/'
# absorbers_directory = './absorbers_2Mpc_LG_from_mw/'

# rays_directory = './rays_2Mpc_LG_to_mw_2000_wrt_mwcenter/'
# spectra_directory = './spectra_C_Si_2Mpc_LG_to_mw_2000_wrt_mwcenter/'
# absorbers_directory = './absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter/'

rays_directory = './rays_test/'
spectra_directory = './rays_test/'
absorbers_directory = './rays_test/'



def make_spectrum(ray, filename):

    sg = trident.SpectrumGenerator(lambda_min = 1000, lambda_max = 1600,
        dlambda = 0.01)

    sg.make_spectrum(ray, lines=line_keys, use_peculiar_velocity=True)
    sg.save_spectrum(filename + '.txt')
    sg.plot_spectrum(filename + '.png')

def load_or_make_spectrum(ray, ray_filename, spectra_directory):

    # this parses the ray_filename extracting the ray and .h5
    # eg. ray_1000_2.33_4.55.h5 ---> _1000_2.33_4.55
    # a little brute, but I guess it can be polished
    spec_filename = 'spec_' + ray_filename[4:-3]

    if not os.path.exists(spectra_directory + spec_filename + '.txt'):

        make_spectrum(ray, spectra_directory + spec_filename)


    wavelength, _, flux, _ = np.loadtxt(fname=spectra_directory+spec_filename+'.txt',
                            delimiter=' ', skiprows=1, unpack=True)

    return wavelength, flux


def plot_line(ax, line, wavelength, flux, bandwidth, ray):
    """
    Given a line from line_table (i.e. 'C II'), plots the relative flux as a
    function of LSR velocity.
    """

    ray_filename = str(ray)
    # abs_filename = 'abs_' + ray_filename[4:-3] + '.txt'
    #
    # if not os.path.exists(absorbers_directory + abs_filename):
    #
    #     lambda_0, N, T, absorber_position = get_absorber_chars(ray, line,
    #                                                            line_list)
    #
    # else:
    #
    #     lambda_0, N, T, absorber_position = get_absorber_chars_from_file(
    #                                         absorbers_directory + abs_filename,
    #                                         line)
    #
    # absorber_region = absorber_region_2Mpc_LG(absorber_position)
    lambda_0 = line_table[line]

    velocity, flux = get_line(lambda_0, wavelength=wavelength, flux=flux,
                    wavelength_interval=bandwidth)

    ax.plot(velocity, flux)#, label = 'N = {:.2e}\nT = {:.2e}\n{}'.format(N, T,
#                                                              absorber_region))

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

        axes.set_title('{} \n {}'.format(sightlines_list[i], 'Si III 1206'), fontsize = 15)

    #fig.suptitle('Absorption lines along sightlines', fontsize = 15)


if __name__ == '__main__':

    fig, axarr = plt.subplots(len(line_keys), len(sightlines_list))

    for col_number, ray_filename in enumerate(sightlines_list):

        ray = yt.load(rays_directory + ray_filename)

        wavelength, flux = load_or_make_spectrum(ray, ray_filename, spectra_directory)

        for row_number, line in enumerate(line_keys):

            plot_line(axarr[row_number, col_number], line,
                      wavelength, flux, bandwidth, ray)

    plot_labels(sightlines_list, axarr)
