# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()
import yt
yt.enable_parallelism()
import trident
from tools import get_line, line_table
from R_N_fig1 import load_or_make_spectrum, plot_line

line_list = ['C II', 'C IV', 'Si III', 'Si II']
bandwidth = 6

rays_directory = './rays_2Mpc_LG_from_outside_mw/'
spectra_directory = './spectra_C_Si_2Mpc_LG_from_outside_mw/'
figs_directory = './R_N_Fig1_by_distance_2MpcLG_outside_mw/'

distances = np.linspace(10, 700, 100)
distances_detail = np.linspace(1, 10, 50)
distances_more_detail = np.linspace(0, 1, 50)
close_up_050 = np.linspace(0.36, 0.51, 50)

all_distances = np.concatenate((distances_more_detail, distances_detail, distances))

def sightlines_filenames(distance):
    """
    Return filenames of ray to m31 and away for the corresponding distance
    (hard coded af, it can maybe be changed)
    """
    # WARNING: CHECK FILENAME FOR DECIMAL PLACES IN FORMAT, NOT ALL SAMPLES EQUAL

    # when using distances_detail or others



    ray_to_m31_filename = 'ray_{:.3f}_0.70_4.19.h5'.format(distance)

    ray_away_filename = 'ray_{:.3f}_1.05_1.40.h5'.format(distance)

    # when using 'distances'
    #ray_to_m31_filename = 'ray_{:.0f}_0.70_4.19.h5'.format(distance)

    #ray_away_filename = 'ray_{:.0f}_1.05_1.40.h5'.format(distance)


    return [ray_to_m31_filename, ray_away_filename]

# for some reason won't work well when imported, maybe fix later
def plot_labels(axarr, distance):

    for axes in axarr[-1,:]:

        axes.set_xlabel('Velocity [km/s]', fontsize = 15)

    axarr[2,0].set_ylabel('Relative Flux', fontsize = 15)

    column_titles = ['Ray to M31, r={:.3f}'.format(distance), 'Ray away, r={:.3f}'.format(distance)]
    for i, axes in enumerate(axarr[0,:]):

        axes.set_title('{} \n {}'.format(column_titles[i], 'Si III'), fontsize = 15)


def make_figure(sightlines_list, distance):

    # figsize here for a big display, maybe adjust it to any computer?
    fig, axarr = plt.subplots(len(line_table), len(sightlines_list), figsize=(18,10.5))

    for col_number, ray_filename in enumerate(sightlines_list):

        ray = yt.load(rays_directory + ray_filename)

        wavelength, flux = load_or_make_spectrum(ray, ray_filename, spectra_directory)

        for row_number, line in enumerate(line_table.keys()):

            plot_line(axarr[row_number, col_number], line, wavelength, flux, bandwidth)
            axarr[row_number, col_number].set_xlim(-600,600) # hardcoded af, to be changed...

    plot_labels(axarr, distance)

    # to maximize figure, then tight_layout and save
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()

    fig.set_tight_layout(0.4)
    fig.savefig(figs_directory + 'r_{:09.2F}.png'.format(distance))
    plt.close(fig)


# ~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for r in all_distances[1:]:

    sightlines_list = sightlines_filenames(r)

    make_figure(sightlines_list, r)
