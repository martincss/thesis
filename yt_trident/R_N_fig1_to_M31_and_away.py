# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()
import yt
from tools import get_line, HUBBLE_2Mpc_LG
from R_N_fig1 import load_or_make_spectrum, plot_line


line_table = {'Si III 1206':1206.5, 'Si II 1193':1193.29,
              'C II 1335': 1334.532, 'C IV 1548':1548.19}

bandwidth = 6

# rays_directory = './rays_2Mpc_LG_from_outside_mw/'
# spectra_directory = './spectra_C_Si_2Mpc_LG_from_outside_mw/'
# figs_directory = './R_N_Fig1_by_distance_2MpcLG_outside_mw/'
#
# rays_directory = './rays_test/'
# spectra_directory = './rays_test/'
# figs_directory = './rays_test/'

rays_directory = './rays_2Mpc_LG_to_mw_2000_wrt_mwcenter'
absorbers_directory = './absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter/'
spectra_directory = './spectra_2Mpc_LG_to_m31_and_away/'
figs_directory = './R_N_Fig1_by_distance_2MpcLG_new/'

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



    ray_to_m31_filename = 'ray_{:.3f}_0.698_4.189.h5'.format(distance)

    ray_away_filename = 'ray_{:.3f}_1.047_1.396.h5'.format(distance)

    # when using 'distances'
    #ray_to_m31_filename = 'ray_{:.0f}_0.70_4.19.h5'.format(distance)

    #ray_away_filename = 'ray_{:.0f}_1.05_1.40.h5'.format(distance)


    return [ray_to_m31_filename, ray_away_filename]

# for some reason won't work well when imported, maybe fix later
def plot_labels(axarr, distance):

    for axes in axarr[-1,:]:

        axes.set_xlabel('Velocity [km/s]', fontsize = 15)

    axarr[2,0].set_ylabel('Relative Flux', fontsize = 15)

    column_titles = ['Ray to M31, r={:.3f}kpc'.format(distance/HUBBLE_2Mpc_LG),
                     'Ray away, r={:.3f}kpc'.format(distance/HUBBLE_2Mpc_LG)]
    for i, axes in enumerate(axarr[0,:]):

        axes.set_title('{} \n {}'.format(column_titles[i], 'Si III 1206'), fontsize = 15)


def make_figure(args):

    distance, sightlines_list = args

    # figsize here for a big display, maybe adjust it to any computer?
    fig, axarr = plt.subplots(len(line_table), len(sightlines_list), figsize=(18,10.5))

    for col_number, ray_filename in enumerate(sightlines_list):

        ray = yt.load(rays_directory + ray_filename)

        wavelength, flux = load_or_make_spectrum(ray, ray_filename, spectra_directory)

        for row_number, line in enumerate(line_table.keys()):

            plot_line(axarr[row_number, col_number], line, wavelength, flux, bandwidth, ray)
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

if __name__=='__main__':
    for r in all_distances[1:]:

        sightlines_list = sightlines_filenames(r)

        make_figure((r, sightlines_list))
