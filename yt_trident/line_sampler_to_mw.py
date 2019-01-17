# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
plt.ion()
import yt
yt.enable_parallelism()
import trident
import gc

from tools import get_2Mpc_LG_dataset, get_mw_center_2Mpc_LG, \
                  ray_start_from_sph, sphere_uniform_grid


# ~~~~~~~~~~~~~~~~~~~~ SETUP ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

snap_file = '../../2Mpc_LG_convert/snapdir_135/snap_LG_WMAP5_2048_135.0'
snap_num = 135
subfind_path = '../../2Mpc_LG'

rays_directory = './rays_2Mpc_LG_to_mw_2000/'
subhalo_rays_directory = './rays_2Mpc_LG_from_subhalos/'

ds = get_2Mpc_LG_dataset()

# from Table 1 in Richter, Nuza, et al (2017)
line_list = ['C II', 'C IV', 'Si III', 'Si II', 'Si IV', 'H']

mw_center = get_mw_center_2Mpc_LG()



# ~~~~~~~~~~~~~~~~~~~~ ACTIONS ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_ray_to_mw(spherical_coords, ray_filename):
    """
    Creates ray with center of MW as ending point (where the observer is),
    and starting point passed by its spherical coordinates from MW center.

    Spherical coordinates of starting point to be passed as iterable.
    """

    ray_start = ray_start_from_sph(mw_center, spherical_coords)

    # for some reason, make_simple_ray overwrites start_position & end_position
    # actually are passed as pointers and changes them to cgs; this can be
    # prevented by passing them as ray_start.copy()
    ray = trident.make_simple_ray(ds,
                                  start_position=ray_start.copy(),
                                  end_position=mw_center.copy(),
                                  data_filename=ray_filename,
                                  fields=['thermal_energy','density'],
                                  lines=line_list,
                                  ftype='Gas')

    return ray


def sample_single_sightline(r, theta, phi):
    """
    Samples ray along the sightline given by the sperical coordinates of
    starting point to mw center as endpoint.
    """

    print('\n NOW SAMPLING r = {}, theta = {}, phi = {} ~~~~~~~~~~~~~~~~~~~'
          ' \n'.format(r, theta, phi))

    ray_filename = rays_directory + \
                   'ray_{:.3f}_{:.3f}_{:.3f}.h5'.format(r, theta, phi)
    make_ray_to_mw((r, theta, phi), ray_filename=ray_filename)



def make_ray_sample(r_interval, theta_interval, phi_interval):
    """
    Given intervals for r, theta and phi, samples rays for each value in such
    intervals (as iterables).
    """

    for r in r_interval:

        for theta in theta_interval:

            for phi in phi_interval:

                sample_single_sightline(r, theta, phi)

                gc.collect()

def make_ray_sample_uniform(r_interval, number_of_sightlines):

    theta_interval, phi_interval = sphere_uniform_grid(number_of_sightlines)

    for r in r_interval:

        for i, (theta, phi) in enumerate(zip(theta_interval, phi_interval)):

            print('\n RAY NUMBER {:d}/{:d} ~~~~~~~~~~~~~~~~~~~'
                  ' \n'.format(i, number_of_sightlines))

            sample_single_sightline(r,theta, phi)

            gc.collect()

#################################################################

def sample_m31_and_away(r_interval):
    """
    Samples rays on fixed directions to m31 and away from m31, varying distance
    to endpoints.
    """

    theta_m31 = 2*pi/9
    phi_m31 = 6*(2*pi)/9

    make_ray_sample(r_interval, [theta_m31], [phi_m31])

    theta_away = 3*pi/9
    phi_away = 2*(2*pi)/9

    make_ray_sample(r_interval, [theta_away], [phi_away])

##################################################################

def ray_to_subhalo(subhalo_idx):
    """
    Samples ray from subhalo center corresponding to subhalo index (subhalo_idx)
    to mw_center.
    """

    subhalo_position = subhalo_center(subfind_path=subfind_path,
                                      snap_num=snap_num,
                                      subhalo_number = subhalo_idx)

    ray_filename = subhalo_rays_directory + \
                   'ray_subhalo_{:02d}.h5'.format(subhalo_idx)

    ray = trident.make_simple_ray(ds,
                                  start_position=subhalo_position.copy(),
                                  end_position=mw_center.copy(),
                                  data_filename=ray_filename,
                                  fields=['thermal_energy','density'],
                                  lines=line_list,
                                  ftype='Gas')

    return ray


def sample_subhalos(number_of_subhalos):
    """
    Generates rays from all subhalos up to number_of_subhalos, except for MW (
    with index = 1).
    """

    for index in range(number_of_subhalos):

        if index != 1:

            print('\n NOW SAMPLING subhalo {:02d} ~~~~~~~~~~~~~~~~~~~'
                  ' \n'.format(index))

            ray_to_subhalo(index)








# ~~~~~~~~~~~~~~~~ RAY SAMPLING ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

r_array = np.array([1000])
theta_array = np.linspace(0, pi, 10)
phi_array = np.linspace(0, 2*pi, 10)

#make_ray_sample(r_array, theta_array, phi_array)

# ~~~~~~~~~~~~~~ RAY SAMPLING M31 ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


distances = np.linspace(0, 1000, 100)
distances_detail = np.linspace(0, 10, 50)
distances_more_detail = np.linspace(0, 1, 50)
close_up_050 = np.linspace(0.36, 0.51, 50)

if __name__ == '__main__':
    #sample_m31_and_away(close_up_050)
    #make_ray_sample_uniform([2000], 500)
    pass
