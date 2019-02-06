# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi as pi
from multiprocessing import Pool, cpu_count
import os
import trident
import gc
import pdb
from tools import get_2Mpc_LG_dataset, get_mw_center_2Mpc_LG, \
                  ray_start_from_sph, sphere_uniform_grid, z_from_distance, \
                  subhalo_center, cart_to_sph


# ~~~~~~~~~~~~~~~~~~~~ SETUP ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rays_directory = './rays_test/'
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
    r, _, _ = spherical_coords

    # for some reason, make_simple_ray overwrites start_position & end_position
    # actually are passed as pointers and changes them to cgs; this can be
    # prevented by passing them as ray_start.copy()
    ray = trident.make_simple_ray(ds,
                                  start_position=ray_start.copy(),
                                  end_position=mw_center.copy(),
                                  data_filename=ray_filename,
                                  redshift=z_from_distance(r),
                                  fields=['thermal_energy','density'],
                                  lines=line_list,
                                  ftype='Gas')

    return ray


def sample_single_sightline(r, theta, phi, rays_directory=rays_directory):
    """
    Samples ray along the sightline given by the sperical coordinates of
    starting point to mw center as endpoint.
    """

    print('\n NOW SAMPLING r = {}, theta = {}, phi = {} ~~~~~~~~~~~~~~~~~~~'
          ' \n'.format(r, theta, phi))

    ray_filename = rays_directory + \
                   'ray_{:.3f}_{:.3f}_{:.3f}.h5'.format(r, theta, phi)

    if not os.path.exists(ray_filename):
        #pdb.set_trace()
        #print(r)
        make_ray_to_mw((r, theta, phi), ray_filename=ray_filename)

    gc.collect()



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

def sample_one_to_map(args):

    i, number_of_sightlines, r, theta, phi, rays_directory = args

    print('\n RAY NUMBER {:d}/{:d} ~~~~~~~~~~~~~~~~~~~'
          ' \n'.format(i, number_of_sightlines))

    sample_single_sightline(r,theta, phi, rays_directory)



def make_ray_sample_uniform(r_interval, number_of_sightlines, pool):

    theta_interval, phi_interval = sphere_uniform_grid(number_of_sightlines)

    for r in r_interval:

        tasks = [(i, number_of_sightlines, r, theta, phi) for i, (theta, phi) in \
                 enumerate(zip(theta_interval, phi_interval))]
        pool.map(sample_one_to_map, tasks)
    pass





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

def ray_to_subhalo(subhalo_idx, subhalo_rays_directory, ray_end = mw_center):
    """
    Samples ray from subhalo center corresponding to subhalo index (subhalo_idx)
    to mw_center.
    """

    subhalo_position = subhalo_center(subhalo_number = subhalo_idx)

    r, theta, phi = cart_to_sph(subhalo_position - ray_end)

    ray_filename = subhalo_rays_directory + \
                   'ray_sub_{:02d}_{:.3f}_{:.3f}_{:.3f}.h5'.format(
                   subhalo_idx, r, theta, phi)

    ray = trident.make_simple_ray(ds,
                                  start_position=subhalo_position.copy(),
                                  end_position=ray_end.copy(),
                                  data_filename=ray_filename,
                                  fields=['thermal_energy','density'],
                                  redshift=z_from_distance(r),
                                  lines=line_list,
                                  ftype='Gas')
    gc.collect()

    return ray


def sample_subhalos_one_to_map(args):

    subhalo_idx, subhalo_rays_directory, ray_end = args

    print('\n NOW SAMPLING subhalo {:02d} ~~~~~~~~~~~~~~~~~~~'
          ' \n'.format(subhalo_idx))

    ray_to_subhalo(subhalo_idx, subhalo_rays_directory, ray_end)


def sample_subhalos(number_of_subhalos, ray_end = mw_center):
    """
    Generates rays from all subhalos up to number_of_subhalos, except for MW (
    with index = 1).
    """

    for index in range(number_of_subhalos):

        if index != 1:

            print('\n NOW SAMPLING subhalo {:02d} ~~~~~~~~~~~~~~~~~~~'
                  ' \n'.format(index))

            ray_to_subhalo(index, subhalo_rays_directory, ray_end)








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
    pass
    #sample_m31_and_away(close_up_050)
    #pool = Pool(number_of_cores)
    #make_ray_sample_uniform([2000], 10, pool)
    #sample_single_sightline(2000,1.0, 5.6)
