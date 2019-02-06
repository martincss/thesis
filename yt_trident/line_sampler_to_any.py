# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi as pi
import os
from trident import make_simple_ray
import gc
import pdb
from tools import get_2Mpc_LG_dataset, get_mw_center_2Mpc_LG, \
                  ray_start_from_sph, sphere_uniform_grid, z_from_distance, \
                  fuzzy_samples_from_subhalo, cart_to_sph

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

def make_ray_from_any(ray_end, spherical_coords_start, ray_filename):
    """
    Creates ray with arbitrary ending point (where the observer is),
    and starting point passed by its spherical coordinates from ending point.

    Parameters
    ----------
    ray_end: iterable
        iterable of three elements containing cartesian coordinates of ending
        point (i.e. location of observer)

    spherical_coords_start: iterable
        iterable of three elements containing spherical coordinates of starting
        point referenced from ray_end

    ray_filename: string
        full path to filename where to store ray dataset

    Returns
    -------
    ray: LightRay object
        created ray from ray_start to ray_end, sampling ions from line_list and
        storing 'density' and 'thermal_energy' additional fields.
    """

    ray_start = ray_start_from_sph(ray_end, spherical_coords_start)
    r, _, _ = spherical_coords_start

    # for some reason, make_simple_ray overwrites start_position and end_position
    # actually are passed as pointers and changes them to cgs; this can be prevented
    # by passing them as ray_start.copy()

    # starting point redshift calculated as to assure z~0 at ending point
    ray = make_simple_ray(ds,
                              start_position=ray_start.copy(),
                              end_position=ray_end.copy(),
                              data_filename=ray_filename,
                              redshift=z_from_distance(r),
                              fields=['thermal_energy','density'],
                              lines=line_list,
                              ftype='Gas')

    return ray


def sample_one_to_map_from_any(args):

    r, theta, phi, ray_end, rays_directory = args

    print('\n NOW SAMPLING r = {}, theta = {}, phi = {} ~~~~~~~~~~~~~~~~~~~ '
          '\n'.format(r, theta, phi))

    ray_filename = rays_directory + \
                   'ray_{:.3f}_{:.2f}_{:.2f}.h5'.format(r, theta, phi)

    make_ray_from_any(ray_end, (r, theta, phi), ray_filename=ray_filename)

    gc.collect()



def make_ray_sample(r_interval, theta_interval, phi_interval, ray_start):

    for r in r_interval:

        for theta in theta_interval:

            for phi in phi_interval:

                print('\n NOW SAMPLING r = {}, theta = {}, phi = {} ~~~~~~~~~~~~~~~~~~~ \n'.format(r, theta, phi))

                ray_filename = rays_directory + 'ray_{:.3f}_{:.2f}_{:.2f}.h5'.format(r, theta, phi)
                make_ray_from_any(ray_start,(r, theta, phi), ray_filename=ray_filename)

                gc.collect()


def sample_m31_and_away(r_interval):
    """
    Samples rays on fixed directions to m31 and away from m31, varying distance
    to endpoints.
    Rays starting points are already 300kpccm away from the center of mw, on
    each direction.

    """

    theta_m31 = 2*pi/9
    phi_m31 = 6*(2*pi)/9
    ray_end_to_m31 = ray_start_from_sph(mw_center, (300, theta_m31, phi_m31))

    make_ray_sample(r_interval, [theta_m31], [phi_m31], ray_start_to_m31)

    theta_away = 3*pi/9
    phi_away = 2*(2*pi)/9
    ray_end_away = ray_start_from_sph(mw_center, (300, theta_m31, phi_m31))

    make_ray_sample(r_interval, [theta_away], [phi_away], ray_start_away)




def rays_fuzzy_from_subhalo(subhalo_idx, ray_end, subhalo_rays_directory,
                            number_of_rays):
    """
    Sample rays from subhalo center corresponding to subhalo index (subhalo_idx)
    plus a random displacement (with radius smaller than subhalo virual radius)
    to ray_end.


    """

    samples = fuzzy_samples_from_subhalo(subhalo_idx, number_of_rays)

    for ray_start in samples:

        # identify ray_start by spherical coordinates as seen from ray_end
        r, theta, phi = cart_to_sph(ray_start - ray_end)

        ray_filename = subhalo_rays_directory + \
                       'ray_fsub_{:02d}_{:.3f}_{:.3f}_{:.3f}.h5'.format(
                       subhalo_idx, r, theta, phi)

        ray = make_simple_ray(ds,
                                  start_position=ray_start.copy(),
                                  end_position=ray_end.copy(),
                                  data_filename=ray_filename,
                                  redshift=z_from_distance(r),
                                  fields=['thermal_energy','density'],
                                  lines=line_list,
                                  ftype='Gas')
        gc.collect()


def subhalo_fuzzy_sampling_one_to_map(args):

    subhalo_idx, ray_end, subhalo_rays_directory, number_of_rays = args

    print('\nNOW SAMPLING RAYS FROM SUBHALO {} ~~~~~~~~~\n'.format(subhalo_idx))

    rays_fuzzy_from_subhalo(subhalo_idx, ray_end, subhalo_rays_directory,
                                number_of_rays)
