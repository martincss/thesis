# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
plt.ion()
import yt
yt.enable_parallelism()
import trident
import gc

from tools import my_field_def, unit_base, subhalo_center, ray_start_from_sph, \
    make_projection, sphere_uniform_grid, _pressure


# ~~~~~~~~~~~~~~~~~~~~ SETUP ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

snap_file = '../../2Mpc_LG_convert/snapdir_135/snap_LG_WMAP5_2048_135.0'
snap_num = 135
subfind_path = '../../2Mpc_LG'

rays_directory = './rays_2Mpc_LG_to_m31_210/'
subhalo_rays_directory = './rays_2Mpc_LG_from_subhalos/'

ds = yt.frontends.gadget.GadgetDataset(filename=snap_file, unit_base= unit_base,
    field_spec=my_field_def)
ds.add_field(("gas", "pressure"), function=_pressure, units="dyne/cm**2")

# from Table 1 in Richter, Nuza, et al (2017)
line_list = ['C II', 'C IV', 'Si III', 'Si II', 'Si IV', 'H']

m31_center = subhalo_center(subfind_path=subfind_path, snap_num=snap_num,
            subhalo_number = 0)

#mw_center = ds.arr(mw_center, 'code_length')


# ~~~~~~~~~~~~~~~~~~~~ ACTIONS ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_ray_to_m31(spherical_coords, ray_filename):
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
                                  end_position=m31_center.copy(),
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
    make_ray_to_m31((r, theta, phi), ray_filename=ray_filename)



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



# ~~~~~~~~~~~~~~~~ RAY SAMPLING ~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    #make_ray_sample_uniform([210], 500)
    pass
