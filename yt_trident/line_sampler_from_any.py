# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
plt.ion()
import yt
yt.enable_parallelism()
import trident
import gc

from tools import my_field_def, unit_base, subhalo_center, ray_start_from_sph, make_projection


# ~~~~~~~~~~~~~~~~~~~~ SETUP ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

snap_file = '../../2Mpc_LG_convert/snapdir_135/snap_LG_WMAP5_2048_135.0'
snap_num = 135
subfind_path = '../../2Mpc_LG'

rays_directory = './falopa/'


ds = yt.frontends.gadget.GadgetDataset(filename=snap_file, unit_base= unit_base,
    field_spec=my_field_def)

# from Table 1 in Richter, Nuza, et al (2017)
line_list = ['C II', 'C IV', 'Si III', 'Si II']

mw_center = subhalo_center(subfind_path=subfind_path, snap_num=snap_num,
            subhalo_number = 1)

#mw_center = ds.arr(mw_center, 'code_length')


# ~~~~~~~~~~~~~~~~~~~~ ACTIONS ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_ray_from_any(ray_end, spherical_coords_start, ray_filename):
    """
    Creates ray with center of MW as starting point, and end point passed by
    its spherical coordinates from MW center.

    Spherical coordinates of end point to be passed as iterable.
    """

    ray_start = ray_start_from_sph(ray_end, spherical_coords_start)

    # for some reason, make_simple_ray overwrites start_position and end_position
    # actually are passed as pointers and changes them to cgs; this can be prevented
    # by passing them as ray_start.copy()
    ray = trident.make_simple_ray(ds,
                                  start_position=ray_start.copy(),
                                  end_position=ray_end.copy(),
                                  data_filename=ray_filename,
                                  lines=line_list,
                                  ftype='Gas')

    return ray



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



# ~~~~~~~~~~~~~~ RAY SAMPLING M31 ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


distances = np.linspace(10, 700, 100)
distances_detail = np.linspace(1, 10, 50)
distances_more_detail = np.linspace(0, 1, 50)
#close_up_050 = np.linspace(0.36, 0.51, 50)

all_distances = np.concatenate((distances_more_detail, distances_detail, distances))

sample_m31_and_away(all_distances[1:])
