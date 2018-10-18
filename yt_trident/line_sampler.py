# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
plt.ion()
import yt
import trident
import gc

from tools import my_field_def, unit_base, subhalo_center, ray_end_from_sph, make_projection


# ~~~~~~~~~~~~~~~~~~~~ SETUP ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

snap_file = '../../2Mpc_LG_convert/snapdir_135/snap_LG_WMAP5_2048_135.0'
snap_num = 135
subfind_path = '../../2Mpc_LG'


ds = yt.frontends.gadget.GadgetDataset(filename=snap_file, unit_base= unit_base,
    field_spec=my_field_def)

# from Table 1 in Richter, Nuza, et al (2017)
line_list = ['C II', 'C IV', 'Si III', 'Si II']

mw_center = subhalo_center(subfind_path=subfind_path, snap_num=snap_num,
            subhalo_number = 1)

#mw_center = ds.arr(mw_center, 'code_length')


# ~~~~~~~~~~~~~~~~~~~~ ACTIONS ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_ray_from_mw(spherical_coords, ray_filename):

    ray_end = ray_end_from_sph(mw_center, spherical_coords)

#    print('el ray_end antes es ', ray_end)
    # for some reason, make_simple_ray overwrites start_position and end_position
    # actually are passed as pointers and changes them to cgs; this can be prevented
    # by passing them as ray_start.copy()
    ray = trident.make_simple_ray(ds,
                                  start_position=mw_center.copy(),
                                  end_position=ray_end.copy(),
#                                  trajectory=spherical_coords,
                                  data_filename=ray_filename,
                                  lines=line_list,
                                  ftype='Gas')

#    print('los mw_center y ray_end después son ', mw_center, ray_end)


    return ray

def make_ray_sample(r_interval, theta_interval, phi_interval):

    for r in r_interval:

        for theta in theta_interval:

            for phi in phi_interval:

                print('\n NOW SAMPLING r = {}, theta = {}, phi = {} ~~~~~~~~~~~~~~~~~~~ \n'.format(r, theta, phi))

                ray_filename = './rays_2Mpc_LG/ray_{:.0f}_{:.2f}_{:.2f}.h5'.format(r, theta, phi)
                make_ray_from_mw((r, theta, phi), ray_filename=ray_filename)

                gc.collect()


def sample_m31_and_away(r_interval):

    theta_m31 = pi/9
    phi_m31 = 6*(2*pi)/9

    make_ray_sample(r_interval, theta_m31, phi_m31)

    theta_away = pi/2 - theta_m31
    phi_away = phi_m31

    make_ray_sample(r_interval, theta_away, phi_away)


# ~~~~~~~~~~~~~~~~ RAY SAMPLING ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

r_array = np.array([1000])
theta_array = np.linspace(0, pi, 10)
phi_array = np.linspace(0, 2*pi, 10)

#make_ray_sample(r_array, theta_array, phi_array)

# ~~~~~~~~~~~~~~ RAY SAMPLING M31 ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


distances = np.linspace(0, 1000, 100)

sample_m31_and_away(distances)
