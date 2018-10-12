# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
plt.ion()
import yt
import trident

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

#    ray_end = ray_end_from_sph(mw_center, spherical_coords)

    print('el ray_end antes es ', ray_end)
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

    for r in r_array:

        for theta in theta_array:

            for phi in phi_array:

                print('\n NOW SAMPLING r = {}, theta = {}, phi = {} ~~~~~~~~~~~~~~~~~~~ \n'.format(r, theta, phi))

                ray_filename = './rays_2Mpc_LG/ray_{:.0f}_{:.1f}_{:.1f}.h5'.format(r, theta, phi)
                ray_from_mw = make_ray_from_mw((r, theta, phi), ray_filename=ray_filename)


# ~~~~~~~~~~~~~~~~ RAY SAMPLING ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

r_array = np.array([1000])
theta_array = np.linspace(0, pi, 10)
phi_array = np.linspace(0, 2*pi, 10)

make_ray_sample(r_array, theta_array, phi_array)


#ray_test = make_ray_from_mw((1000, 0, 0), './rays_2Mpc_LG/ray_test.h5')
#ray_from_mw = make_ray_from_mw((1000, 0, 0), './rays_2Mpc_LG/ray_1000_0_0.h5')
