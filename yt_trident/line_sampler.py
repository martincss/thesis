import numpy as np
import matplotlib.pyplot as plt
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

    ray_end = ray_end_from_sph(mw_center, spherical_coords)

    ray = trident.make_simple_ray(ds,
                                  start_position=mw_center,
                                  end_position=ray_end,
#                                  trajectory=spherical_coords,
                                  data_filename=ray_filename,
                                  lines=line_list,
                                  ftype='Gas')

    return ray

def make_ray_sample(r_interval, theta_interval, phi_interval):

    for r, theta, phi in zip(r_interval, theta_interval, phi_interval):

        ray_filename = './rays_2Mpc_LG/ray_{:.0f}_{:.0f}_{:.0f}.h5'.format(r, theta, phi)

        make_ray_from_mw(*(r, theta, phi), ray_filename=ray_filename=)







#ray_from_mw = make_ray_from_mw((1000, 0, 0), './rays_2Mpc_LG/ray_1000_0_0.h5')