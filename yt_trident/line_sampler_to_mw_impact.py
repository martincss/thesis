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

rays_directory = './rays_2Mpc_LG_to_mw_impact/'

ds = yt.frontends.gadget.GadgetDataset(filename=snap_file, unit_base= unit_base,
    field_spec=my_field_def)
ds.add_field(("gas", "pressure"), function=_pressure, units="dyne/cm**2")

# from Table 1 in Richter, Nuza, et al (2017)
line_list = ['C II', 'C IV', 'Si III', 'Si II', 'Si IV', 'H']

mw_center = subhalo_center(subfind_path=subfind_path, snap_num=snap_num,
            subhalo_number = 1)


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
