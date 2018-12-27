# -*- coding: utf-8 -*-

import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
plt.ion()
import yt
yt.toggle_interactivity()
yt.enable_parallelism()
from tools import my_field_def, unit_base, subhalo_center, ray_start_from_sph, make_projection, make_slice, plot_ray_in_projection, plot_ray_in_slice

snap_file = '../../2Mpc_LG_convert/snapdir_135/snap_LG_WMAP5_2048_135.0'
snap_num = 135
subfind_path = '../../2Mpc_LG'


ds = yt.frontends.gadget.GadgetDataset(filename=snap_file, unit_base= unit_base,
    field_spec=my_field_def)

mw_center = subhalo_center(subfind_path=subfind_path, snap_num=snap_num,
            subhalo_number = 1)

px = make_projection(ds, mw_center, 2000, 'x')
px.show()
