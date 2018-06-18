# -*- coding: utf-8 -*-
"""
Test script to try out subfind and plotting
for each subhalo

"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match


subfind_path = '../../2Mpc_LG'
snap_num = 135
snap_dir = '../../2Mpc_LG'


cat = SubfindCatalogue(subfind_path, snap_num)
# Directory and snapnum are passed instead of filename, to open snapshot in multiple parts
snap = load_snapshot(directory = snap_dir.format(snap_num), snapnum = snap_num, label_table = cecilia_labels)

gas_ids = snap['ID  '][0]
stars_ids = snap['ID  '][4]
#gas_groups = cat.

fig, ax = plt.subplots()

for subhalo in cat.subhalo[:5]:

	sub_ids = subhalo.ids
	indexes = match(sub_ids, gas_ids)
	indexes = indexes[indexes != -1]

	subhalo_positions = snap['POS '][0][indexes]
	x, y, z = np.split(subhalo_positions, 3, axis = 1)

	ax.scatter(x,y, s = 0.01)

fig.show()




