# -*- coding: utf-8 -*-
"""
Test script to try out subfind and plotting for each subhalo.
Isolates and plots in different colors particles for each subhalo.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

# paths to the parent directory of the folders containing snapshots and subfind
# outputs must be supplied
subfind_path = '../../2Mpc_LG'
snap_num = 135
snap_dir = '../../2Mpc_LG'

# we create the Subfind object for that particular snapshot
cat = SubfindCatalogue(subfind_path, snap_num)
# Directory and snapnum are passed instead of filename, to open snapshot in multiple parts
snap = load_snapshot(directory = snap_dir.format(snap_num), snapnum = snap_num,
	label_table = cecilia_labels)

gas_ids = snap['ID  '][0]
stars_ids = snap['ID  '][4]


def plotxy_subhalo(subhalo, num_species):
	"""
	Given a subhalo object, and species number (0 for gas, 4 for stars, etc.),
	plots the XY projection of the particles onto the axis object.
	"""

	# we first find the indices of the IDs array in which the subhalo IDs are
	# located. We then filter out the -1s in the indexes array, since these are
	# the subhalo IDs unmatched to the snap IDs for the particular species.
	indexes = match(subhalo.ids, snap['ID  '][num_species])
	indexes = indexes[indexes != -1]

	subhalo_positions = snap['POS '][num_species][indexes]
	x, y, z = np.split(subhalo_positions, 3, axis = 1)

	ax.scatter(x,y, s = 0.01)


fig, ax = plt.subplots()

# Define number of subhaloes to plot (if -1, plot all of them)
number_of_subhaloes = 2

for subhalo in cat.subhalo[:number_of_subhaloes]:

	plotxy_subhalo(subhalo, 4)

fig.show()
