# -*- coding: utf-8 -*-
"""
Script to test animation creating in matplotlib

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels


number_of_snapshots = 200   # actually includes snap_000

file = '../outputs/snap_{:03d}'

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax_1, ax_2, ax_3 = plt.subplots(1, 3)

def scatter(ax, positions):

	x, y, z = np.split(positions, 3, axis = 1)

	ax_1.scatter(x, y)
	ax_2.scatter(x, z)
	ax_3.scatter(y, z)

	return ax_1, ax_2, ax_3

