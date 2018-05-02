# -*- coding: utf-8 -*-
"""
Program to plot time evolution of galaxy projections
in snapshots. 
Given a set of Gadget snapshots, plots projections
by three coordinate planes and saves figures as
ordered frames.

"""

import numpy as np
import matplotlib.pyplot as plt

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels


def multiplot(positions, colour, point_size = 0.001, xlims = (-1500, 1500), 
	          ylims = (-1500, 1500), zlims = (-1500, 1500)):
	"""
	Given the array of gas or stars positions, creates three 
	subplots, each containing the XY, XZ, YZ scatters.
	Marker color is given by colour argument.
	Scatter marker size is given by point_size.
	Axis limits may be specified by length two tuples.

	"""

	x, y, z = np.split(positions, 3, axis = 1)

	plt.subplot(1,3,1)
	plt.scatter(x, y, color = colour, s = point_size)

	plt.grid(True)
	plt.xlabel('$x$', fontsize = 15)
	plt.ylabel('$y$', fontsize = 15)
	plt.xlim(xlims)
	plt.ylim(ylims)


	plt.subplot(1,3,2)
	plt.scatter(x, z, color = colour, s = point_size)

	plt.grid(True)
	plt.xlabel('$x$', fontsize = 15)
	plt.ylabel('$z$', fontsize = 15)
	plt.xlim(xlims)
	plt.ylim(ylims)


	plt.subplot(1,3,3)
	plt.scatter(y, z, color = colour, s = point_size)

	plt.grid(True)
	plt.xlabel('$y$', fontsize = 15)
	plt.ylabel('$z$', fontsize = 15)
	plt.xlim(xlims)
	plt.ylim(ylims)


	#plt.tight_layout()


number_of_snapshots = 200   # actually includes snap_000

file = '/home/martin/Documents/Tesis/outputs/snap_{:03d}'


plt.figure(figsize=(13.0, 6.0)) # in inches!

for i in range(number_of_snapshots + 1):

	snap = load_snapshot(file.format(i), label_table = cecilia_labels)
	gas_pos = snap['POS '][0]
	stars_pos = snap['POS '][4]

	# clears figure, plots, and saves ordered frame as .png
	plt.clf()
	
	multiplot(gas_pos, 'b', xlims = (-500, 500), ylims = (-500, 500),
	          zlims = (-500, 500))

	multiplot(stars_pos, 'r', xlims = (-500, 500), ylims = (-500, 500),
	          zlims = (-500, 500))
	
	plt.suptitle('Galaxy projections at snapshot {:03d}'.format(i), fontsize = 20)
	plt.subplots_adjust(top=0.9, bottom=0.11, left=0.125, right=0.9, hspace=0.2,
                        wspace=0.4)
	#plt.tight_layout()

	# to undo figure saving and preview time evolution, comment following
	# line and uncomment pause

	#plt.savefig('../frames2/frame_{:03d}.png'.format(i), dpi = 300, bbox_inches = 'tight')

	plt.pause(0.001)





