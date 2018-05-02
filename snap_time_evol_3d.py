# -*- coding: utf-8 -*-
"""
Program to plot time evolution of galaxy in snapshots. 
Given a set of Gadget snapshots, plots 3D scatter of
galaxy gas and star particles, and saves figures as
ordered frames.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels



number_of_snapshots = 200   # actually includes snap_000

file = '/home/martin/Documents/Tesis/outputs/snap_{:03d}'


fig = plt.figure(figsize=(13.0, 6.0)) # in inches!
ax3 = Axes3D(fig)

for i in range(153, number_of_snapshots + 1):

	snap = load_snapshot(file.format(i), label_table = cecilia_labels)
	gas_pos = snap['POS '][0]
	stars_pos = snap['POS '][4]

	# clears figure, plots, and saves ordered frame as .png

	ax3.set_xlim3d(-300, 300)
	ax3.set_ylim3d(-300, 300)
	ax3.set_zlim3d(-300, 300)

	ax3.scatter(gas_pos[:,0], gas_pos[:,1], gas_pos[:,2], color = 'b', s = 0.001)
	ax3.scatter(stars_pos[:,0], stars_pos[:,1], stars_pos[:,2], color = 'r', s = 0.001)

	plt.title('Galaxy at snapshot {:03d}'.format(i), fontsize = 20)
	
	# to undo figure saving and preview time evolution, comment following
	# line and uncomment pause

	#plt.savefig('frames3d/frame_{:03d}.png'.format(i), dpi = 100, bbox_inches = 'tight')
	
	plt.pause(0.001)


	ax3.clear()




