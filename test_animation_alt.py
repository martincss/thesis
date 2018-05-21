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
fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3)
points_1, = ax_1.plot([], [], 'bo')
points_2, = ax_2.plot([], [], 'bo')
points_3, = ax_3.plot([], [], 'bo')

points = [points_1, points_2, points_3]

# Function called to create the base frame upon which the animation takes place.
def init():

	points[0].set_data([], [])
	#points[1].set_offsets()
	#points[2].set_offsets()	

	return points,

# Function to be called by animation
def animate(i):

	snap = load_snapshot(file.format(i), label_table = cecilia_labels)
	gas_pos = snap['POS '][0]
	#stars_pos = snap['POS '][4]

	x, y, z = np.split(gas_pos, 3, axis = 1)

	points[0].set_data(x, y)	
	#points[1].set_offsets([x,z])
	#points[2].set_offsets([y,z])
	
	return points,

# call the animator.  blit=True means only re-draw the parts that have changed
anim = animation.FuncAnimation(fig, animate, init_func = init, frames = number_of_snapshots, interval = 20)#, blit = True)

#save the animation using ffmpeg
dpi = 30
writer = animation.writers['ffmpeg'](fps=30)
anim.save('demo_alt.mp4',writer=writer,dpi=dpi)




