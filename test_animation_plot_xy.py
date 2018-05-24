# -*- coding: utf-8 -*-
"""
Script to test animation creating in matplotlib

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels


number_of_snapshots = 100   # actually includes snap_000

file = '../outputs/snap_{:03d}'

lim = 1500


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize = (13.0, 6.0))
ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim))
points, = ax.plot([], [], 'bo', ms = 0.01)

# Function called to create the base frame upon which the animation takes place.
def init():

	
	ax.set_xlim(-lim, lim)
	ax.set_ylim(-lim, lim)
	ax.grid(True)

	points.set_data([], [])

	return points,

# Function to be called by animation
def animate(i):

	snap = load_snapshot(file.format(i), label_table = cecilia_labels)
	gas_pos = snap['POS '][0]
	#stars_pos = snap['POS '][4]

	x, y, z = np.split(gas_pos, 3, axis = 1)

	points.set_data(x, y)
	
	print(i)

	return points,

# call the animator.  blit=True means only re-draw the parts that have changed
anim = animation.FuncAnimation(fig, animate, init_func = init, frames = number_of_snapshots, interval = 200)#, blit = True)

#save the animation using ffmpeg
dpi = 30
writer = animation.writers['ffmpeg'](fps=30)
#anim.save('demo_alt.mp4',writer=writer,dpi=dpi)

anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'], dpi = dpi)

plt.show()



