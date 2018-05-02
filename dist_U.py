# -*- coding: utf-8 -*-
"""
Program to plot 
"""


import numpy as np
import matplotlib.pyplot as plt

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels

plt.ion()

number_of_snapshots = 200   # actually includes snap_000

file = '/home/martin/Documents/Tesis/outputs/snap_{:03d}'

snap = load_snapshot(file.format(200), label_table = cecilia_labels)

gas_pos = snap['POS '][0]

gas_x = gas_pos[:,0]
gas_y = gas_pos[:,1]
gas_z = gas_pos[:,2]

gas_r = np.sqrt(gas_x**2 + gas_y**2 + gas_z**2)

gas_U = snap['U   '][0]

gas_rho = snap['RHO '][0]

plt.figure()
ax = plt.gca()

plt.scatter(gas_r, gas_U, c = gas_rho, s = 0.01)
plt.xlabel('$r$', fontsize = 15)
plt.ylabel('$\\mathcal{U}$', fontsize = 15)
#ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(True)



plt.figure()
ax = plt.gca()

plt.scatter(gas_x, gas_y, c = gas_U, s = 0.01, cmap = 'jet')
plt.xlim([-500, 500])
plt.ylim([-500, 500])
plt.xlabel('$r$', fontsize = 15)
plt.ylabel('$\\mathcal{U}$', fontsize = 15)
plt.grid(True)



