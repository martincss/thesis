#!/usr/bin/env python
from iccpy.gadget import load_snapshot
from iccpy_labels_new_metal import cecilia_labels
from numpy import sqrt
from numpy.linalg import norm
import matplotlib.pyplot as plt
plt.ion()

snapdir_lo_res = '../../copiar_cecilia_nuevo'
snap_low = load_snapshot(directory = snapdir_lo_res, snapnum = 36,
                         label_table = cecilia_labels)

snapdir_hi_res = '../../copiar_cecilia_nuevo/new_outputs_martin_wrt_MWcenter'
snap_hi = load_snapshot(directory = snapdir_hi_res, snapnum = 1000,
                        label_table = cecilia_labels)

vels_low = norm(snap_low['VEL '][0], axis=1)
vels_hi = norm(snap_hi['VEL '][0], axis=1)



plt.figure()
plt.hist(vels_hi, bins=500, edgecolor='black', color = 'crimson',
         label ='high resolution', log=True, density=True)

plt.hist(vels_low, bins=500,edgecolor='black',color = 'magenta',
         label ='low resolution', alpha = 0.5, log=True, density=True)

plt.xlabel('$|V| [km\\slash s]$', fontsize=15)
plt.ylabel('frecuencies', fontsize=15)
plt.xlim(0, 4000)
plt.title('Particle velocities', fontsize=15)
plt.grid()
plt.legend()
