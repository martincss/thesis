import os
import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
plt.ion()
import yt
import trident

rays_directory_from_mw = './rays_2Mpc_LG_from_mw/'
absorbers_directory_from_mw = './absorbers_2Mpc_LG_from_mw/'

line_list = ['C II', 'C IV', 'Si III', 'Si II']

ray_to_m31_name = 'ray_990_0.70_4.19.h5'
ray_away_name = 'ray_990_1.05_1.40.h5'

species = 'Si II 1304'

###########################################

def make_absorbers_file(ray, ray_filename, absorbers_directory):

    absorbers_filename = 'abs_' + ray_filename[4:-3]

    sg = trident.SpectrumGenerator(lambda_min = 1150, lambda_max = 1600,
        dlambda = 0.01)

    if not os.path.exists(absorbers_directory + absorbers_filename + '.txt'):

        sg.make_spectrum(ray, lines=line_list,
                        output_absorbers_file = absorbers_directory + absorbers_filename + '.txt',
                        store_observables = True)
    else:

        sg.make_spectrum(ray, lines=line_list, store_observables = True)

    return sg

#####################################################
ray_to_m31 = yt.load(rays_directory_from_mw + ray_to_m31_name)
ray_away = yt.load(rays_directory_from_mw + ray_away_name)

sg_m31 = make_absorbers_file(ray_to_m31, ray_to_m31_name, absorbers_directory_from_mw)
sg_away = make_absorbers_file(ray_away, ray_away_name, absorbers_directory_from_mw)

z_m31 = ray_to_m31.r['redshift']
z_away = ray_away.r['redshift']

col_dens_m31 = sg_m31.line_observables_dict[species]['column_density']
col_dens_away = sg_away.line_observables_dict[species]['column_density']

###################################################

plt.figure()
plt.semilogy(-z_m31, col_dens_m31, color = 'blue', label = 'ray to M31')
plt.semilogy(-z_away, col_dens_away, color = 'red', label = 'ray away from M31')
plt.xlabel('$Redshift $', fontsize=25)
plt.ylabel('$N\;\; [cm^{-2}]$', fontsize = 25)
plt.title('Column density by redshift', fontsize = 25)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid(True)
plt.legend()


###################################################

plt.figure()

log_cols_m31 = np.log10(np.asarray(col_dens_m31))[col_dens_m31 != 0]
log_cols_away = np.log10(np.asarray(col_dens_away))[col_dens_away != 0]

plt.hist(np.asarray(log_cols_away), bins=13, color = 'red', edgecolor='black',
        linewidth=1.2, label = 'away from M31', alpha = 0.5)
plt.hist(np.asarray(log_cols_m31), bins=20, color = 'blue', edgecolor='black',
        linewidth=1.2, label = 'to M31', alpha = 0.5)
plt.grid(True)
plt.xlabel('log N [cm^-2]')
plt.legend()
