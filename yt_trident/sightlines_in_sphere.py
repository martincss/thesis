# -*- coding: utf-8 -*-
from numpy import pi, cos, sin, array
from numpy.linalg import norm
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from tools import sphere_uniform_grid, sightline_in_subsample, \
                  select_m31_rays, select_polar_rays, get_m31_center_2Mpc_LG, \
                  cart_to_sph, get_disk_normal_vector_mw_2Mpc_LG, \
                  get_sun_position_2Mpc_LG, absorber_mean_vlos


def plot_polar_and_m31():

    ax3.scatter(*polar_vect, c = 'red', s = 200, marker = 'x')

    ax3.scatter(*-polar_vect, c = 'red', s = 200, marker = 'x')

    ax3.scatter(*unit_to_m31, c = 'blue', s = 200, marker = '*')



def get_colors(selected,
               absorbers_directory='absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter/'):

    v_los_array = []

    for theta, phi in selected:

        absorber_filename = absorbers_directory +\
                            'abs_2000.000_{:.3f}_{:.3f}.txt'.format(theta, phi)

        v_los_array.append(absorber_mean_vlos(absorber_filename))

    return array(v_los_array)


def plot_in_sphere(selection):

    thetas, phis = array(selection).T

    x, y, z = cos(phis) * sin(thetas), sin(phis) * sin(thetas), cos(thetas)

    pts = ax3.scatter(x,y,z, c= get_colors(selection), cmap = 'coolwarm')
    fig.colorbar(pts, shrink=0.5, aspect=5)


if __name__ == '__main__':

    m31_center = get_m31_center_2Mpc_LG()
    unit_to_m31 = m31_center - get_sun_position_2Mpc_LG()
    unit_to_m31 /= norm(unit_to_m31)

    polar_vect = get_disk_normal_vector_mw_2Mpc_LG()


    thetas, phis = sphere_uniform_grid(500)
    # x, y, z = cos(thetas) * sin(phis), sin(thetas) * sin(phis), cos(phis);


    selected = [(theta, phi) for theta, phi in zip(thetas, phis) if \
                sightline_in_subsample(theta, phi, amplitude_polar=1)]

    # selected_polar = [(theta, phi) for theta, phi in zip(thetas, phis) if \
    #                    select_polar_rays(theta, phi)]
    #
    # selected_m31 = [(theta, phi) for theta, phi in zip(thetas, phis) if \
    #                   select_m31_rays(theta, phi)]
    selected.pop(selected.index((0.8579363494812754, 6.031481072577428)))

    fig = plt.figure()
    ax3 = Axes3D(fig)

    plot_polar_and_m31()
    #plot_in_sphere(selected_m31, 'violet')
    #plot_in_sphere(selected_polar, 'red')
    plot_in_sphere(selected)
