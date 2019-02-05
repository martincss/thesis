# -*- coding: utf-8 -*-
from numpy import pi, cos, sin, array
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from tools import sphere_uniform_grid, sightline_in_subsample, \
                  select_m31_rays, select_polar_rays, get_m31_center_2Mpc_LG, \
                  cart_to_sph, get_disk_normal_vector_mw_2Mpc_LG, \
                  get_sun_position_2Mpc_LG

m31_center = get_m31_center_2Mpc_LG()
_, theta_m31, phi_m31 = cart_to_sph(m31_center - get_sun_position_2Mpc_LG())

polar_vect = get_disk_normal_vector_mw_2Mpc_LG()
_, polar_theta, polar_phi = cart_to_sph(polar_vect)

antipolar_theta = pi - polar_theta
antipolar_phi = (polar_phi + pi) % 2*pi


thetas, phis = sphere_uniform_grid(500)
# x, y, z = cos(thetas) * sin(phis), sin(thetas) * sin(phis), cos(phis);


selected = [(theta, phi) for theta, phi in zip(thetas, phis) if \
            sightline_in_subsample(theta, phi)]

# selected_polar = [(theta, phi) for theta, phi in zip(thetas, phis) if \
#                   select_polar_rays(theta, phi)]
#
# selected_m31 = [(theta, phi) for theta, phi in zip(thetas, phis) if \
#                   select_m31_rays(theta, phi)]

fig = plt.figure()
ax3 = Axes3D(fig)


def plot_polar_and_m31():

    ax3.scatter(cos(polar_phi) * sin(polar_theta), sin(polar_phi) * sin(polar_theta),
    cos(polar_theta), c = 'red', s = 200, marker = 'x')

    ax3.scatter(cos(antipolar_phi) * sin(antipolar_theta), sin(antipolar_phi) * sin(antipolar_theta),
    cos(antipolar_theta), c = 'red', s = 200, marker = 'x')

    ax3.scatter(cos(phi_m31) * sin(theta_m31), sin(phi_m31) * sin(theta_m31),
    cos(theta_m31), c = 'blue', s = 200, marker = '*')



def plot_in_sphere(selection, color):

    thetas, phis = array(selection).T

    x, y, z = cos(phis) * sin(thetas), sin(phis) * sin(thetas), cos(thetas)

    ax3.scatter(x,y,z, c= color)


plot_polar_and_m31()
plot_in_sphere(selected, 'violet')
