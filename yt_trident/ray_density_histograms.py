# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi as pi
import glob
import matplotlib.pyplot as plt
plt.ion()
import yt
import trident
from tools import ray_mean_density


theta_array = np.linspace(0, pi, 10)
phi_array = np.linspace(0, 2*pi, 10)

#mean_densities = np.zeros(np.len(theta_array)*np.len(phi_array))


def get_mean_densities():

    mean_densities = []

    for handle in glob.glob('./rays_2Mpc_LG/ray_1000*'):

        ray = yt.load(handle)

        mean_densities.append(ray_mean_density(ray, 'C_p1_number_density'))

    return np.array(mean_densities)

def get_single_densities():

    singular_densities = []

    for handle in glob.glob('./rays_2Mpc_LG/ray_1000*'):

        ray = yt.load(handle)

        singular_densities.append(np.asarray(ray.r['gas','C_p1_number_density']))

    return np.array(singular_densities)


mean_densities = get_mean_densities()
singular_densities = get_single_densities()

plt.figure()
plt.hist(mean_densities, bins=20, color = 'red', edgecolor='black', linewidth=1.2)

#plt.figure()
#plt.hist(mean_densities, bins=20, color = 'red', edgecolor='black', linewidth=1.2)
