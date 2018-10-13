# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi as pi
import glob
import matplotlib.pyplot as plt
plt.ion()
import yt
from tools import ray_mean_density
import pandas as pd

theta_array = np.linspace(0, pi, 10)
phi_array = np.linspace(0, 2*pi, 10)

#mean_densities = np.zeros(np.len(theta_array)*np.len(phi_array))


def get_mean_densities():
    """
    Returns mean density of each ray in an array.
    """

    mean_densities = []

    for handle in glob.glob('./rays_2Mpc_LG/ray_1000_*_*'):

        ray = yt.load(handle)

        mean_densities.append(ray_mean_density(ray, 'C_p1_number_density'))

    return np.array(mean_densities)

def get_single_densities():
    """
    Returns array of all densities for each cell in each ray.
    """

    singular_densities = []

    for handle in glob.glob('./rays_2Mpc_LG/ray_1000*'):

        ray = yt.load(handle)

        singular_densities.append(np.asarray(ray.r['gas','C_p1_number_density'])[:])

    return np.hstack(singular_densities)

mean_densities = get_mean_densities()
singular_densities = get_single_densities()



plt.figure()
plt.hist(mean_densities, bins=20, color = 'red', edgecolor='black', linewidth=1.2)
plt.xlabel('C_p1_number_density (cm^-3)', fontsize = 15)
plt.ylabel('Counts', fontsize = 15)
plt.title('Mean line number densities', fontsize = 15)


#plt.figure()
#plt.hist(singular_densities, bins=200, color = 'red', edgecolor='black', linewidth=1.2, log=True)
