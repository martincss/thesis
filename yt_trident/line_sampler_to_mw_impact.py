# -*- coding: utf-8 -*-
import numpy as np
import yt
yt.enable_parallelism()
import trident
import gc

from tools import get_2Mpc_LG_dataset, get_mw_center_2Mpc_LG, \
                  ray_start_from_sph, HUBBLE_2Mpc_LG


# ~~~~~~~~~~~~~~~~~~~~ SETUP ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rays_directory = './rays_2Mpc_LG_to_mw_impact/'

ds = get_2Mpc_LG_dataset()

# from Table 1 in Richter, Nuza, et al (2017)
line_list = ['C III', 'C IV', 'Si III', 'Si II', 'O VI']

mw_center = get_mw_center_2Mpc_LG()


def make_ray_to_mw_with_bz(spherical_coords, impact_param, ray_filename):
    """
    Creates ray from the point specified by spherical_coords (from mw center),
    directed to mw center (where the obserber is), with impact parameter b (thus
    shifting starting and ending points) in the z axis.

    Parameters
    ----------
    spherical_coords: iterable
        iterable containing r (in kpc/h), theta, phi as defined from mw center
    impact_param: float
        value of the impact parameter b in which to parallel offset the ray in
        the z direction (in kpc/h)
    ray_filename: string
        full path to filename in which to save resulting lightray

    Returns
    -------
    ray: LightRay object
        lightray created by direction and impact parameter specified
    """

    ray_start = np.asarray(ray_start_from_sph(mw_center, spherical_coords)) + \
                np.array([0, 0, impact_param])

    ray_end = mw_center + np.array([0, 0, impact_param])

    # for some reason, make_simple_ray overwrites start_position & end_position
    # actually are passed as pointers and changes them to cgs; this can be
    # prevented by passing them as ray_start.copy()
    ray = trident.make_simple_ray(ds,
                                  start_position=ray_start.copy(),
                                  end_position=ray_end.copy(),
                                  data_filename=ray_filename,
                                  fields=['thermal_energy','density'],
                                  lines=line_list,
                                  ftype='Gas')

    return ray

def sample_ray_from_away(distance, impact_param):
    """
    Creates ray from the "away" direction to mw center, with impact parameter b
    parallel shifted in the z axis

    Parameters
    ----------
    distance: float
        distance from mw center to ray start in the "away" direction (in kpc/h)
    impact_param: float
        value of the impact parameter b in which to parallel offset the ray in
        the z direction (in kpc/h)

    Returns
    -------
    ray: LightRay object
        lightray created by direction and impact parameter specified
    """

    theta_away = 1.036
    phi_away = 1.314

    ray_filename = rays_directory + \
                   'ray_away_b_{:.2f}.h5'.format(impact_param)

    ray = make_ray_to_mw_with_bz((distance, theta_away, phi_away), impact_param,
                                 ray_filename)

    return ray

def make_ray_sample(impact_param_values, distance=2000):

    for b in impact_param_values:

        print('\n SAMPLING RAY FROM AWAY AT b = {} kpc/h \n'.format(b))

        sample_ray_from_away(distance = distance, impact_param = b)

        gc.collect()


if __name__ == '__main__':

    b_values_phys = np.array([20, 50, 100, 200, 500])
    b_values_code = b_values_phys * HUBBLE_2Mpc_LG

    make_ray_sample(b_values_code)
