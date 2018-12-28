# -*- coding: utf-8 -*-
"""
Toolbox of Convenience functions to retrive arrays from ray object and
line_observables_dict.
"""
import numpy as np
import gc

def get_z_cosmo(ray):

    z_cosmo = np.array(ray.r['redshift'])

    return z_cosmo

def get_z_dopp(ray):

    z_dopp = np.array(ray.r['redshift_dopp'])

    return z_dopp

def get_z_eff(ray):

    z_eff = np.array(ray.r['redshift_eff'])

    return z_eff

def get_lambda_obs_angstrom(line_observables_dict):

    lambda_obs = np.array(line_observables_dict['lambda_obs'])

    return lambda_obs

def get_delta_lambda_angstrom(line_observables_dict):

    delta_lambda = np.array(line_observables_dict['delta_lambda'])

    return delta_lambda

def get_tau_ray(line_observables_dict):

    tau_ray = np.array(line_observables_dict['tau_ray'])

    return tau_ray

def get_column_density_cm2(line_observables_dict):

    column_density = np.array(line_observables_dict['column_density'])

    return column_density


def get_thermal_broadening_kms(line_observables_dict):

    thermal_b = np.array(line_observables_dict['thermal_b'])

    return thermal_b

def get_equivalent_width_angstrom(line_observables_dict):
    """
    This quantity is a single number for the whole line; to make things easier
    I'm copying that same value for each cell.
    """

    size = len(line_observables_dict['column_density'])
    EW = float(line_observables_dict['EW']) * np.ones(size)

    return EW

def get_velocity_los_kms(ray):

    v_los = np.array(ray.r['velocity_los'].in_units('km/s'))

    return v_los

def get_temperature(ray):

    temp = np.array(ray.r['temperature'])

    return temp

def get_dl_kmccmh(ray):

    dl = np.array(ray.r['dl'].in_units('kpccm/h'))

    return dl

def get_cell_volume_kpccmh3(ray):

    cell_volume = np.array( ray.r['dx'].in_units('kpccm/h') * \
    ray.r['dy'].in_units('kpccm/h') * ray.r['dz'].in_units('kpccm/h') )

    return cell_volume

def get_x_kmccmh(ray):

    x = np.array(ray.r['x'].in_units('kpccm/h'))

    return x

def get_y_kmccmh(ray):

    y = np.array(ray.r['y'].in_units('kpccm/h'))

    return y

def get_z_kmccmh(ray):

    z = np.array(ray.r['z'].in_units('kpccm/h'))

    return z

def get_vx_kmccmh(ray):

    vx = np.array(ray.r['velocity_x'].in_units('km/s'))

    return vx

def get_vy_kmccmh(ray):

    vy = np.array(ray.r['velocity_y'].in_units('km/s'))

    return vy

def get_vz_kmccmh(ray):

    vz = np.array(ray.r['velocity_z'].in_units('km/s'))

    return vz


def get_data_array(ray, line_observables_dict):

    z_cosmo = get_z_cosmo(ray)
    z_dopp = get_z_dopp(ray)
    z_eff = get_z_eff(ray)
    lambda_obs = get_lambda_obs_angstrom(line_observables_dict)
    delta_lambda = get_delta_lambda_angstrom(line_observables_dict)
    tau_ray = get_tau_ray(line_observables_dict)
    column_density = get_column_density_cm2(line_observables_dict)
    thermal_b = get_thermal_broadening_kms(line_observables_dict)
    EW = get_equivalent_width_angstrom(line_observables_dict)
    v_los = get_velocity_los_kms(ray)
    temp = get_temperature(ray)
    dl = get_dl_kmccmh(ray)
    cell_volume = get_cell_volume_kpccmh3(ray)
    x = get_x_kmccmh(ray)
    y = get_y_kmccmh(ray)
    z = get_z_kmccmh(ray)
    vx = get_vx_kmccmh(ray)
    vy = get_vy_kmccmh(ray)
    vz = get_vz_kmccmh(ray)

    array_tuple = (z_cosmo, z_dopp, z_eff, lambda_obs, delta_lambda, tau_ray,
                column_density, thermal_b, EW, v_los, temp, dl, cell_volume,
                x, y, z, vx, vy, vz)

    return np.column_stack(array_tuple)
