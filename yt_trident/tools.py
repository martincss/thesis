import yt
import trident
from iccpy.gadget import load_snapshot
from iccpy.gadget.subfind import SubfindCatalogue
import numpy as np
from numpy import pi
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
import pandas as pd
import pdb

LIGHTSPEED = 299792 # in km/s
HUBBLE_2Mpc_LG = 0.7

# do not comment any further fields to load in dataset; for some reason,
# (seemingly an YT bug), gas density is not smoothed correctly if fewer fields
# than these are present

my_field_def = ( "Coordinates",
            "Velocities",
            "ParticleIDs",
            "Mass",
            ("InternalEnergy", "Gas"),
            ("Density", "Gas"),
            ("ElectronAbundance", "Gas"),
            ("NeutralHydrogenAbundance", "Gas"),
            ("Temperature", "Gas"),
            ("SmoothingLength", "Gas"),
#            ("StarFormationRate", "Gas"),
            ("Age", "Stars"),
            ("Metallicity", "Gas"),
#            ("Metallicity", "Stars"),
)
# length es kpc/h, h en el header
# mass es 10^10 m_sun / h
# vel es km/s
unit_base = {'length'   :  (1.0, 'kpccm/h'),
             'mass'     :   (1.0e10, 'Msun'),
             'velocity' :      (1.0, 'km/s')}

line_table = {'Si III':1206, 'Si IIa':1190, 'Si IIb':1260, 'C II': 1334.532,
              'C IV':1548}

all_line_keys = ['C II* 1336', 'C II 1335', 'C II* 1037', 'C II 1036',
                 'C II 904', 'C IV 1551', 'C IV 1548', 'Si III 1206',
                 'Si II* 1817', 'Si II 1808', 'Si II* 1309', 'Si II 1304',
                 'Si II* 1265', 'Si II 1260', 'Si II* 1197', 'Si II* 1194',
                 'Si II 1193', 'Si II 1190', 'Si II* 1024', 'Si II 1021',
                 'Si II* 993', 'Si II 990']


def make_SpectrumGenerator():
    """
    Convenience function to generate a SpectrumGenerator instance with preset
    wavelength range.
    """

    sg = trident.SpectrumGenerator(lambda_min = 1150, lambda_max = 1600,
        dlambda = 0.01)

    return sg

def get_line_observables_dict(ray, sg, line_list):
    """
    Convenience function to make and retrive line_observables_dict dictionary
    from an SpectrumGenerator object.
    """

    sg.make_spectrum(ray, lines = line_list, store_observables = True)

    return sg.line_observables_dict



def subhalo_center(subfind_path, snap_num, subhalo_number):
    """
    Finds and returns subhalo center by locating its minimum of potential, for
    the specified subhalo_number.
    """

    cat = SubfindCatalogue(subfind_path, snap_num)
    #center = cat.subhalo[subhalo_number].pot_min
    center = cat.subhalo[subhalo_number].com

    return center


def sphere_uniform_grid(number_of_points):
    """
    Given a number of points, generates two arrays containing the theta and phi
    coordinates of a uniform distribution of points on the sphere.

    Taken from https://stackoverflow.com/a/44164075
    """

    indices = np.arange(0, number_of_points, dtype=float) + 0.5

    theta = np.arccos(1 - 2*indices/number_of_points)
    phi = pi * (1 + 5**0.5) * indices

    return theta, phi


def ray_start_from_sph(ray_end, trajectory):
    """
    Since supplying 'trajectory' argument to make_simple_ray does not seem to be
    working because some issue with yt units; am now reproducing the function
    here.

    Given an ending position, and the spherical coordinates for a starting point
    (using such ending position as the origin), returns ray start point as array.
    """

    ray_end = np.asarray(ray_end)

    r, theta, phi = trajectory

    ray_start = ray_start +  r * np.array([np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta), np.cos(theta)])

    return ray_start


def make_projection(ds, center, side, axis):

    # for this to work fine and prevent colorbar to linear scale, center and
    # box_size must be passed with these units
    center = ds.arr(center, 'code_length')
    new_box_size = ds.quan(side,'kpccm/h')

    px = yt.ProjectionPlot(ds, axis, ('gas', 'density'), center=center,
        width=new_box_size)

    return px

def plot_ray_in_projection(px, ray):

    px.annotate_ray(ray, arrow=True)


def make_slice(ds, center, side, axis):

    center = ds.arr(center, 'code_length')
    new_box_size = ds.quan(side,'kpccm/h')

    slc = yt.SlicePlot(ds, axis, ('gas', 'density'), center = center,
        width = new_box_size)

    return slc

def plot_ray_in_slice(slc, ray):

    slc.annotate_ray(ray, arrow=True)


def ray_mean_density(ray, field):

    density = np.asarray(ray.r['gas', field])

    return np.mean(density)


def get_line(lambda_0, wavelength, flux, wavelength_interval):

    """
    Given a central wavelength lambda_0 and a wavelength bandwidth,
    isolates wavelengths and flux in that interval, and converts wavelength to
    velocity in LSR
    Return velocity and flux arrays for the selected line

    lambda_0 and wavelength_interval in angstroms
    """


    try:
        central_index = np.argmin(np.abs(wavelength - lambda_0))
        right_index = np.argmin(np.abs(wavelength - lambda_0 - wavelength_interval/2))
        left_index = np.argmin(np.abs(wavelength - lambda_0 + wavelength_interval/2))

        wavelength_section = wavelength[left_index:right_index]
        delta_lambda = wavelength_section - lambda_0
        flux_section = flux[left_index:right_index]
        velocity = LIGHTSPEED * delta_lambda/lambda_0
        #pdb.set_trace()

    except:
        velocity = np.array([])
        flux_section = np.array([])
        print('Wavelength not in spectrum')

    return velocity, flux_section
    #return wavelength_section, flux_section


def get_absorber_chars(ray, line_key, line_list):
    """
    Given a ray and a line_key to the line_observables_dict (e.g. 'C II 1335'),
    returns wavelength at cell with max column_density, such column_density,
    temperature median along the ray and position of max column_density.

    line_list must be supplied.
    """

    sg = make_SpectrumGenerator()
    observables_dict = get_line_observables_dict(ray, sg, line_list)

    N = np.array(observables_dict[line_key]['column_density']).max()
    T = np.median(np.array(ray.r['temperature']))

    index_absorber = np.argmax(np.array(observables_dict[line_key]['column_density']))

    lambda_obs = np.array(observables_dict[line_key]['lambda_obs'])[index_absorber]

    position = (np.array(ray.r['x'].in_units('kpccm/h'))[index_absorber],
                np.array(ray.r['y'].in_units('kpccm/h'))[index_absorber],
                np.array(ray.r['z'].in_units('kpccm/h'))[index_absorber])

    return lambda_obs, N, T, position


def get_absorber_chars_from_file(absorber_filename, line_key):

    df = pd.read_csv(absorber_filename, skiprows=1)
    line_df = (df[df['Line']== line_key])

    N = line_df['N'].max()
    T = line_df['T'].median()

    index_absorber = np.argmax(line_df['N'])

    lambda_obs = line_df['lambda'][index_absorber]

    position = (line_df['x'][index_absorber], line_df['y'][index_absorber],
                line_df['z'][index_absorber])

    return lambda_obs, N, T, position


def absorber_region_2Mpc_LG(absorber_position):

    mw_center = subhalo_center(subfind_path='../../2Mpc_LG', snap_num=135,
                subhalo_number = 1)

    m31_center = subhalo_center(subfind_path='../../2Mpc_LG', snap_num=135,
                subhalo_number = 0)

    R_vir_mw = 222.2 * HUBBLE_2Mpc_LG
    R_vir_m31 = 244.9 * HUBBLE_2Mpc_LG

    # R_vir_mw = 100
    # R_vir_m31 = 100

    R_gg = m31_center - mw_center

    r_mw_abs = np.array(absorber_position) - mw_center

    r_disc_abs = r_mw_abs - (r_mw_abs @ R_gg) * R_gg / norm(R_gg)**2

    if norm(np.array(absorber_position) - mw_center) <= R_vir_mw:

        return 'MW halo'

    elif norm(np.array(absorber_position) - m31_center) <= R_vir_m31:

        return 'M31 halo'

    elif norm(r_disc_abs) <= max((R_vir_mw, R_vir_m31)) and \
         norm(r_mw_abs) < norm (R_gg):

         return 'bridge'

    else:

        return 'IGM'
