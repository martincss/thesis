import yt
from yt import derived_field
import trident
from iccpy.gadget import load_snapshot
from iccpy.gadget.subfind import SubfindCatalogue
import numpy as np
from numpy import pi
from numpy.linalg import norm
from numpy.random import random
import matplotlib.pyplot as plt
import math
import pandas as pd
import pdb
import glob
from multiprocessing import Pool, cpu_count
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.cosmology.core import CosmologyError
import astropy.units as u


LIGHTSPEED = 299792 # in km/s
HUBBLE_2Mpc_LG = 0.7
K_BOLTZMANN = 1.38064852e-16 # erg/K
GRAV_CONST = 4.3071e-6 # kpc M_sun^-1 (km/s)^2

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

line_table = {'Si III 1206':1206.5, 'Si II 1193':1193.29,
              'C II 1335': 1334.532, 'C IV 1548':1548.19, 'Ly a':1215.67}

all_line_keys=['C II 1036', 'C II 1335', 'C II 904', 'C II* 1037', 'C II* 1336',
       'C IV 1548', 'C IV 1551', 'Ly 10', 'Ly 11', 'Ly 12', 'Ly 13',
       'Ly 14', 'Ly 15', 'Ly 16', 'Ly 17', 'Ly 18', 'Ly 19', 'Ly 20',
       'Ly 21', 'Ly 22', 'Ly 23', 'Ly 24', 'Ly 25', 'Ly 26', 'Ly 27',
       'Ly 28', 'Ly 29', 'Ly 30', 'Ly 31', 'Ly 32', 'Ly 33', 'Ly 34',
       'Ly 35', 'Ly 36', 'Ly 37', 'Ly 38', 'Ly 39', 'Ly 6', 'Ly 7',
       'Ly 8', 'Ly 9', 'Ly a', 'Ly b', 'Ly c', 'Ly d', 'Ly e',
       'O VI 1032', 'O VI 1038',
       'Si II 1021', 'Si II 1190', 'Si II 1193', 'Si II 1260',
       'Si II 1304', 'Si II 1808', 'Si II 990', 'Si II* 1024',
       'Si II* 1194', 'Si II* 1197', 'Si II* 1265', 'Si II* 1309',
       'Si II* 1817', 'Si II* 993', 'Si III 1206', 'Si IV 1403']


# @derived_field(name="pressure", units="dyne/cm**2")
# def _pressure(field, data):
#     return data["density"] * data["thermal_energy"]

cosmo = FlatLambdaCDM(H0=70, Om0=0.279)

def usable_cores():

    if cpu_count() == 12:
         number_of_cores = 5
    else:
         number_of_cores = 2

    return number_of_cores

def z_from_distance(r):
    """
    Computes redshift for a given comoving length in kpc/h
    """
    try:
        # length must be in physical units first
        z = z_at_value(cosmo.comoving_distance, r/HUBBLE_2Mpc_LG * u.kpc)
    except CosmologyError:
        z = 0

    return z


def get_2Mpc_LG_dataset():

    #snap_file = './snapdir_135/snap_LG_WMAP5_2048_135.0'
    snap_file = ('../../2Mpc_LG_corrected_MWcenter/snapdir_135/'
                'snap_LG_WMAP5_2048_135.0')

    ds = yt.frontends.gadget.GadgetDataset(filename = snap_file,
                                           unit_base = unit_base,
                                           field_spec = my_field_def)
    #ds.add_field(("gas", "pressure"), function=_pressure, units="dyne/cm**2")

    return ds



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



def subhalo_center(subhalo_number, subfind_path='../../2Mpc_LG',
                          snap_num=135):
    """
    Finds and returns subhalo center by locating its minimum of potential, for
    the specified subhalo_number.
    """

    cat = SubfindCatalogue(subfind_path, snap_num)
    center = cat.subhalo[subhalo_number].pot_min
    #center = cat.subhalo[subhalo_number].com

    return center


def subhalo_virial_radius(subhalo_number, subfind_path='../../2Mpc_LG',
                          snap_num=135):
    """
    Finds and returns subhalo virual radius by
    for
    the specified subhalo_number.
    """

    cat = SubfindCatalogue(subfind_path, snap_num)
    # as mass units are 10^10 M_sun, we scale mass for units to be M_sun
    subhalo_mass = cat.subhalo[subhalo_number].mass*10**10

    r_200 = (GRAV_CONST * HUBBLE_2Mpc_LG * subhalo_mass)**(1/3) # in kpc/h

    return r_200



def get_mw_center_2Mpc_LG():
    """
    Convenience function to return mw subhalo center from 2Mpc_LG dataset

    Returns
    -------
    mw_center: ndarray
        Three dimensional array containing coordinates of mw subhalo center
    """

    mw_center = subhalo_center(subfind_path = '../../2Mpc_LG', snap_num = 135,
                               subhalo_number = 1)

    return mw_center

def get_m31_center_2Mpc_LG():
    """
    Convenience function to return m31 subhalo center from 2Mpc_LG dataset

    Returns
    -------
    mw_center: ndarray
        Three dimensional array containing coordinates of m31 subhalo center
    """

    m31_center = subhalo_center(subfind_path = '../../2Mpc_LG', snap_num = 135,
                               subhalo_number = 0)

    return m31_center

def get_disk_normal_vector_mw_2Mpc_LG():
    """
    Convenience function to return a vector normal to the MW disk from 2Mpc_LG
    dataset

    Returns
    -------
    perp: ndarray
        Three dimensional array containing components of a vector normal to MW
        disk
    """

    # rotation matrix from sim coordinates to coordinates with disk on XY plane
    rot = np.array([[0.38444049,-0.89949922,-0.20762143],
                    [0.70601157,0.43138525,-0.56165330],
                    [0.59477153,0.069339139,0.80089882]])

    # we map the new z axis (perp to the disk) to the old (sim) coordinates by
    # inverting the rotation
    perp = rot.transpose() @ np.array([0,0,1])

    return perp


def get_sun_position_2Mpc_LG():
    """
    Convenience function to return a sun position from 2Mpc_LG dataset

    Returns
    -------
    perp: ndarray
        Three dimensional array containing coordinates of possible sun
    """


    mw_center = get_mw_center_2Mpc_LG()
    normal = get_disk_normal_vector_mw_2Mpc_LG()

    # there is no unique choice for this vector
    arbitrary_vector = [1,0,0]
    r_center_to_sun = np.cross(normal, arbitrary_vector)

    # we set the sun 10kpc from mw center
    r_center_to_sun *= 10*HUBBLE_2Mpc_LG / norm(r_center_to_sun)

    r_sun = mw_center + r_center_to_sun

    return r_sun

def sph_to_cart(coordinates):

    r, theta, phi = coordinates

    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return np.array([x,y,z])


def cart_to_sph(coordinates):
    """
    Converts three dimensional array in cartesian coordinates to spherical
    coordinates

    Parameters
    ----------
    coordinates: iterable
        should have three elements representing cartesian coordinates of a
        vector

    Returns
    -------
    (r, theta, phi): tuple of floats
        spherical coordinates of inputted vector
    """

    x,y,z = coordinates

    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y,x) % (2*pi)
    theta = np.arccos(z/r)

    return (r, theta, phi)


def sphere_uniform_grid(number_of_points):
    """
    Given a number of points, generates two arrays containing the theta and phi
    coordinates of a uniform distribution of points on the sphere.

    Taken from https://stackoverflow.com/a/44164075
    """

    indices = np.arange(0, number_of_points, dtype=float) + 0.5

    theta_array = np.arccos(1 - 2*indices/number_of_points)
    phi_array = pi * (1 + 5**0.5) * indices
    phi_array %= (2*pi)

    return theta_array, phi_array


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

    ray_start = ray_end +  r * np.array([np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta), np.cos(theta)])

    return ray_start


def ray_end_from_sph(ray_start, trajectory):
    """
    Since supplying 'trajectory' argument to make_simple_ray does not seem to be
    working because some issue with yt units; am now reproducing the function
    here.

    Given an ending position, and the spherical coordinates for a starting point
    (using such ending position as the origin), returns ray start point as array.
    """

    ray_end = np.asarray(ray_end)

    r, theta, phi = trajectory

    ray_end = ray_start -  r * np.array([np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta), np.cos(theta)])

    return ray_end


def displacement_from_subhalo(R_vir):
    """
    Given a subhalo virial radius, returns a vector with random direction and
    length from 0 to 2*R_vir; representing a random displacement from a subhalo
    center.

    Parameters
    ----------
    R_vir: float
        Virial radius of subhalo to consider.

    Returns
    -------
    disp: ndarray
        three dimensional vector with length between 0 and 2*R_vir (in same
        units as R_vir).
    """

    disp = random(3)
    distance_from_center = random(1)*2*R_vir

    disp *= distance_from_center/norm(disp)

    return disp

def fuzzy_samples_from_subhalo(subhalo_number, number_of_samples):
    """
    For the selected subhalo, generates a number of (absolute) positions of
    random points inside its virial radius.

    Parameters
    ----------
    subhalo_number: integer
        index for the desired subhalo (0 for M31, 1 for MW, etc)

    number_of_samples: integer
        number of random points to construct

    Returns
    -------
    ray_starts: list
        list of absolute positions of the sampled points.

    """

    center = subhalo_center(subhalo_number)
    R_vir = subhalo_virial_radius(subhalo_number)


    ray_starts = [displacement_from_subhalo(R_vir) + center for i in \
                  range(number_of_samples)]


    return ray_starts


def make_projection(ds, center, side, axis, field='density'):

    # for this to work fine and prevent colorbar to linear scale, center and
    # box_size must be passed with these units
    center = ds.arr(center, 'code_length')
    new_box_size = ds.quan(side,'kpccm/h')

    p = yt.ProjectionPlot(ds, axis, ('gas', field), center=center,
        width=new_box_size)

    return p

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

def lambda_to_velocity(wavelength, lambda_0):

    velocity = (wavelength - lambda_0)/lambda_0 * LIGHTSPEED

    return velocity
    #return wavelength

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
        right_index = np.argmin(np.abs(wavelength - lambda_0 - \
                                       wavelength_interval/2))
        left_index = np.argmin(np.abs(wavelength - lambda_0 + \
                                      wavelength_interval/2))

        wavelength_section = wavelength[left_index:right_index]
        flux_section = flux[left_index:right_index]
        velocity = lambda_to_velocity(wavelength_section, lambda_0)
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

    #lambda_obs = np.array(observables_dict[line_key]['lambda_obs'])[index_absorber]
    lambda_obs = line_table[line_key]

    position = (np.array(ray.r['x'].in_units('kpccm/h'))[index_absorber],
                np.array(ray.r['y'].in_units('kpccm/h'))[index_absorber],
                np.array(ray.r['z'].in_units('kpccm/h'))[index_absorber])

    return lambda_obs, N, T, position


def get_absorber_chars_from_file(absorber_filename, line_key):
    """
    Returns absorber characteristics from an absorber file for a single
    absorption line

    Parameters
    ----------
    absorber_filename: string
        full path to absorber file
    line_key: string
        full line name present in absorber file (e.g. 'C II 1335')

    Returns
    -------
    lambda_obs: float
        central wavelength in absorption line
    N: float
        column density corresponding to the cell with lambda_obs
    T: float
        median temperature across all cells in ray
    position: tuple of floats
        absolute position of the cell with lambda_obs
    """

    df = pd.read_csv(absorber_filename, skiprows=1)
    line_df = (df[df['Line']== line_key])

    #N = line_df['N'].max()
    T = line_df['T'].median()

    #index_absorber = np.argmax(line_df['N'])

    # identifies _the_ absorber by the maximum rho and its
    # wavelength as the central
    try:
        index_absorber = np.argmax(line_df['rho'])

    except KeyError:
        index_absorber = np.argmax(line_df['N'])

    N = line_df['N'][index_absorber]

    #lambda_obs = line_df['lambda'][index_absorber]
    lambda_obs = line_table[line_key]


    position = (line_df['x'][index_absorber], line_df['y'][index_absorber],
                line_df['z'][index_absorber])

    return lambda_obs, N, T, position


def absorber_region_2Mpc_LG(absorber_position):

    mw_center = get_mw_center_2Mpc_LG()

    m31_center = get_m31_center_2Mpc_LG()

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

def absorber_mean_vlos(absorber_filename):

    df = pd.read_csv(absorber_filename, skiprows=1)
    linekey = df['Line'].iloc[0]

    v_los = df[df['Line']==linekey]['v_los']

    return v_los.mean()


def identify_hvcs_single_line(df, line):
    """
    Retrieves (at most 2) HVCs from an absorber dataframe for a given absorption
    line.
    HVCs are selected as first to maxima of optical depth tau, if the second if
    greater than 33% of the first.

    Parameters
    ----------
    df: dataframe
        absorber dataframe
    line: string
        absorption line string

    Returns
    -------
    dataframe with identified HVCs
    """

    line_df = df[df['Line'] == line]

    # index_absorber = line_df['rho'].argmax()
    # lambda_obs = line_df['lambda'][index_absorber]
    lambda_obs = line_table[line]
    vel = lambda_to_velocity(line_df['lambda'], lambda_obs)

    condition = (np.abs(vel) > 100) & (line_df['tau'] != 0)
    candidates = line_df[condition].sort_values('tau', ascending = False)
    candidates['vel_spectrum'] = vel[condition]

    try:
        absorbers = [candidates.iloc[0]]

    except IndexError:

        absorbers = None

    try:
        tau_1st = candidates['tau'].iloc[0]
        tau_2nd = candidates['tau'].iloc[1]

        if tau_2nd > tau_1st/3:

            absorbers.append(candidates.iloc[1])
#        pdb.set_trace()
        absorbers = pd.concat(absorbers, axis=1)

    except IndexError:

        absorbers = None

    return absorbers

def identify_hvcs_all_lines(df):

    absorbers = []

    #for line in df['Line'].unique():
    for line in ['Si III 1206', 'C II 1335', 'Si II 1193', 'C IV 1548']:

        identified = identify_hvcs_single_line(df, line)

        if identified is not None:

            absorbers.append(identified)
    try:
#        pdb.set_trace()
        df_all = pd.concat(absorbers, axis=1)

    except ValueError:

        df_all = None

    return df_all

def identify_one_to_map(path_to_absorber):

    df = pd.read_csv(path_to_absorber, skiprows=1)
    #print(path_to_absorber)

    hvcs = identify_hvcs_all_lines(df)

    return hvcs


def retrieve_all_hvcs(absorbers_directory, pool, subsampling=True):

    if subsampling:
        handles = [handle for handle in glob.glob(absorbers_directory + 'abs*') if \
                   handle_in_subsample(handle, amplitude_polar=1)]

    else:
        handles = [handle for handle in glob.glob(absorbers_directory + 'abs*')]

    results = pool.map(identify_one_to_map, handles)

    hvcs = pd.concat([df for df in results if df is not None], axis=1)

    return hvcs.T


def extract_angles_from_handle(handle, file='abs'):
    """

    Parameters
    ----------
    handle
        must be in the form
        './absorbers_2Mpc_LG_to_m31_210/abs_210.000_0.168_4.166.txt'
        with both angles as {:.3f}

    """
    end_by_filetype = {'rays':-3, 'abs':-4}

    theta, phi =handle[:end_by_filetype[file]].split('_')[-2:]
    theta = float(theta); phi = float(phi)

    return theta, phi


def extract_sub_coordinates_from_handle(handle, file='abs'):
    """

    Parameters
    ----------
    handle
        must be in the form
   './absorbers_2Mpc_LG_from_subhalos_fuzzy/abs_fsub_00_488.712_1.156_4.498.txt'
        with both angles as {:.3f}

    """
    end_by_filetype = {'rays':-3, 'abs':-4}

    sub, r, theta, phi =handle[:end_by_filetype[file]].split('_')[-4:]
    sub = int(sub); r = float(r); theta = float(theta); phi = float(phi)

    return sub, r, theta, phi



def select_polar_rays(theta, phi, amplitude = 0.52):

    # polar_vect already has unit norm
    polar_vect = get_disk_normal_vector_mw_2Mpc_LG()

    # select for rays close to north pole (with polar_theta) or south pole
    # (with pi - polar_theta)

    in_cone = (np.abs(polar_vect @ sph_to_cart((1, theta, phi))) > np.cos(amplitude))

    return in_cone

def select_m31_rays(theta, phi, amplitude = 0.52,
                    observer = get_sun_position_2Mpc_LG()):


    m31_center = get_m31_center_2Mpc_LG()
    unit_to_m31 = m31_center - observer
    unit_to_m31 /= norm(unit_to_m31)

    in_cone = ((unit_to_m31 @ sph_to_cart((1, theta, phi))) > np.cos(amplitude))

    return in_cone


def sightline_in_subsample(theta, phi, amplitude_polar):

    in_cone = select_polar_rays(theta, phi, amplitude_polar) or \
                select_m31_rays(theta, phi)

    return in_cone


def handle_in_subsample(handle, amplitude_polar):

    theta, phi = extract_angles_from_handle(handle)

    return sightline_in_subsample(theta, phi, amplitude_polar)
