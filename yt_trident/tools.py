import yt
import trident
from iccpy.gadget import load_snapshot
from iccpy.gadget.subfind import SubfindCatalogue
import numpy as np
import matplotlib.pyplot as plt
import pdb

LIGHTSPEED = 299792 # in km/s

my_field_def = ( "Coordinates",
            "Velocities",
            "ParticleIDs",
            "Mass",
            ("InternalEnergy", "Gas"),
            ("Density", "Gas"),
            ("ElectronAbundance", "Gas"),
#            ("NeutralHydrogenAbundance", "Gas"),
            ("Temperature", "Gas"),
            ("SmoothingLength", "Gas"),
            #("StarFormationRate", "Gas"),
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

line_table = {'Si III':1206, 'Si IIa':1190, 'Si IIb':1260, 'C II': 1334,
              'C IV':1548}

sg = trident.SpectrumGenerator(lambda_min = 1150, lambda_max = 1600,
    dlambda = 0.01)

def subhalo_center(subfind_path, snap_num, subhalo_number):
    """
    Finds and returns subhalo center by locating its minimum of potential, for
    the specified subhalo_number.
    """

    cat = SubfindCatalogue(subfind_path, snap_num)
    center = cat.subhalo[subhalo_number].pot_min
    #center = cat.subhalo[subhalo_number].com

    return center


def ray_end_from_sph(ray_start, trajectory):
    """
    Since supplying 'trajectory' argument to make_simple_ray does not seem to be
    working because some issue with yt units; am now reproducing the function
    here.

    Given a starting position, and the spherical coordinates for an end point
    (using such starting position as the origin), returns ray end point as array.
    """

    ray_start = np.asarray(ray_start)

    r, theta, phi = trajectory

    ray_end = ray_start +  r * np.array([np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta), np.cos(theta)])

    return ray_end


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


def ray_mean_density(ray, field):

    density = np.asarray(ray.r['gas', field])

    return np.mean(density)


def get_line(line, wavelength, flux, wavelength_interval):

    """
    Given a line from line_table (i.e. 'C II') and a wavelength bandwidth,
    isolates the line's wavelengths and flux, and converts wavelength to
    velocity in LSR
    Return velocity and flux arrays for the selected line

    wavelength_interval in angstroms
    """

    lambda_0 = line_table[line]

    # quedó re cabeza ese indexeado en la salida del where, ver como se arregla
    try:
        central_index = np.where(wavelength == lambda_0)[0][0]
        right_index = np.where(wavelength == int(lambda_0 + wavelength_interval/2))[0][0]
        left_index = np.where(wavelength == int(lambda_0 - wavelength_interval/2))[0][0]

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
