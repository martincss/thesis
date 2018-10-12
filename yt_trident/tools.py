import yt
import trident
from iccpy.gadget import load_snapshot
from iccpy.gadget.subfind import SubfindCatalogue
import numpy as np
import matplotlib.pyplot as plt

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

def subhalo_center(subfind_path, snap_num, subhalo_number):
    """
    Finds and returns subhalo center by locating its minimum of potential, for
    the specified subhalo_number.
    """

    cat = SubfindCatalogue(subfind_path, snap_num)
    center = cat.subhalo[subhalo_number].pot_min

    return center


def ray_end_from_sph(ray_start, trajectory):
    """
    Since supplying 'trajectory' argument to make_simple_ray does not seem to be
    working because some issue with yt units; am now reproducing the function
    here.
    """

    r, theta, phi = trajectory

    ray_end = ray_start +  r * np.array([np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta), np.cos(theta)])

    return ray_end


def make_projection(ds, center, side):

    # for this to work fine and prevent colorbar to linear scale, center and 
    # box_size must be passed with these units
    center = ds.arr(center, 'code_length')
    new_box_size = ds.quan(side,'kpccm/h')

    px = yt.ProjectionPlot(ds, 'x', ('gas', 'density'), center=center,
        width=new_box_size)

    return px

def plot_ray_in_projection(px):

    px.annotate_ray(ray, arrow=True)


def ray_mean_density(ray, field):

    density = np.asarray(ray.r['index', field])

    return np.mean(density)
