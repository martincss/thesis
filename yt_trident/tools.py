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
