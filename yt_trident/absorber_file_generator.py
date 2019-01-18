# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
plt.ion()
import yt
yt.enable_parallelism()
import trident
from tools import make_SpectrumGenerator, get_line_observables_dict
from get_arrays_from_ray import get_data_array
import pandas as pd

#line_list = ['C II', 'C IV', 'Si III', 'Si II', 'Si IV', 'H I']
line_list = ['C III', 'C IV', 'Si III', 'Si II', 'O VI']


rays_directory = './rays_2Mpc_LG_to_mw_impact/'
absorbers_directory = './absorbers_2Mpc_LG_to_mw_impact/'
#absorbers_directory = './'


def get_lines(line_observables_dict):
    """
    Retrieves absorption line names (e.g. 'C II 1036') from a spectrum
    line_observables_dict, as its keys.

    Parameters
    ----------
    line_observables_dict: dictionary attribute from SpectrumGenerator

    Returns
    -------
    lines: list containing keys to line_observables_dict, sorted by name
    """


    lines = sorted(list(line_observables_dict.keys()))

    return lines

def write_headers(handle):
    """
    Writes long header (containing complete physical quantity names and units,
    to be skipped while opening) and short header (containing column names to
    be used in dataframe) to absorber data file.

    Parameters
    ----------
    handle: handle to open file object

    """


    header_long = ('Line\t z_cosmo\t z_doppler\t z_effective\t wavelength [A]\t'
                'delta wavelength [A]\t optical depth\t'
                'column density [cm^-2]\t thermal broadening [km/s]\t'
                'equivalent width [A]\t velocity_los [km/s]\t'
                'temperature [K]\t density [g/cm^3]\t pressure [dyne/cm^2]\t'
                'dl [kpccm/h]\t cell_volume [kpccm/h^3]\t x [kpccm/h]\t'
                'y [kpccm/h]\t z [kpccm/h]\t v_x [km/s]\t v_y [km/s]\t'
                'v_z [km/s]\t \n')

    header_short = ('Line,z_cosmo,z_dopp,z_eff,lambda,delta_lamdba,tau,N,'
                   'thermal_b,EW,v_los,T,rho,p,dl,cell_volume,x,y,z,vx,vy,vz\n')

    handle.write(header_long)
    handle.write(header_short)


def write_line(handle, cell_number, spectral_line, data_array):
    """
    Retrieves absorber attribute (e.g. N) orderded by radial distance along
    sightline, for a given absorber dataframe (df) and absorption line (line)

    Parameters
    ----------
    df: absorber dataframe
    line: line string (e.g. 'C II 1036')

    Returns
    -------
    r: radial distances from ray_end
    att: selected attribute from dataframe
    """


    line = '{:s}' + 3*',{:e}' + 2*',{:.6f}' + 16*',{:e}'+'\n'

    handle.write(line.format(spectral_line, *data_array[cell_number,:]))


def generate_absorbers_file(rays_directory, ray_filename, absorbers_directory):
    """
    Generates absorber data file for a given ray/sightline.

    Parameters
    ----------
    ray_filename: string containing filename, appended to rays_directory as path
    absorbers_directory: path to directory in which store output files
    """

    absorber_filename = 'abs_' + ray_filename[4:-3] + '.txt'

    ray = yt.load(rays_directory + ray_filename)

    sg = make_SpectrumGenerator()
    line_observables_dict = get_line_observables_dict(ray, sg, line_list)
    lines = get_lines(line_observables_dict)

    number_of_cells = ray.num_particles['grid']

    handle = open(absorbers_directory + absorber_filename, 'w')

    write_headers(handle)

    for line in lines:

        data_array = get_data_array(ray, line_observables_dict[line])

        for i in range(number_of_cells):

            write_line(handle, i, line, data_array)


if __name__ == '__main__':

    #rays_list = ['ray_1000_0.70_4.19.h5', 'ray_1000_1.05_1.40.h5', 'ray_1000_2.8_2.1.h5']

    #for ray_filename in rays_list:

    #    generate_absorbers_file(ray_filename, absorbers_directory)

    for i, handle in enumerate(glob.glob(rays_directory + 'ray*')):

        ray_filename = handle.split('/')[2]
        absorber_filename = 'abs_' + ray_filename[4:-3] + '.txt'

        if not os.path.exists(absorbers_directory + absorber_filename):

            print('\n Generating file for ray #{:2d} ~~~~~~~~~~~\n'.format(i+1))

# kind of hard-coded, ray_filename extracted from rays_directory + ray_filename
            generate_absorbers_file(rays_directory, ray_filename,
                                    absorbers_directory)
