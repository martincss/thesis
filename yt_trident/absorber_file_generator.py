# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()
import yt
yt.enable_parallelism()
import trident
from tools import make_SpectrumGenerator, get_line_observables_dict
from get_arrays_from_ray import get_data_array
import pandas as pd

line_list = ['C II', 'C IV', 'Si III', 'Si II']

rays_directory = './rays_2Mpc_LG_from_mw/'
#absorbers_directory = './absorbers_2Mpc_LG_from_mw/'
absorbers_directory = './'


def get_lines(line_observables_dict):

    lines = sorted(list(line_observables_dict.keys()))

    return lines

def write_headers(handle):

    header_long = 'Line\tz_cosmo\tz_doppler\tz_effective\twavelength [A]\t \
                delta wavelength [A]\t optical depth\t column density [cm^-2]\t \
                thermal broadening [km/s]\t equivalent width [A]\t velocity_los [km/s]\t \
                temperature [K]\t dl [kpccm/h]\t cell_volume [kpccm/h^3]\t \
                x [kpccm/h]\t y [kpccm/h]\t z [kpccm/h]\t v_x [km/s]\t v_y [km/s]\t \
                v_z [km/s]\t\n'

    header_short = 'Line,z_cosmo,z_dopp,z_eff,lambda,delta lamdba,tau,N,thermal_b,EW,v_los,T,dl,cell_volume,x,y,z,vx,vy,vz\n'

    handle.write(header_long)
    handle.write(header_short)


def write_line(handle, cell_number, spectral_line, data_array):

    line = '{:s}' + 3*',{:e}' + 2*',{:.6f}' + 14*',{:e}'+'\n'

    handle.write(line.format(spectral_line, *data_array[cell_number,:]))


def generate_absorbers_file(ray_filename, absorbers_directory):
    """

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

    ray_filename = 'ray_990_0.70_4.19.h5'

    generate_absorbers_file(ray_filename, absorbers_directory)

#
# ray_filename = 'rays_2Mpc_LG_from_mw/ray_990_0.70_4.19.h5'
#
#
# line_list = ['C II', 'C IV', 'Si III', 'Si II']
# ray = yt.load(ray_filename)
# sg = make_SpectrumGenerator()
# line_observables_dict = get_line_observables_dict(ray, sg, line_list)
# dicc = line_observables_dict['C II 1335']
