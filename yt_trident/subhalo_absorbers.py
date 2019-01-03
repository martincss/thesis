# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from absorber_file_generator import generate_absorbers_file

rays_directory = './rays_2Mpc_LG_from_subhalos/'
absorbers_directory = './absorbers_2Mpc_LG_subhalos/'


def generate_all_absorber_files():

    for i, handle in enumerate(glob.glob(rays_directory + 'ray*')):

        ray_filename = handle.split('/')[2]
        absorber_filename = 'abs_' + ray_filename[4:-3] + '.txt'

        if not os.path.exists(absorbers_directory + absorber_filename):

            print('\n Generating file for ray #{:2d} ~~~~~~~~~~~\n'.format(i+1))

# kind of hard-coded, ray_filename extracted from rays_directory + ray_filename
            generate_absorbers_file(ray_filename, absorbers_directory)
