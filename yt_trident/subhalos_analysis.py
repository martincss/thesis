#!/usr/bin/env python
from tools import subhalo_center, subhalo_virial_radius, \
                  get_sun_position_2Mpc_LG as sun, sph_to_cart, \
                  extract_sub_coordinates_from_handle

import numpy as np
from numpy.linalg import norm
from pandas import read_csv, DataFrame

abs_lines = ['Si III 1206', 'Si II 1193', 'C II 1335', 'C IV 1548']

def impact_parameter_from_handle(handle):

    sub_num, r, theta, phi = extract_sub_coordinates_from_handle(handle)

    sub_pos = subhalo_center(sub_num)
    sun_pos = sun()

    sun_to_sub = (sub_pos-sun_pos)
    normal = sun_to_sub/ norm(sun_to_sub)

    disp_from_sub = sph_to_cart((r,theta,phi)) - sun_to_sub

    r_impact = disp_from_sub - (disp_from_sub @ normal)*normal
    impact_parameter = norm(r_impact)
    # assert impact_parameter < norm(disp_from_sub)
    # import pdb; pdb.set_trace()
    return impact_parameter


def absorbers_in_subhalo(handle, max_flux = 1.0):

    sub_num, _, _, _ = extract_sub_coordinates_from_handle(handle)

    df = read_csv(handle, skiprows=1)

    N_by_line = {}

    for line in abs_lines:

        dfl = df[df['Line'] == line]

        x = np.array(dfl['x']).reshape((len(dfl['x']),1))
        y = np.array(dfl['y']).reshape((len(dfl['y']),1))
        z = np.array(dfl['z']).reshape((len(dfl['z']),1))

        disp_from_sub = np.hstack([x,y,z]) - subhalo_center(sub_num)

        in_subhalo = norm(disp_from_sub, axis=1) < \
                     2*subhalo_virial_radius(sub_num)
        df_in = dfl[in_subhalo]

        absorbers = df_in[ np.exp(-df_in['tau']) < max_flux ]
        absorbers_N = absorbers['N'].sum()

        N_by_line[line] = [absorbers_N]

    return N_by_line


def subhalo_analysis_one_to_map(handle):

    sub_num, _, _, _ = extract_sub_coordinates_from_handle(handle)
    b = impact_parameter_from_handle(handle)
    N_by_line = absorbers_in_subhalo(handle)

    N_by_line['sub'] = [sub_num]; N_by_line['b'] = [b]

    analysis = DataFrame(N_by_line)
    analysis = analysis.reindex(columns = ['sub', 'b'] + abs_lines)

    return analysis
