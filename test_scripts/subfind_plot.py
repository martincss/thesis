# -*- coding: utf-8 -*-
"""
Test script to try out subfind and plotting
for each subhalo

"""

import numpy as np
import matplotlib.pyplot as plt

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue

subfind_path = '../../2Mpc_LG'
snap_num = 135
snap_dir = '../../2Mpc_LG'


cat = SubfindCatalogue(subfind_path, snap_num)
# Directory and snapnum are passed instead of filename, to open snapshot in multiple parts
snap = load_snapshot(directory = snap_dir.format(snap_num), snapnum = snap_num, label_table = cecilia_labels)



