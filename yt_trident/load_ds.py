import yt
import trident
from iccpy.gadget import load_snapshot
from gadget_test_one_metal.labels import cecilia_labels
import numpy as np
HUBBLE = 0.7

file = 'snapdir_135/snap_LG_WMAP5_2048_135.0'
snap_dir = '.'

snap = load_snapshot(directory = snap_dir, snapnum = 135, label_table = cecilia_labels)

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

ds = yt.frontends.gadget.GadgetDataset(filename=file, unit_base= unit_base, field_spec=my_field_def)

density = ds.r[("Gas","density")]
wdens = np.where(density == np.max(density))
coordinates = ds.r[("Gas","Coordinates")]
center = coordinates[wdens][0]

new_box_size = ds.quan(2000,'kpccm/h')

left_edge = center - new_box_size/2
right_edge = center + new_box_size/2

px = yt.ProjectionPlot(ds, 'x', ('gas', 'density'), center=center, width=new_box_size)
px.show()
px.save('prueba_projection.png')
