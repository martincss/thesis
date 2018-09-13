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
px.save('load_projection_trans.png', mpl_kwargs = {'transparent':True})


line_list = ['C', 'N', 'O', 'Mg']

# se quejaba de el lighray anterior que pasaba por todas celdas de gas con temp=0,
ray = trident.make_simple_ray(ds,
                              start_position=left_edge,
                              end_position=right_edge,
                              data_filename="ray.h5",
                              lines=line_list,
                              ftype='Gas')

px.annotate_ray(ray, arrow=True)
px.save('load_projection_ray_trans.png', mpl_kwargs = {'transparent':True})

sg = trident.SpectrumGenerator('COS-G130M')
sg.make_spectrum(ray, lines=line_list)
sg.save_spectrum('spec_raw.txt')
sg.plot_spectrum('spec_raw.png')

#sg.add_qso_spectrum()
#sg.add_milky_way_foreground()
sg.apply_lsf()
sg.add_gaussian_noise(30)

sg.save_spectrum('spec_final.txt')
sg.plot_spectrum('spec_final.png')
