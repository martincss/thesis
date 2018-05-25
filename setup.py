import sys

try:
	from setuptools import setup
	have_setuptools = True

except ImportError:
	from distutils.core import setup
	have_setuptools = False

setup_kwargs = {
	'name': 'iccpy',
	#'version': '0.1',
	'packages': ['iccpy', 'iccpy.figures', 'iccpy.flash', 'iccpy.gadget', 'iccpy.graphs', 
	'iccpy.gravsolve', 'iccpy.halo', 'iccpy.idl', 'iccpy.particles', 'iccpy.simulations'],
	'package_dir': {
	'iccpy': 'iccpy', 'iccpy.figures': 'iccpy/figures', 'iccpy.flash': 'iccpy/flash', 'iccpy.gadget': 'iccpy/gadget', 
	'iccpy.graphs': 'iccpy/graphs', 'iccpy.gravsolve': 'iccpy/gravsolve', 'iccpy.halo': 'iccpy/halo',
	'iccpy.idl': 'iccpy/idl', 'iccpy.particles': 'iccpy/particles', 'iccpy.simulations': 'iccpy/simulations'
	}
	
	}

if __name__ == '__main__':

	setup(**setup_kwargs)
