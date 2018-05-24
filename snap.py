import numpy as np
import pylab as pl

from iccpy.gadget import load_snapshot

from iccpy.gadget.labels import cecilia_labels


#f = load_snapshot('/Users/cecilia/data/Aquarius/halo_C09/snap_C09_200_converted_128', label_table=cecilia_labels)

#f = load_snapshot('/home/cscannapieco/data/isolated_random/single_explosion/outputs/snap_200', label_table=cecilia_labels)
f = load_snapshot('../outputs/snap_200', label_table=cecilia_labels)
list(f)



f.header

f.header.mass

gas_pos = f['POS '][0]

pl.plot(gas_pos[:,0], gas_pos[:,1], ',')
pl.plot(gas_pos[:,0], gas_pos[:,1], '.r') # the r is for red, the . is for dot
#pl.xlim(45,55)
#pl.ylim(45,55)
pl.xlabel('x')
pl.ylabel('y')
pl.title('first plot, latex also e.g. $\sim$ 2')

pl.show()



gas_dens = f['RHO '][0]
gas_u    = f['U   '][0] 
print(("The highest number entered was ", max(gas_dens), ".\n"))
print(("The lowest number entered was ", min(gas_dens), ".\n"))

print(("max(u)", max(gas_u), "\n"))
print(("min(u)", min(gas_u), "\n"))


#np.histogram(gas_dens, bins=100, range=None, normed=False, weights=None, density=None)


#pl.hist(gas_dens, bins=10, range=[min(gas_dens),1.e2])

#pl.plot(80*np.log(gas_u),np.log(gas_dens),'.')
##pl.xlim(min(np.log(gas_dens)),2)
##pl.ylim(1.e-4,6)
#pl.show()


quit()

