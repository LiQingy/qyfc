'''
this is a template to show the distribution of galaxies in the observation sky.
'''

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
from PIL import Image 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.coordinates import Angle


hdul   = fits.open('xxx.fits') 
image  = hdul[0].data
image = Image.open('xxx.jpeg') 
wcs2d  = WCS(hdul[0].header)

fig = plt.figure(figsize = (16, 16))
ax = fig.add_subplot(111, projection=wcs2d)
plt.imshow(image, origin='lower', cmap=plt.cm.viridis, norm = LogNorm() )

#show galaxies
xgal, ygal = wcs2d.wcs_world2pix(galra, galdec, 0)
ax.scatter(xgal, ygal, facecolor = 'None', edgecolor = 'b', s = 20) 

ax.legend()
ax.set_xlabel('RA')
ax.set_ylabel('Dec')

#set xlabel
ax.coords.grid('icrs',color = 'none')
ax.coords[0].set_axislabel('RA')
ax.coords[1].set_axislabel('Dec')
ax.coords[0].set_ticklabel(size="small")
ax.coords[1].set_ticklabel(size="small")

lon = ax.coords[0]
lon.set_major_formatter('d.ddd')
lat = ax.coords[1]
lat.set_major_formatter('d.ddd')
lat.set_ticks_position('l')
lon.set_ticks_position('b')
