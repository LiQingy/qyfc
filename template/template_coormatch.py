'''match coordinate according sky coordinate
'''

def match_coordinate(ra1, dec1, ra2, dec2, sep = 2.0):
  from astropy.coordinates import SkyCoord
  from astropy import units as u
  from astropy.coordinates import search_around_sky
  
  coords1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
  coords2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
  
  max_sep = sep * u.arcsec  #set max radius for matching
  
  idx_sample1, idx_sample2, d2d, _ = search_around_sky(coords1, coords2, seplimit=max_sep)
  return idx_sample1, idx_sample2, d2d
