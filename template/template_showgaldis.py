'''show galaxy distribution in the Sky with healpy
'''

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pandas import value_counts


ra = t['RA'][:]
dec = t['DEC'][:]

nside = 64  # NSIDE 越大，分辨率越高
npix = hp.nside2npix(nside)  # 计算总像素数

ra = np.deg2rad(ra)
dec = np.pi/2.-np.deg2rad(dec)
pxl = hp.ang2pix(nside, dec, ra)

pxlc = value_counts(pxl)
colorpix = np.zeros(npix)
colorpix[pxlc.index] = pxlc.values
colorpix[colorpix==0] = np.nan

plt.figure()

#hp.cartview()
hp.mollview(colorpix, 
            title='Galaxy Number Counts',
            rot=(270.,0.,0.), 
            bgcolor = 'white',
            xsize =2000,
            badcolor = 'white',
            min = 0,
            max=np.percentile(colorpix[colorpix>0], 95))


hp.graticule(dpar=30, coord='C')

# 添加RA坐标标签（在赤道Dec=0的位置）
for ra_label in np.arange(0, 360, 30):  # 每30°标注一个RA值
    hp.projtext(
        ra_label, 0,  # 在(RA, Dec)=(ra_label, 0)的位置添加文本
        f"{ra_label}°",  # 显示的文本
        lonlat=True,  # 输入是经纬度
        coord="C",  # 赤道坐标系
        color='black',  # 文本颜色
        fontsize=10,  # 字体大小
        va='center', ha='center'  # 文本对齐方式
    )
for dec_label in np.arange(-60, 90, 30):  # 从-60°到+60°，每30°标注
    hp.projtext(
        0, dec_label,  # 在(RA, Dec)=(0, dec_label)的位置添加文本
        f"{dec_label}°",
        lonlat=True,
        coord="C",
        color='black',
        fontsize=10,
        va='center', ha='center'
    )

#绘制银河系平面（红色粗线）
l = np.linspace(0, 360, 1000)  # 银经 0° 到 360°
b = np.zeros(1000)              # 银纬 0°（银河系平面）

rot = hp.Rotator(coord=['G', 'C'])  # 从银河系坐标转到赤道坐标
theta_gal = np.pi/2. - np.deg2rad(b)  # HEALPix θ ∈ [0, π]
phi_gal = np.deg2rad(l)               # φ ∈ [0, 2π]
theta_eq, phi_eq = rot(theta_gal, phi_gal)  # 转换为赤道坐标
dec_galplane = np.rad2deg(np.pi/2. - theta_eq)  # 转回 Dec
ra_galplane = np.rad2deg(phi_eq) % 360          # 转回 RA (0-360°)

hp.projplot(
    ra_galplane,
    dec_galplane,
    'r-',
    lonlat=True,
    coord="C",  # 在赤道坐标系下绘制
    linewidth=3  # 加粗线条
)

# from astropy.coordinates import SkyCoord, ICRS
# import astropy.units as u
# galactic_plane_color = 'r'
# ax = plt.subplot(111, projection=projection)
# if galactic_plane_color is not None:
#     galactic_l = np.linspace(0, 2 * np.pi, 1000)
#     galactic = SkyCoord(l=galactic_l*u.radian, b=np.zeros_like(galactic_l)*u.radian,
#                         frame='galactic').transform_to(ICRS)
#     #
#     # Project to map coordinates and display.  Use a scatter plot to
#     # avoid wrap-around complications.
#     #
#     paths = ax.scatter(projection_ra(0, galactic.ra.degree),
#                        projection_dec(0, galactic.dec.degree),
#                        marker='.', s=20, lw=0, alpha=0.75,
#                        c=galactic_plane_color, zorder=20)

plt.show()

totalpixel = hp.nside2npix(nside)
gal_area = pxlc.shape[0] / totalpixel * (4*180**2/np.pi)
print("the cover of galaxy region is %s dge2" %gal_area)
