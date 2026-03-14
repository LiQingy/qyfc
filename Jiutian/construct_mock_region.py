'''
construct mock region covered by dark matter particles 
Written Qingyang Li, 2025.11.26
'''

import numpy as np
import pandas as pd
import h5py

import sys
sys.path.append("/home/qyli/work2025/densityfield/pipeline")
import sampling, gridsmooth, fouriertransf, exttool, phyconvert, exttool, surveyvol
import time


def get_redshift_edge(redzbin = [0.0, 0.4]):
    '''get redshift edge and return snapshot number
    '''
    with h5py.File('/home/qyli/Jiutian/mock/mockDESIDR9_9tian0_z0.0_1.0Mo_info.hdf5', 'r') as fw5:
        #1. new halo id; 2. host halo id; 3, galaxy index in mock catalog
        dhalo = fw5['halo_prop'][:]
        
    #mock velocity grid
    filemock = '/home/qyli/work2025/densityfield/project/data/halovel_9tianmock_NGC_z0.0-1.0.hdf5' 
    dvel = h5py.File(filemock, 'r')
    halonewid = np.int64(dvel['newhaloid'][:])
    halo_vel = dvel['halovel'][:]
    halomass = 10**dvel['halomass'][:]
    halosp = dvel['snapshotnum'][:]
    newhalo_ra = dhalo[halonewid-1,2]; newhalo_dec = dhalo[halonewid-1,3]; newhalo_z = dhalo[halonewid-1,5]; newhalo_obsz = dhalo[halonewid-1,4]

    idx_ = (newhalo_z >= redzbin[0]) & (newhalo_z <= redzbin[1]) #redshift control
    sps = halosp[idx_]
    z0 = newhalo_obsz[idx_] #observation redshift

    unqval = np.flip(np.unique(sps))
    zmax = np.zeros(len(unqval))
    zmin = np.zeros(len(unqval))
    for ii,val in enumerate(unqval):
        if ii == 0:
            zmin[ii] = redzbin[0]
            zmax[ii] = np.max(z0[sps == val]) 
        else:
            zmin[ii] = zmax[ii-1]
            zmax[ii] = np.max(z0[sps == val]) 

    spshell_prop = {
        'snapshotnum': unqval,
        'zmax': zmax,
        'zmin': zmin
    }
    return spshell_prop

def tile_27(pos, vel, Lbox, zmin, zmax):
    shifts = np.array([-Lbox, 0, Lbox])
    tiled_pos = []
    tiled_vel = []

    for dx in shifts:
        for dy in shifts:
            for dz in shifts:
                pos0 = pos + np.array([dx, dy, dz])
                ra, dec, z = phyconvert.cartesian_to_equatorial(pos0, observer=(500, 500, 500))
                zboxmax = np.max(z)
                zboxmin = np.min(z)
                if zboxmax < zmin or zboxmin > zmax:
                    continue
                else:
                    idx_ = (ra > 92) & (ra < 281) & (dec > -10) & (dec < 86) & (z <= zmax) & (z > zmin)
                    if np.sum(idx_):
                        pos0 = pos0[idx_] - np.array([500,500,500])
                        tiled_pos.append(pos0)
                        tiled_vel.append(vel[idx_]) 
    tiled_pos = np.concatenate(tiled_pos, axis=0, dtype = np.float32) 
    tiled_vel = np.concatenate(tiled_vel, axis=0, dtype = np.float32) 
    return tiled_pos, tiled_vel

def tile_125(pos, vel, Lbox, zmin, zmax):
    shifts = np.array([-2*Lbox, -Lbox, 0, Lbox, 2*Lbox])
    tiled_pos = []
    tiled_vel = []

    for dx in shifts:
        for dy in shifts:
            for dz in shifts:
                pos0 = pos + np.array([dx, dy, dz])
                ra, dec, z = phyconvert.cartesian_to_equatorial(pos0, observer=(500, 500, 500))
                zboxmax = np.max(z)
                zboxmin = np.min(z)
                if zboxmax < zmin or zboxmin > zmax:
                    continue
                else:
                    idx_ = (ra > 92) & (ra < 281) & (dec > -10) & (dec < 86) & (z <= zmax) & (z > zmin)
                    if np.sum(idx_):
                        pos0 = pos0[idx_] - np.array([500,500,500])
                        tiled_pos.append(pos0)
                        tiled_vel.append(vel[idx_]) 
    tiled_pos = np.concatenate(tiled_pos, axis=0, dtype = np.float32) 
    tiled_vel = np.concatenate(tiled_vel, axis=0, dtype = np.float32) 
    return tiled_pos, tiled_vel

def get_pts(spshell_prop, spnum):
    '''get particles in a periodic box
    '''
    spnums = spshell_prop["snapshotnum"][:]
    zmin = spshell_prop["zmin"][:]
    zmax = spshell_prop["zmax"][:]
    
    totpos = []
    totvel = []
    start_time = time.time()

    idx_ = spnums == spnum
    zmin = zmin[idx_]
    zmax = zmax[idx_]

    print("reading POS at ", spnum)
    fposname = '/home/cossim/qyli/Jiutian/M1000/snapshot/sample_M1000_snapshot%03d_POS_rd0.01.hdf5'%spnum
    dpos = h5py.File(fposname, "r")
    ptpos = np.float32(dpos['POS'][:])
    dpos.close()
    fvelname = '/home/cossim/qyli/Jiutian/M1000/snapshot/sample_M1000_snapshot%03d_VEL_rd0.01.hdf5'%spnum
    dvel = h5py.File(fvelname, "r")
    ptvel = np.float32(dvel['VEL'][:])
    dvel.close()

    print("building box for snapshot... ", spnum)
    if spnum >= 95:
        posbox, velbox = tile_27(ptpos, ptvel, 1000, zmin, zmax)
    else:
        posbox, velbox = tile_125(ptpos, ptvel, 1000, zmin, zmax)

    print("Save data......")
    fvelname = '/home/qyli/Jiutian/snapshot/mockregion/mockregion_M1000_rd0.01_sp%03d.hdf5'%spnum
    dh5 = h5py.File(fvelname, "w")
    dh5['POS'] = posbox
    dh5['VEL'] = velbox
    dh5.close()
    
    end_time = time.time()
    print("used time", (end_time - start_time)/60)

def main(spnum):
    spshell_prop = get_redshift_edge(redzbin = [0.0, 1.0])
    get_pts(spshell_prop, spnum)

if __name__ == "__main__":
    spnum = 85
    main(spnum)


    # spshell_prop = get_redshift_edge(redzbin = [0.0, 1.0])
    # dd = np.c_[spshell_prop['snapshotnum'][:], spshell_prop['zmin'][:], spshell_prop['zmax'][:]]
    # np.savetxt('/home/qyli/Jiutian/snapshot/mockregion/zlist', dd, fmt = '%d   %.8f  %.8f', header = '1. snapshot; 2. zmin; 3. zmax')

     
