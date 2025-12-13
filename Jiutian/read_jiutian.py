'''
A multiprocess version writen for extracting data from Jiutian simulation.  

read_groups: extract halo/subhalo informations of one snapshot from all subfind group files.

read_hbt: read subhalo information of one snapshot from all SubSnap files. 

Wirtten by Qingyang Li, Yizhou Gu (2022/11); revised in 2024/1
''' 

import glob
import multiprocessing as mp
import numpy as np
import time
import os
import h5py
import sys
import re

# __ALL__ = ['collect4csstmock', 'read_groups', 'read_hbt', 'nexthaloid']

def read_hbt_head(filename):
    '''
    Read some headers from hbt files. 

    Note: 

    Cosmology in all files are same at same snapshot.
    '''
    HEAD = {}
    data = h5py.File(filename,'r')
    HEAD['HubbleParam'] = data['Cosmology/HubbleParam'][0]
    HEAD['OmegaLambda0'] = data['Cosmology/OmegaLambda0'][0]
    HEAD['OmegaM0'] = data['Cosmology/OmegaM0'][0]
    HEAD['ScaleFactor'] = data['Cosmology/ScaleFactor'][0]
    HEAD['NumberOfSubhalosInAllFiles'] = data['NumberOfSubhalosInAllFiles'][0]
    HEAD['NumberOfFiles'] = data['NumberOfFiles'][0]

    #we supplyment information here
    HEAD['NumberOfSubhalosThisFile'] = data['Subhalos'].fields('TrackId')[:].shape[0]
    data.close()

    return HEAD


def read_hbt_subhalos(filename, blocks):
    '''
    read subhalo information from one SubSnap file

    Wirtten by Qingyang Li

    Parameters
    ----------
    filename: path of file
    blocks: name of blocks

    Returns
    -------
    Nsub: subhalo number
    DATA: dict, datasets of blocks
    DATATYPE: dict, data type of blocks 
    '''
    
    import numpy as np
    import h5py

    #all names of blocks
    names = ['TrackId', 'Nbound', 'Mbound', 'HostHaloId', 'Rank', 'Depth', 'LastMaxMass', 'SnapshotIndexOfLastMaxMass', 'SnapshotIndexOfLastIsolation', 'SnapshotIndexOfBirth', 'SnapshotIndexOfDeath', 'SnapshotIndexOfSink', 'RmaxComoving', 'VmaxPhysical', 'LastMaxVmaxPhysical', 'SnapshotIndexOfLastMaxVmax', 'R2SigmaComoving', 'RHalfComoving', 'BoundR200CritComoving', 'BoundM200Crit', 'SpecificSelfPotentialEnergy', 'SpecificSelfKineticEnergy', 'SpecificAngularMomentum', 'InertialEigenVector', 'InertialEigenVectorWeighted', 'InertialTensor', 'InertialTensorWeighted', 'ComovingAveragePosition', 'PhysicalAverageVelocity', 'ComovingMostBoundPosition', 'PhysicalMostBoundVelocity', 'MostBoundParticleId', 'SinkTrackId']
    types = ['<i8', '<i8', '<f4', '<i8', '<i8', '<i4', '<f4', '<i4', '<i4', '<i4', '<i4', '<i4', '<f4', '<f4', '<f4', '<i4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<i8', '<i8']

    data = h5py.File(filename,'r')
    nbound= data['Subhalos'].fields('Nbound')[:]
    idx = np.where(nbound > 1)[0]
    Nsub = idx.shape[0]
    
    DATA = {}
    DATATYPE = {}
    for block in blocks:
        readdata = data['Subhalos'].fields(block)[:]
        readdata = readdata[idx] #select Nbound != 1
        nblock = names.index(block)
        datatype = types[nblock]
        # datatype = data['Subhalos'].dtype[nblock]
        # datatype = data['Subhalos'].dtype.descr[nblock][1]
        # print('Read subhalo blocks: %s,' %block + ' dtype:', datatype)

        DATA[block] = np.array(readdata, dtype = datatype)
        DATATYPE[block] = datatype
    data.close()
    return Nsub, DATA, DATATYPE

def read_hbt_subhalos_split(filenames, blocks):
    '''
    The worker of read_hbt_subhalos for parallel 
    '''
    DATA = {}; DATATYPE = {} 
    for block in blocks:  DATA[block] = []
    
    Nsub = []; 
    for filename in filenames:
        #print(filename)
        Nsub_, DATA_, DATATYPE_ = read_hbt_subhalos(filename, blocks) 
        for block in blocks: DATA[block].append(DATA_[block])
        Nsub.extend([Nsub_])
    DATATYPE = DATATYPE_
    Nsub = np.sum(Nsub, dtype = np.int64)
    blockin3D = ['ComovingAveragePosition', 'PhysicalAverageVelocity', 'ComovingMostBoundPosition', 'PhysicalMostBoundVelocity'] 
    for block in blocks: 
        if block in blockin3D: 
            DATA[block] = np.vstack(DATA[block])
        else: 
            DATA[block] = np.hstack(DATA[block])
    return Nsub, DATA, DATATYPE

def read_hbt(snapnum, blocks, basedir_hbt = '/home/cossim/Jiutian/M1000/hbt/'): 
    '''
    read subhalo information from all SubSnap file with multiprocessing
    Wirtten by Qingyang Li

    Parameters: 
    ----------
    snapnum: int
          the number of snapshot
    blocks:  list
          the name of blocks
    basedir_groups: str 
          file path for subfind data (FoF)

    Returns:
    -------
    DATAALL: dict  
          datasets of blocks. for example, Nbound = DATAALL['Nbound'][:]
    DATATYPE: dict 
          data type of blocks


    Note also
    ---------
    Subhalos with Nbound<=1 are not included. 
    ''' 
    
    #initialize data
    DATAALL = {}
    for block in blocks: DATAALL[block] = []
    filedir   = basedir_hbt + str(snapnum).zfill(3) + \
                '/SubSnap_'+ str(snapnum).zfill(3)  + '.'
    filenames = glob.glob(filedir + '*.hdf5') 
    division  = len(filenames) # the divided  number
    filenames = [filedir + '%s.hdf5'%d for d in range(division) ] 
    
    size        = mp.cpu_count() 
    if division == 0:
        print('No found in ' + filedir)
        exit()
    else:
        print('Reading %s hbt divisions of %sth snapshot with %s threads'%(division, snapnum, size), ) 

    #split task & start parallel
    task_splits = np.array_split( filenames, size)
    pool = mp.Pool(size) 
    res  = []
    for ii in range(size):
        r = pool.apply_async(read_hbt_subhalos_split, args = (task_splits[ii], blocks) ) 
        res.append(r)
    pool.close() 
    pool.join()
    NSUB = []
    for r in res:
        Nsub_, DATA, DATATYPE = r.get() 
        for block in blocks: DATAALL[block].append(DATA[block])
        NSUB.append(Nsub_) 
    NSUB = np.sum(NSUB, dtype = np.int64)
    blockin3D = ['ComovingAveragePosition', 'PhysicalAverageVelocity', 'ComovingMostBoundPosition', 'PhysicalMostBoundVelocity'] 
    for block in blocks: 
        if block in blockin3D: 
            DATAALL[block] = np.vstack(DATAALL[block])
        else: 
            DATAALL[block] = np.hstack(DATAALL[block]) 
    return DATAALL, DATATYPE

class subfind_catalog:
  '''
  code for reading Subfind's subhalo_tab files
  '''
  def __init__(self, curfile): 
 
    #self.filebase = basedir + "/groups_" + str(snapnum).zfill(3) + "/subhalo_tab_" + str(snapnum).zfill(3) + "."
 
    #print()
    #print("reading subfind catalog for snapshot",snapnum,"of",basedir)
 
    have_veldisp = True
 
    #curfile = self.filebase + str(filenum)
    
    if (not os.path.exists(curfile)):
      print("file not found:", curfile)
      sys.exit()
    
    f = open(curfile,'rb')
    
    ngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]    # Number of groups within this file
    totngroups = np.fromfile(f, dtype=np.uint64, count=1)[0] # Total number of groups for this snapshot.
    nids = np.fromfile(f, dtype=np.uint32, count=1)[0]       
    totnids = np.fromfile(f, dtype=np.uint64, count=1)[0]
    ntask   = np.fromfile(f, dtype=np.uint32, count=1)[0]
    nsubs   = np.fromfile(f, dtype=np.uint32, count=1)[0]
    totnsubs = np.fromfile(f, dtype=np.uint64, count=1)[0]
    
    self.ngroups= ngroups # Number of    groups within this file chunk.
    self.nids   = nids  # ???? print(self.nids)  
    self.nfiles = ntask # Total number of file chunks the group catalog is split between.
    self.nsubs  = nsubs # Number of subgroups within this file chunk.

    self.totngrp = totngroups # Total number of    groups for this snapshot.
    self.totnsub = totnsubs   # Total number of subgroups for this snapshot.


    self.group_len = np.empty(ngroups, dtype=np.uint32)
    self.group_offset = np.empty(ngroups, dtype=np.uint32)
    self.group_nr = np.empty(ngroups, dtype=np.uint64)
    self.group_cm = np.empty(ngroups, dtype=np.dtype((np.float32,3)))
    self.group_vel = np.empty(ngroups, dtype=np.dtype((np.float32,3)))
    self.group_pos = np.empty(ngroups, dtype=np.dtype((np.float32,3)))
    self.group_m_mean200 = np.empty(ngroups, dtype=np.float32)
    self.group_m_crit200 = np.empty(ngroups, dtype=np.float32)
    self.group_m_tophat200 = np.empty(ngroups, dtype=np.float32)
    self.group_veldisp = np.empty(ngroups, dtype=np.float32)
    if have_veldisp:
      self.group_veldisp_mean200 = np.empty(ngroups, dtype=np.float32)
      self.group_veldisp_crit200 = np.empty(ngroups, dtype=np.float32)
      self.group_veldisp_tophat200 = np.empty(ngroups, dtype=np.float32)
    self.group_nsubs = np.empty(ngroups, dtype=np.uint32)
    self.group_firstsub = np.empty(ngroups, dtype=np.uint32)
    
    if nsubs > 0: 
      self.sub_len = np.empty(nsubs, dtype=np.uint32)
      self.sub_offset = np.empty(nsubs, dtype=np.uint32)
      self.sub_grnr = np.empty(nsubs, dtype=np.uint64)
      self.sub_nr = np.empty(nsubs, dtype=np.uint64)
      self.sub_pos = np.empty(nsubs, dtype=np.dtype((np.float32,3)))
      self.sub_vel = np.empty(nsubs, dtype=np.dtype((np.float32,3)))
      self.sub_cm = np.empty(nsubs, dtype=np.dtype((np.float32,3)))
      self.sub_spin = np.empty(nsubs, dtype=np.dtype((np.float32,3)))
      self.sub_veldisp = np.empty(nsubs, dtype=np.float32)
      self.sub_vmax = np.empty(nsubs, dtype=np.float32)
      self.sub_vmaxrad = np.empty(nsubs, dtype=np.float32)
      self.sub_halfmassrad = np.empty(nsubs, dtype=np.float32)
      #self.sub_shape = np.empty(nsubs, dtype=np.dtype((np.float32,6)))
      self.sub_ebind = np.empty(nsubs, dtype=np.float32)
      self.sub_pot = np.empty(nsubs, dtype=np.float32)
      #self.sub_profile = np.empty(nsubs, dtype=np.dtype((np.float32,9)))
      self.sub_parent = np.empty(nsubs, dtype=np.uint32)
      self.sub_idbm = np.empty(nsubs, dtype=np.uint64)
    #--------------------------------------------------------------------
    self.group_len = np.fromfile(f, dtype=np.uint32, count=ngroups)
    # group_len - Integer counter of the total number of particles/cells of all types in this group.
    self.group_offset = np.fromfile(f, dtype=np.uint32, count=ngroups)
    self.group_nr = np.fromfile(f, dtype=np.uint64, count=ngroups)

    self.group_cm = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=ngroups)
    self.group_vel = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=ngroups)
    self.group_pos = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=ngroups)
    # group_cm  (N,3) - Center of mass of the group (mass weighted relative coordinates of all particles/cells in the group)
    # group_vel (N,3) - Velocity of the group, The peculiar velocity is obtained by multiplying this value by 1/a. 
    # group_pos (N,3) - Spatial position within the periodic box (of the particle with the minimum gravitational potential energy)
    self.group_m_mean200 = np.fromfile(f, dtype=np.float32, count=ngroups)
    self.group_m_crit200 = np.fromfile(f, dtype=np.float32, count=ngroups)
    self.group_m_tophat200 = np.fromfile(f, dtype=np.float32, count=ngroups)
    self.group_veldisp = np.fromfile(f, dtype=np.float32, count=ngroups)

    if have_veldisp:
      self.group_veldisp_mean200 = np.fromfile(f, dtype=np.float32, count=ngroups)
      self.group_veldisp_crit200 = np.fromfile(f, dtype=np.float32, count=ngroups)
      self.group_veldisp_tophat200 = np.fromfile(f, dtype=np.float32, count=ngroups)

    self.group_nsubs = np.fromfile(f, dtype=np.uint32, count=ngroups)
    self.group_firstsub = np.fromfile(f, dtype=np.uint32, count=ngroups)

    if nsubs > 0:
      self.sub_len = np.fromfile(f, dtype=np.uint32, count=nsubs)
      self.sub_offset = np.fromfile(f, dtype=np.uint32, count=nsubs)
      self.sub_grnr = np.fromfile(f, dtype=np.uint64, count=nsubs)
      self.sub_nr = np.fromfile(f, dtype=np.uint64, count=nsubs)
      self.sub_pos = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=nsubs)
      self.sub_vel = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=nsubs)
      self.sub_cm = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=nsubs)
      # group_cm  (N,3) - Center of mass of the subgroup
      # group_vel (N,3) - Velocity of the subgroup 
      # group_pos (N,3) - Spatial position within the periodic box 
      self.sub_spin = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=nsubs)
      self.sub_veldisp = np.fromfile(f, dtype=np.float32, count=nsubs)
      # sub_veldisp N   - One-dimensional velocity dispersion of all the member particles/cells (the 3D dispersion divided by sqrt(3) ).
      self.sub_vmax = np.fromfile(f, dtype=np.float32, count=nsubs)
      self.sub_vmaxrad = np.fromfile(f, dtype=np.float32, count=nsubs)
      self.sub_halfmassrad = np.fromfile(f, dtype=np.float32, count=nsubs)
      #self.sub_shape = np.fromfile(f, dtype=np.dtype((np.float32,6)), count=nsubs)
      self.sub_ebind = np.fromfile(f, dtype=np.float32, count=nsubs)
      self.sub_pot = np.fromfile(f, dtype=np.float32, count=nsubs)
      #self.sub_profile = np.fromfile(f, dtype=np.dtype((np.float32,9)), count=nsubs)
      self.sub_parent = np.fromfile(f, dtype=np.uint32, count=nsubs)
      self.sub_idbm = np.fromfile(f, dtype=np.uint64, count=nsubs)

      curpos = f.tell()
      f.seek(0,os.SEEK_END)
      if curpos != f.tell():
        print("Warning: finished reading before EOF for file",filenum)
      f.close()  


class subfind_catalog_hdf5:
  '''
  code for reading Subfind's subhalo_tab files
  '''
  def __init__(self, curfile): 
 
    #self.filebase = basedir + "/groups_" + str(snapnum).zfill(3) + "/subhalo_tab_" + str(snapnum).zfill(3) + "."
 
    #print()
    #print("reading subfind catalog for snapshot",snapnum,"of",basedir)
 
    have_veldisp = False
 
    #curfile = self.filebase + str(filenum)
    
    if (not os.path.exists(curfile)):
      print("file not found:", curfile)
      sys.exit()
    
    f = h5py.File(curfile,'r')

    self.ngroups= len(f['Group/GroupPos'][:])  # Number of    groups within this file chunk.
    self.nsubs  = len(f['Subhalo/SubhaloPos'][:]) # Number of subgroups within this file chunk.

    #--------------------------------------------------------------------
    self.group_len = f['Group/GroupLen'][:]
    # group_len - Integer counter of the total number of particles/cells of all types in this group.
    self.group_offsettype = f['Group/GroupOffsetType'][:]
    self.group_mass = f['Group/GroupMass'][:]

    self.group_vel = f['Group/GroupVel'][:]
    self.group_pos = f['Group/GroupPos'][:]
    # group_vel (N,3) - Velocity of the group, The peculiar velocity is obtained by multiplying this value by 1/a. 
    # group_pos (N,3) - Spatial position within the periodic box (of the particle with the minimum gravitational potential energy)
    self.group_m_mean200 = f['Group/Group_M_Mean200'][:]
    self.group_m_crit200 = f['Group/Group_M_Crit200'][:]
    self.group_m_tophat200 = f['Group/Group_M_TopHat200'][:]

    self.group_r_mean200 = f['Group/Group_R_Mean200'][:]
    self.group_r_crit200 = f['Group/Group_R_Crit200'][:]
    self.group_r_tophat200 = f['Group/Group_R_TopHat200'][:]

    self.group_nsubs = f['Group/GroupNsubs'][:]
    self.group_firstsub = f['Group/GroupFirstSub'][:]

    if self.nsubs > 0:
      self.sub_len = f['Subhalo/SubhaloLen'][:]
      self.sub_pos = f['Subhalo/SubhaloPos'][:]
      self.sub_grnr = f['Subhalo/SubhaloGroupNr'][:]
      self.sub_vel = f['Subhalo/SubhaloVel'][:]
      self.sub_cm = f['Subhalo/SubhaloCM'][:]
      # group_cm  (N,3) - Center of mass of the subgroup
      # group_vel (N,3) - Velocity of the subgroup 
      # group_pos (N,3) - Spatial position within the periodic box 
      self.sub_spin = f['Subhalo/SubhaloSpin'][:]
      self.sub_veldisp = f['Subhalo/SubhaloVelDisp'][:]
      # sub_veldisp N   - One-dimensional velocity dispersion of all the member particles/cells (the 3D dispersion divided by sqrt(3) ).
      self.sub_vmax = f['Subhalo/SubhaloVmax'][:]
      self.sub_vmaxrad = f['Subhalo/SubhaloVmaxRad'][:]
      self.sub_halfmassrad = f['Subhalo/SubhaloHalfmassRad'][:]
      self.sub_parent = f['Subhalo/SubhaloParentRank'][:]
      # self.sub_idbm = f['Subhalo/SubhaloIDMostbound'][:]
      self.sub_mass = f['Subhalo/SubhaloMass'][:]
      #self.sub_shape = np.fromfile(f, dtype=np.dtype((np.float32,6)), count=nsubs)
      f.close() 

    
def read_groups_subhalos(filename, blocks):
    if 'M1000' in filename:
        dmpm = 0.03722953 #dark matter particle mass
    elif 'M300' in filename:
        dmpm = 0.001005
    elif 'M2G' in filename:
        dmpm = 0.2978

    DATAALL = {}
    DATATYPE = {}
    for block in blocks: 
        DATAALL[block] = []
        DATATYPE[block] = []
    if '.hdf5' in filename:
        grp_= subfind_catalog_hdf5(filename) 
        nh_ = grp_.ngroups
        ns_ = grp_.nsubs
    else:
        grp_= subfind_catalog(filename) 
        nh_ = np.shape(grp_.group_nr)[0] #number of groups
        ns_ = np.shape(grp_.sub_nr)[0] #number of subhalos 
    #group information
    for block in blocks:
        if block == 'group_mass':
                DATAALL[block]  = np.log10(grp_.group_len*dmpm)+10
                DATATYPE[block] = 'float32'
        elif block == 'sub_mass':
                DATAALL[block]  = np.log10(grp_.sub_len*dmpm)+10 
                DATATYPE[block] = 'float32'
        else:
                DATAALL[block]  = getattr(grp_, block) 
                DATATYPE[block] = getattr(grp_, block).dtype
    return nh_, ns_, DATAALL, DATATYPE

def read_groups_subhalos_split(filenames, blocks):
    '''
    The worker of read_groups_subhalos for parallel 
    '''
    DATA = {}; DATATYPE = {} 
    for block in blocks:  DATA[block] = []
    
    Ns = []; Nh = [] 
    for filename in filenames:
        Nh_, Ns_, DATA_, DATATYPE_ = read_groups_subhalos(filename, blocks) 
        for block in blocks: DATA[block].append(DATA_[block])
        Nh.extend([Nh_])
        Ns.extend([Ns_])
    Nh = np.sum(Nh, dtype = np.int64)
    Ns = np.sum(Ns, dtype = np.int64)
    DATATYPE = DATATYPE_
    blockin3D = ['group_cm', 'group_vel', 'group_pos', 'sub_pos', 'sub_vel', 'sub_cm', 'sub_spin']
    for block in blocks: 
        if block in blockin3D: 
            DATA[block] = np.vstack(DATA[block])
        else: 
            DATA[block] = np.hstack(DATA[block])
    return Nh, Ns, DATA, DATATYPE

def file_block_check(fullfilename):
    '''
    some sub files do not have 'Group', especially for M300
    '''

    filenms = []
    for filename in fullfilename:
        f = h5py.File(filename, 'r')
        if 'Group' in f.keys():
            filenms.append(filename)
        f.close()
    return np.array(filenms)

def read_groups(snapnum, blocks, basedir_groups = '/home/cossim/Jiutian/M1000/groups/'): 
    '''
    extract halo/subhalo informations from all subfind group files with multiprocessing 

    Wirtten by Qingyang Li, Yizhou Gu

    Parameters: 
    ----------
    snapnum: int
          the number of snapshot
    blocks:  list
          the name of blocks
    basedir_groups: str 
          file path for subfind data (FoF)

    Returns:
    -------
    DATAALL: dict  
          datasets of blocks. For example, group_pos = DATAALL['group_pos'][:]
    DATATYPE: dict 
          data type of blocks
    
    Note also: 
    ---------
    Host halo and subhalo information can be obtained together.
    '''
        
    #initialize data
    DATAALL = {}
    for block in blocks: DATAALL[block] = []
    filedir   = basedir_groups + '/groups_'+  str(snapnum).zfill(3) + \
                '/*subhalo_tab_'+ str(snapnum).zfill(3)  + '.'
    filenames = glob.glob(filedir + '*') 

        #file check for M300
    if 'M300' in basedir_groups:
        filenames = file_block_check(filenames)
        print(filenames)

    division  = len(filenames) # the divided  number
    # filenames = [filedir + '%s'%d for d in range(division) ] 
    
    #filenames sorted with numbers
    natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]
    filenames = sorted(filenames, key = natsort)

    size        = mp.cpu_count() 
    
    if division == 0:
        print('No found in ' + filedir)
        exit()
    else:
        print('Reading %s groups divisions of %sth snapshot with %s threads'%(division, snapnum, size)) 
        
    if ('.hdf5' in filenames[0]) and ('group_nr' in blocks):
        blocks_temp = blocks * 1
        blocks.remove('group_nr')
        

    #split task & start parallel
    task_splits = np.array_split(filenames, size)
    pool = mp.Pool(size) 
    res  = []
    for ii in range(size):
        r = pool.apply_async(read_groups_subhalos_split, args = (task_splits[ii], blocks) ) 
        res.append(r)
    pool.close() 
    pool.join()

    NSUB = []
    for r in res:
        Nh_, Ns_, DATA, DATATYPE = r.get() 
        for block in blocks: DATAALL[block].append(DATA[block])
        
    blockin3D = ['group_cm', 'group_vel', 'group_pos', 'sub_pos', 'sub_vel', 'sub_cm', 'sub_spin']
    for block in blocks: 
        if block in blockin3D: 
            DATAALL[block] = np.vstack(DATAALL[block])
        else: 
            DATAALL[block] = np.hstack(DATAALL[block])
            
    if ('.hdf5' in filenames[0]) and ('group_nr' in blocks) and ('M2G' in basedir_groups):
        DATAALL['group_nr'] = np.arange(396976641, dtype = np.int64) #for M2G
        DATATYPE['group_nr'] = 'np.int64'
    return DATAALL, DATATYPE


