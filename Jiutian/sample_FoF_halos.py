import numpy as np 
import h5py
import read_jiutian

snapnum = 106
basedir_groups = '/home/cossim/Jiutian/M1000/groups'
# blocks = ['group_nr', 'group_mass', 'group_pos', 'group_vel', 'group_m_mean200', 'group_m_crit200', 'sub_grnr', 'sub_nr', 'sub_pos', 'sub_vel', 'sub_mass']
blocks = ['group_mass', 'group_pos', 'group_vel']
# DATACOLLECT_groups, DATATYPE_groups = read_groups.read_groups_allfiles_mpi(basedir_groups, snapnum, blocks)

DATACOLLECT_groups, DATATYPE_groups = read_jiutian.read_groups(snapnum, blocks, basedir_groups)

filename = None
d5 = None
block = None; ni = None
pos_group = None; pos_subs = None
idx_group = None; idx_subs = None

filename = '../data/sample_fof_M1000_snapshot%s_halos.hdf5'%snapnum
d5 = h5py.File(filename, 'w')
pos_group = DATACOLLECT_groups['group_pos'][:]
# pos_subs = DATACOLLECT_groups['sub_pos'][:]
# idx_group = np.where((pos_group[:,0] > 0) & (pos_group[:,0] < 300)& 
#                     (pos_group[:,1] > 0) & (pos_group[:,1] < 300)& 
#                     (pos_group[:,2] > 0) & (pos_group[:,2] < 300))[0]
# idx_subs = np.where((pos_subs[:,0] > 100) & (pos_subs[:,0] < 300)& 
#     (pos_subs[:,1] > 100) & (pos_subs[:,1] < 300)& 
#     (pos_subs[:,2] > 100) & (pos_subs[:,2] < 300))[0]

for ni,block in enumerate(blocks):
    print('saving block: ', block)
    if block in ['group_nr', 'group_mass', 'group_pos', 'group_vel', 'group_m_mean200', 'group_m_crit200']:
        d5.create_dataset(block, data = DATACOLLECT_groups[block][idx_group], dtype = DATATYPE_groups[block])
    else:
        d5.create_dataset(block, data = DATACOLLECT_groups[block][idx_group], dtype = DATATYPE_groups[block])
d5.close()





