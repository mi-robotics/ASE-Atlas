from isaacgym.torch_utils import *
import torch
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.visualization.plot_simple import SimplePlotter



import os

a1_inv_dir = './data/a1_v3/inv'
a1_dir = './data/a1_v3'

# mocap_inv_dir = './data/a1_v3/inv'
# mocap_dir = './data/dog_mocap'

a1_files = os.listdir(a1_dir)
a1_inv_files = os.listdir(a1_inv_dir)
# mocap_files = os.listdir(mocap_dir)
# mocap_inv_files = os.listdir(mocap_inv_dir)

# a1 (50FPS)
for f_name in a1_files:
    name = f_name.split('.npz')[0]
    f_path = a1_dir + '/' + f_name

    if os.path.isfile(f_path):
        try:
            data = np.load(f_path)
            pos = torch.Tensor(data['pos'])
            rot = torch.Tensor(data['rot'])
            
            if len(pos) > 50:

                a1_skeleton = SkeletonState.from_file('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/a1_tpose_v2.npy').skeleton_tree

                a1_state = SkeletonState.from_rotation_and_root_translation(
                            a1_skeleton, r=rot, t=pos, is_local=True
                        )
                motion = SkeletonMotion.from_skeleton_state(a1_state, fps=50)
                # plot_skeleton_motion_interactive(motion)

                motion.to_file(f'data/a1_v3_processed/{name}.npy')
        except:
            print('error')
            print(len(pos))

            input()


#a1 inv (50FPS)
for f_name in a1_inv_files:
    name = f_name.split('.npz')[0]
    f_path = a1_inv_dir + '/' + f_name

    if os.path.isfile(f_path):
        try:
            data = np.load(f_path)
            pos = torch.Tensor(data['pos'])
            rot = torch.Tensor(data['rot'])

       
            if len(pos) > 50:
                a1_skeleton = SkeletonState.from_file('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/a1_tpose_v2.npy').skeleton_tree

                a1_state = SkeletonState.from_rotation_and_root_translation(
                            a1_skeleton, r=rot, t=pos, is_local=True
                        )
                motion = SkeletonMotion.from_skeleton_state(a1_state, fps=50)

                # plot_skeleton_motion_interactive(motion)

                motion.to_file(f'data/a1_v3_processed/inv/{name}.npy')

        except:
            print('error')
            print(len(pos))
            input()

# input()
# #mocap 
# for f_name in mocap_files:
#     name = f_name.split('.npz')[0]
#     f_path = mocap_dir + '/' + f_name

#     if os.path.isfile(f_path):
#         try:
#             data = np.load(f_path)
#             pos = torch.Tensor(data['pos'])
#             rot = torch.Tensor(data['rot'])

#             a1_skeleton = SkeletonState.from_file('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/a1_tpose_v2.npy').skeleton_tree

#             a1_state = SkeletonState.from_rotation_and_root_translation(
#                         a1_skeleton, r=rot, t=pos, is_local=True
#                     )
#             motion = SkeletonMotion.from_skeleton_state(a1_state, fps=60)

#             motion.to_file(f'data/dog_mocap_processed/{name}.npy')
#         except:
#             print('error')
#             print(len(pos))
#             input()
        
# input()
# #mocap inv
# for f_name in mocap_inv_files:
#     name = f_name.split('.npz')[0]
#     f_path = mocap_inv_dir + '/' + f_name

#     if os.path.isfile(f_path):
#         try:
#             data = np.load(f_path)
#             pos = torch.Tensor(data['pos'])
#             rot = torch.Tensor(data['rot'])

#             a1_skeleton = SkeletonState.from_file('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/a1_tpose_v2.npy').skeleton_tree

#             a1_state = SkeletonState.from_rotation_and_root_translation(
#                         a1_skeleton, r=rot, t=pos, is_local=True
#                     )
#             motion = SkeletonMotion.from_skeleton_state(a1_state, fps=60)

#             motion.to_file(f'data/dog_mocap_processed/inv/{name}.npy')
#         except:
#             print('error')
#             print(len(pos))
#             input()

# # Load the data
# data = np.load(f'./data/a1_recording/1704478684.7267365.npz')

# # Access the arrays
# positions = torch.Tensor(data['pos'])
# rotations = torch.Tensor(data['rot'])

# a1_skeleton = SkeletonState.from_file('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/a1_tpose_v2.npy').skeleton_tree

# a1_state = SkeletonState.from_rotation_and_root_translation(
#             a1_skeleton, r=rotations, t=positions, is_local=True
#         )
# motion = SkeletonMotion.from_skeleton_state(a1_state, fps=60)

# # plot_skeleton_motion_interactive(motion)
# motion.to_file('data/a1_recording_processed/demo_recording.npy')