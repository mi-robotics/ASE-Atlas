from isaacgym.torch_utils import *
import torch
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.visualization.plot_simple import SimplePlotter

# rotations = torch.Tensor(np.load('./rotations-v6.npy'))
# positions = torch.Tensor(np.load('./positions-v6.npy'))

# Load the data
data = np.load(f'./data/a1_recording/1704478684.7267365.npz')

# Access the arrays
positions = torch.Tensor(data['pos'])
rotations = torch.Tensor(data['rot'])

toes = torch.Tensor([[0., 0., 0., 1.]]).unsqueeze(0).repeat(rotations.size(0),1,1)

r0 = rotations[:, :4, :]
r1 = rotations[:, 4:7, :]
r2 = rotations[:, 7:10, :]
r3 = rotations[:, 10:, :]


rotations = torch.cat((r0, toes, r1, toes, r2, toes, r3, toes), dim=1)
print(rotations[:, [4,8,12,16]])


a1_skeleton = SkeletonState.from_file('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/a1_tpose_v2.npy').skeleton_tree

# print(a1_skeleton.to_dict()['node_names'])
# input()
a1_state = SkeletonState.from_rotation_and_root_translation(
            a1_skeleton, r=rotations, t=positions, is_local=True
        )
motion = SkeletonMotion.from_skeleton_state(a1_state, fps=60)

# plot_skeleton_motion_interactive(motion)
motion.to_file('data/a1_recording_processed/demo_recording.npy')