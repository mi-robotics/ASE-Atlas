from isaacgym.torch_utils import *
import torch
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.visualization.plot_simple import SimplePlotter

rotations = torch.Tensor(np.load('./rotations-v6.npy'))
positions = torch.Tensor(np.load('./positions-v6.npy'))




a1_skeleton = SkeletonState.from_file('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/a1_tpose_v2.npy').skeleton_tree

# print(a1_skeleton.to_dict()['node_names'])
# input()
a1_state = SkeletonState.from_rotation_and_root_translation(
            a1_skeleton, r=rotations, t=positions, is_local=True
        )
motion = SkeletonMotion.from_skeleton_state(a1_state, fps=60)

# plot_skeleton_motion_interactive(motion)
motion.to_file('data/demo_v7.npy')