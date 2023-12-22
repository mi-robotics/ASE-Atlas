

# Trying to avoid writing a new package - used https://anyconv.com/bvh-to-fbx-converter/ to convert files 




import os
import json
import math
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import torch

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state
from poselib.visualization.plot_simple import SimplePlotter
"""
- roation: the rotation different between tposes
- scale: seem more closely related to arm span
- root_height_offset:
"""
a1_skeleton = SkeletonState.from_file('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/a1_tpose.npy')
dog_skeleton = SkeletonState.from_file('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/dog_tpose.npy')

print(a1_skeleton.skeleton_tree.node_names)

# cmu_dots = cmu_skeleton.global_translation.numpy()[:,2]
# amp_dots = amp_skeleton.global_translation.numpy()[:,2]

# print(sorted(cmu_dots))

# cmu_min, cmu_max = min(cmu_dots), max(cmu_dots)
# cmu_max = 23.679756
# cmu_min = 1.133996
# amp_min, amp_max = min(amp_dots), max(amp_dots)

# print(amp_min, cmu_min)
# print(amp_max/cmu_max)
# print((amp_max-amp_min)/(cmu_max-cmu_min))



plot_skeleton_state(a1_skeleton)
plot_skeleton_state(dog_skeleton)
