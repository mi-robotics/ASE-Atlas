import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive


import torch

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state
from poselib.visualization.plot_simple import SimplePlotter

fbx_file = '/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/poselib/data/dog_demo.fbx'

motion, skeleton_state = SkeletonMotion.from_fbx(
    fbx_file_path=fbx_file,
    root_joint="Hips",
    fps=60,
    return_skeleton_state=True
)

# plotter = SimplePlotter(skeleton_state=skeleton_state)

print(skeleton_state.skeleton_tree.node_names)

zero_pose = SkeletonState.zero_pose(skeleton_state.skeleton_tree)
skeleton = zero_pose.skeleton_tree

local_rotation = zero_pose.local_rotation

local_rotation[skeleton.index("LeftUpLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), 
    local_rotation[skeleton.index("LeftUpLeg")]
)
local_rotation[skeleton.index("RightUpLeg")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), 
    local_rotation[skeleton.index("RightUpLeg")]
)

local_rotation[skeleton.index("Spine")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([180.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), 
    local_rotation[skeleton.index("Spine")]
)

local_rotation[skeleton.index("LeftShoulder")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), 
    local_rotation[skeleton.index("LeftShoulder")]
)

local_rotation[skeleton.index("RightShoulder")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), 
    local_rotation[skeleton.index("RightShoulder")]
)

plotter = SimplePlotter(skeleton_state=zero_pose)

zero_pose.to_file("data/dog_tpose.npy")