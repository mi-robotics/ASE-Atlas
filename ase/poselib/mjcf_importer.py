# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state
from poselib.visualization.plot_simple import SimplePlotter



# load in XML mjcf file and save zero rotation pose in npy format
xml_path = "../../../../assets/mjcf/nv_humanoid.xml"
xml_path = "/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/data/assets/mjcf/amp_humanoid.xml"
xml_path = "/home/milo/Documents/cdt-1/assets/mujoco_menagerie/unitree_a1/a1.xml"
xml_path = "/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/data/assets/parkour/a1/urdf/a1.urdf"

skeleton = SkeletonTree.from_urdf(xml_path)

print(skeleton.to_dict()['node_names'])
print(skeleton.to_dict()['parent_indices'])
# print(skeleton.node_names)
# print(len(skeleton.node_names))

# input()

zero_pose = SkeletonState.zero_pose(skeleton)
# plotter = SimplePlotter(skeleton_state=zero_pose)
# zero_pose.to_file("data/a1_tpose_v2.npy")

# visualize zero rotation pose
# plot_skeleton_state(zero_pose)