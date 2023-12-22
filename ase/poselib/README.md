# poselib

`poselib` is a library for loading, manipulating, and retargeting skeleton poses and motions. It is separated into three modules: `poselib.core` for basic data loading and tensor operations, `poselib.skeleton` for higher-level skeleton operations, and `poselib.visualization` for displaying skeleton poses. This library is built on top of the PyTorch framework and requires data to be in PyTorch tensors.

## poselib.core
- `poselib.core.rotation3d`: A set of Torch JIT functions for computing quaternions, transforms, and rotation/transformation matrices.
    - `quat_*` manipulate and create quaternions in [x, y, z, w] format (where w is the real component).
    - `transform_*` handle 7D transforms in [quat, pos] format.
    - `rot_matrix_*` handle 3x3 rotation matrices.
    - `euclidean_*` handle 4x4 Euclidean transformation matrices.
- `poselib.core.tensor_utils`: Provides loading and saving functions for PyTorch tensors.

## poselib.skeleton
- `poselib.skeleton.skeleton3d`: Utilities for loading and manipulating skeleton poses, and retargeting poses to different skeletons.
    - `SkeletonTree` is a class that stores a skeleton as a tree structure. This describes the skeleton topology and joints.
    - `SkeletonState` describes the static state of a skeleton, and provides both global and local joint angles.
    - `SkeletonMotion` describes a time-series of skeleton states and provides utilities for computing joint velocities.

## poselib.visualization
- `poselib.visualization.common`: Functions used for visualizing skeletons interactively in `matplotlib`.
    - In SkeletonState visualization, use key `q` to quit window.
    - In interactive SkeletonMotion visualization, you can use the following key commands:
        - `w` - loop animation
        - `x` - play/pause animation
        - `z` - previous frame
        - `c` - next frame
        - `n` - quit window

## Key Features
Poselib provides several key features for working with animation data. We list some of the frequently used ones here, and provide instructions and examples on their usage.

### Importing from FBX
Poselib supports importing skeletal animation sequences from .fbx format into a SkeletonMotion representation. To use this functionality, you will need to first set up the Python FBX SDK on your machine using the following instructions.

This package is necessary to read data from fbx files, which is a proprietary file format owned by Autodesk. The latest FBX SDK tested was FBX SDK 2020.2.1 for Python 3.7, which can be found on the Autodesk website: https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-2-1.

Follow the instructions at https://help.autodesk.com/view/FBX/2020/ENU/?guid=FBX_Developer_Help_scripting_with_python_fbx_installing_python_fbx_html for download, install, and copy/paste instructions for the FBX Python SDK.

This repo provides an example script `fbx_importer.py` that shows usage of importing a .fbx file. Note that `SkeletonMotion.from_fbx()` takes in an optional parameter `root_joint`, which can be used to specify a joint in the skeleton tree as the root joint. If `root_joint` is not specified, we will default to using the first node in the FBX scene that contains animation data. 

### Importing from MJCF
MJCF is a robotics file format supported by Isaac Gym. For convenience, we provide an API for importing MJCF assets into SkeletonTree definitions to represent the skeleton topology. An example script `mjcf_importer.py` is provided to show usage of this.

This can be helpful if motion sequences need to be retargeted to your simulation skeleton that's been created in MJCF format. Importing the file to SkeletonTree format will allow you to generate T-poses or other retargeting poses that can be used for retargeting. We also show an example of creating a T-Pose for our AMP Humanoid asset in `generate_amp_humanoid_tpose.py`.

### Retargeting Motions
Retargeting motions is important when your source data uses skeletons that have different morphologies than your target skeletons. We provide APIs for performing retarget of motion sequences in our SkeletonState and SkeletonMotion classes.

To use the retargeting API, users must provide the following information:
  - source_motion: a SkeletonMotion npy representation of a motion sequence. The motion clip should use the same skeleton as the source T-Pose skeleton.
  - target_motion_path: path to save the retargeted motion to
  - source_tpose: a SkeletonState npy representation of the source skeleton in it's T-Pose state
  - target_tpose: a SkeletonState npy representation of the target skeleton in it's T-Pose state (pose should match source T-Pose)
  - joint_mapping: mapping of joint names from source to target
  - rotation: root rotation offset from source to target skeleton (for transforming across different orientation axes), represented as a quaternion in XYZW order.
  - scale: scale offset from source to target skeleton

We provide an example script `retarget_motion.py` to demonstrate usage of the retargeting API for the CMU Motion Capture Database. Note that the retargeting data for this script is stored in `data/configs/retarget_cmu_to_amp.json`.

Additionally, a SkeletonState T-Pose file and retargeting config file are also provided for the SFU Motion Capture Database. These can be found at `data/sfu_tpose.npy` and `data/configs/retarget_sfu_to_amp.json`.

### Documentation
We provide a description of the functions and classes available in poselib in the comments of the APIs. Please check them out for more details.



##
In the context of retargeting motion data from one skeleton to another, the parameters you've mentioned—rotation, scale, root_height_offset—play important roles in aligning and adapting the source motion to the target skeleton. Here's an explanation of each parameter:

    Rotation (Quaternion):
        Purpose: Rotation represents the orientation of the source skeleton in relation to the target skeleton. It's often used to account for differences in the initial pose or orientation between the two skeletons.
        Usage: The provided quaternion (four-element vector) describes a rotation that should be applied to the source motion to align it with the target motion. In your example, the quaternion [0, 0, 0.7071068, 0.7071068] represents a rotation of 90 degrees (π/2 radians) around the Z-axis. This means that the source skeleton needs to be rotated by 90 degrees to match the orientation of the target skeleton.

    Scale:
        Purpose: Scale accounts for differences in the size or proportions of the source and target skeletons. It's used to ensure that the retargeted motion is scaled appropriately.
        Usage: The scale parameter specifies a scaling factor that should be applied to the source motion. It determines how much the source motion should be resized to match the proportions of the target skeleton. A smaller value would shrink the motion, while a larger value would enlarge it. The specific value (0.056444 in your example) depends on the scaling factor required to match the source and target skeletons.

    Root Height Offset:
        Purpose: The root height offset is used to adjust the height of the source skeleton's root (often the hips or pelvis) relative to the target skeleton's root. This is important for ensuring that the source skeleton's motion aligns with the target skeleton's position in the vertical axis.
        Usage: The root_height_offset parameter specifies how much the source skeleton's root should be raised or lowered to match the height of the target skeleton. A positive value raises the source skeleton, while a negative value lowers it. The specific value (0.05 in your example) depends on the vertical offset needed to align the skeletons' root positions.

These parameters are crucial for making sure that the retargeted motion appears natural and correctly aligned with the target skeleton. They are typically determined through a combination of manual adjustment and experimentation to achieve the best visual results.

Finding the precise values for these parameters often involves a trial-and-error process, where you visually inspect the retargeted motion and make adjustments until it looks natural and properly aligned with the target skeleton. It's common to work iteratively, refining these parameters until you achieve the desired result.

Additionally, you may use specialized software or tools that provide visualization and adjustment capabilities to aid in the retargeting process. The specific values will depend on the characteristics of the source and target skeletons and the motion data you're working with.