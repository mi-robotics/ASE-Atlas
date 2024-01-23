
import subprocess



# command = 'python ase/run.py \
#     --task A1ASE \
#     --num_envs 128 \
#     --cfg_env ase/data/cfg/a1_ase_env.yaml \
#     --cfg_train ase/data/cfg/train/rlg/ase_a1.yaml \
#     --motion_file ase/data/motions/dogo/dogo_data_1.yaml'

# command = 'python ase/run.py \
#     --task A1ASE \
#     --cfg_env ase/data/cfg/experiments/velocity_a1/a1_vel_est_env.yaml \
#     --cfg_train ase/data/cfg/experiments/velocity_a1/a1_vel_est_train.yaml \
#     --motion_file ase/data/motions/dogo/dogo_data_1.yaml \
#    --headless'

command = 'python ase/run.py \
    --task A1ASE \
    --cfg_env ase/data/cfg/experiments/velocity_a1/large/dogo_all/a1_vel_est_env.yaml \
    --cfg_train ase/data/cfg/experiments/velocity_a1/large/dogo_all/a1_vel_est_train.yaml \
    --motion_file ase/data/motions/dog_mocap_processed/all_inv.yaml\
   --headless'

subprocess.run(command, shell=True, check=True, text=True)


command = 'python ase/run.py \
    --task A1ASE \
    --cfg_env ase/data/cfg/experiments/velocity_a1/large/a1_all/a1_vel_est_env.yaml \
    --cfg_train ase/data/cfg/experiments/velocity_a1/large/a1_all/a1_vel_est_train.yaml \
    --motion_file ase/data/motions/a1_recording_processed/all_inv.yaml \
   --headless'

subprocess.run(command, shell=True, check=True, text=True)
