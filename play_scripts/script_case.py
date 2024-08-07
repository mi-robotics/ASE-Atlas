import subprocess


command = 'cd .. && python ./ase/run.py \
    --test --task A1CASE \
    --noise_level 1.0 \
    --friction_overide 1 \
    --use_delay 0 \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/experiments/CASE_inference/env.yaml \
    --cfg_train ./ase/data/cfg/experiments/CASE_inference/train.yaml \
    --motion_file ./ase/data/motions/all_play.yaml\
    --checkpoint ./output/CASE_FOCAL_FINAL_ALL_SKILLS_2_31-17-09-00/nn/CASE_FOCAL_FINAL_ALL_SKILLS_2.pth'

# command = 'cd .. && python ./ase/run.py \
#     --test --task A1ASE \
#     --noise_level 2.0 \
#     --num_envs 1 \
#     --cfg_env ./ase/data/cfg/experiments/velocity_a1/noised_asym/a1_vel_est_env.yaml \
#     --cfg_train ./ase/data/cfg/experiments/velocity_a1//noised_asym/a1_vel_est_train.yaml \
#     --motion_file ./ase/data/motions/dogo/dogo_data_1.yaml \
#     --checkpoint ./output/ase_a1_OFFICIAL_asymetric_noised_19-12-51-04/nn/ase_a1_OFFICIAL_asymetric_noised.pth'


subprocess.run(command, shell=True, check=True, text=True)