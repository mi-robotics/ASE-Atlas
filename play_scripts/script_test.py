import subprocess


command = 'cd .. && python ./ase/run.py \
    --test --task A1ASE \
    --noise_level 2.0 \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/experiments/velocity_a1/large/dogo_all/a1_vel_est_env.yaml \
    --cfg_train ./ase/data/cfg/experiments/velocity_a1/large/dogo_all/a1_vel_est_train.yaml \
    --motion_file ./ase/data/motions/dogo/dogo_data_1.yaml \
    --checkpoint ./output/demo/nn/LARGE_DOGO_ALL_00050000.pth'



# command = 'cd .. && python ./ase/run.py \
#     --test --task A1ASE \
#     --noise_level 2.0 \
#     --num_envs 1 \
#     --cfg_env ./ase/data/cfg/experiments/velocity_a1/noised_asym/a1_vel_est_env.yaml \
#     --cfg_train ./ase/data/cfg/experiments/velocity_a1//noised_asym/a1_vel_est_train.yaml \
#     --motion_file ./ase/data/motions/dogo/dogo_data_1.yaml \
#     --checkpoint ./output/ase_a1_OFFICIAL_asymetric_noised_19-12-51-04/nn/ase_a1_OFFICIAL_asymetric_noised.pth'


subprocess.run(command, shell=True, check=True, text=True)