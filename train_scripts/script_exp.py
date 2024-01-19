import subprocess


# # joined skill obs recon (0.1 beta)
# command = 'python ase/run.py \
#     --task A1ASE \
#     --cfg_env ase/data/cfg/experiments/velocity_a1/train_est/a1_vel_est_env.yaml \
#     --cfg_train ase/data/cfg/experiments/velocity_a1/train_est/a1_vel_est_train.yaml \
#     --motion_file ase/data/motions/dogo/dogo_data_1.yaml \
#    --headless'


# subprocess.run(command, shell=True, check=True, text=True)

command = 'python ase/run.py \
    --task A1ASE \
    --cfg_env ase/data/cfg/experiments/velocity_a1/noised_asym/a1_vel_est_env.yaml \
    --cfg_train ase/data/cfg/experiments/velocity_a1/noised_asym/a1_vel_est_train.yaml \
    --motion_file ase/data/motions/dogo/dogo_data_1.yaml \
   --headless'


subprocess.run(command, shell=True, check=True, text=True)
