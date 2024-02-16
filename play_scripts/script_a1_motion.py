import subprocess


command = 'cd .. && python ./ase/run.py --test \
    --task A1ViewMotion --num_envs 1 \
    --cfg_env ./ase/data/cfg/experiments/velocity_a1/large/dogo_all/a1_vel_est_env.yaml \
    --cfg_train ./ase/data/cfg/experiments/velocity_a1/large/dogo_all/a1_vel_est_train.yaml \
    --motion_file ./ase/data/motions/v3_inv.yaml'


# /home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/data/motions/all_no_jump_inv.yaml
subprocess.run(command, shell=True, check=True, text=True)