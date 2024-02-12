import subprocess


command = 'cd .. && python ./ase/run.py \
    --test --task Go2Transfer \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/experiments/go2Transfer/a1_vel_est_env.yaml \
    --cfg_train ./ase/data/cfg/experiments/go2Transfer/a1_vel_est_train.yaml \
    --motion_file ./ase/data/motions/dogo/dogo_data_1.yaml \
    --checkpoint ./output/LARGE_A1_ALL_26-04-51-20/nn/LARGE_A1_ALL.pth'


subprocess.run(command, shell=True, check=True, text=True)