import subprocess


command = 'cd .. && python ./ase/run.py \
    --test --task A1ASE \
    --noise_level 1.0 \
    --friction_overide 1 \
    --use_delay 0 \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/experiments/curiase/test1/env.yaml \
    --cfg_train ./ase/data/cfg/experiments/curiase/test1/train.yaml \
    --motion_file ./ase/data/motions/dogo/dogo_data_1.yaml \
    --checkpoint ./output/CURIASE_TEST_1_13-09-02-19/nn/CURIASE_TEST_1.pth'




subprocess.run(command, shell=True, check=True, text=True)