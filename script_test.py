import subprocess


command = 'python ase/run.py \
    --test --task A1ASE \
    --num_envs 1 \
    --cfg_env ase/data/cfg/a1_ase_env.yaml \
    --cfg_train ase/data/cfg/train/rlg/ase_a1.yaml \
    --motion_file ase/data/motions/dogo/dogo_data_1.yaml \
    --checkpoint output/dog_single_long_ld24/nn/Humanoid.pth'
subprocess.run(command, shell=True, check=True, text=True)