import subprocess


command = 'python ase/run.py \
    --test --task HumanoidAMP \
    --num_envs 16 \
    --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ase/data/cfg/train/rlg/lasd_humanoid.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml \
    --checkpoint output/LASD_V1_single/nn/Humanoid.pth'
subprocess.run(command, shell=True, check=True, text=True)