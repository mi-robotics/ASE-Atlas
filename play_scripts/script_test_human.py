import subprocess


command = 'cd .. && python ase/run.py \
    --test --task HumanoidAMP \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/experiments/lsgm/train_v1.yaml \
    --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_1.yaml \
    --checkpoint ./output/LSGM_V1_LARGE_28-21-14-37/nn/LSGM_V1_LARGE.pth'
subprocess.run(command, shell=True, check=True, text=True)