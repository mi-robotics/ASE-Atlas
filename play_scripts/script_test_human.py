import subprocess

#BEST
command = 'cd .. && python ase/run.py \
    --test --task HumanoidAMP \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/experiments/lsgm/train_v1.yaml \
    --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_1.yaml \
    --checkpoint ./output/LSGM_V3_21-15-42-50/nn/LSGM_V3.pth'





# command = 'cd .. && python ase/run.py \
#     --test --task HumanoidAMP \
#     --num_envs 1 \
#     --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
#     --cfg_train ./ase/data/cfg/train/rlg/amp_humanoid.yaml\
#     --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_1.yaml \
#     --checkpoint ./output/Humanoid_AMP_13-13-06-34/nn/Humanoid_AMP_00001000.pth'


subprocess.run(command, shell=True, check=True, text=True)