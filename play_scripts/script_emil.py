import subprocess

#BEST
# command = 'cd .. && python ase/run.py \
#     --test --task HumanoidAMP \
#     --num_envs 1 \
#     --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
#     --cfg_train ./ase/data/cfg/experiments/lsgm/train_v1.yaml \
#     --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_1.yaml \
#     --checkpoint ./output/LSGM_V1_REPRO_TEST_smooth_23-21-23-09/nn/LSGM_V1_REPRO_TEST_smooth.pth'





command = 'cd .. && python ase/run.py \
    --test --task HumanoidAMP \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/experiments/lsgm/train_emil.yaml \
    --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
    --checkpoint ./output/EMIL_DEBUG_21-15-29-47/nn/EMIL_DEBUG.pth'


subprocess.run(command, shell=True, check=True, text=True)