import subprocess


# command = 'python ase/run.py \
#     --task A1 \
#     --cfg_env ase/data/cfg/a1_ase_env.yaml \
#     --cfg_train ase/data/cfg/train/rlg/ase_a1.yaml \
#     --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml'# \
#     #--headless'


command = 'python ase/run.py \
    --task HumanoidAMP \
    --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ase/data/cfg/train/rlg/lasd_humanoid.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
    --headless'



subprocess.run(command, shell=True, check=True, text=True)