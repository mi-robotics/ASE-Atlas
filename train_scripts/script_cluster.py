import subprocess


command = 'python ase/run.py \
    --task HumanoidAMP \
    --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ase/data/cfg/experiments/cluster/test.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
    --headless'

subprocess.run(command, shell=True, check=True, text=True)
