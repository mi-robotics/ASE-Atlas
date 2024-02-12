import subprocess


command = 'cd .. && python ase/run.py \
    --task HumanoidAMP \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/experiments/lsgm/train_v1.yaml \
    --motion_file ./ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml\
    --headless'



subprocess.run(command, shell=True, check=True, text=True)


