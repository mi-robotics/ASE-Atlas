import subprocess


command = 'cd .. && python ./ase/run.py \
    --task HumanoidAMP \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/train/rlg/amp_humanoid.yaml\
    --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
    --headless'


subprocess.run(command, shell=True, check=True, text=True)

command = 'cd .. && python ./ase/run.py \
    --task HumanoidAMP \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/train/rlg/amp_humanoid.yaml\
    --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_2.yaml\
    --headless'


subprocess.run(command, shell=True, check=True, text=True)

command = 'cd .. && python ./ase/run.py \
    --task HumanoidAMP \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/train/rlg/amp_humanoid.yaml\
    --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_3.yaml\
    --headless'


subprocess.run(command, shell=True, check=True, text=True)

command = 'cd .. && python ./ase/run.py \
    --task HumanoidAMP \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/train/rlg/amp_humanoid.yaml\
    --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_4.yaml\
    --headless'


subprocess.run(command, shell=True, check=True, text=True)

command = 'cd .. && python ./ase/run.py \
    --task HumanoidAMP \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/train/rlg/amp_humanoid.yaml\
    --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_5.yaml\
    --headless'


subprocess.run(command, shell=True, check=True, text=True)