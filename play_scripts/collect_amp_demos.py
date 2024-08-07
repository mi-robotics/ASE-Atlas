import subprocess


experiments = [
    {'dataset':'reallusion_1.yaml', 'model':'Humanoid_AMP_24-13-34-52'},    
    {'dataset':'reallusion_2.yaml', 'model':'Humanoid_AMP_24-13-34-52'},
    {'dataset':'reallusion_3.yaml', 'model':'Humanoid_AMP_24-13-34-52'},
    {'dataset':'reallusion_4.yaml', 'model':'Humanoid_AMP_24-13-34-52'},
    {'dataset':'reallusion_5.yaml', 'model':'Humanoid_AMP_24-13-34-52'},
]

command = 'cd .. && python ase/run.py \
    --test --task HumanoidAMP \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ./ase/data/cfg/train/rlg/amp_humanoid.yaml\
    --motion_file ./ase/data/motions/reallusion_sword_shield/reallusion_1.yaml \
    --checkpoint ./output/Humanoid_AMP_24-13-34-52/nn/Humanoid_AMP_00002000.pth'


subprocess.run(command, shell=True, check=True, text=True)