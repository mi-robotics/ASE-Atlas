import subprocess

# # joined skill obs recon
# command = 'python ase/run.py \
#     --task HumanoidAMP \
#     --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
#     --cfg_train ase/data/cfg/experiments/vae/vae_joined_skill_obs.yaml \
#     --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
#     --headless'

# subprocess.run(command, shell=True, check=True, text=True)

# joined skill obs recon (128 ld)
# command = 'python ase/run.py \
#     --task HumanoidAMP \
#     --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
#     --cfg_train ase/data/cfg/experiments/vae/vae_joined_skill_obs_128.yaml \
#     --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
#     --headless'

# subprocess.run(command, shell=True, check=True, text=True)

# joined skill obs recon (0.1 beta)
# command = 'python ase/run.py \
#     --task HumanoidAMP \
#     --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
#     --cfg_train ase/data/cfg/experiments/vae/vae_joined_skill_obs_beta.yaml \
#     --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
#     --headless'

# subprocess.run(command, shell=True, check=True, text=True)

# joined skill obs recon (0.1 beta)
command = 'python ase/run.py \
    --task HumanoidAMP \
    --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
    --cfg_train ase/data/cfg/experiments/vae/vae_joined_skill_2beta.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
    --headless'

subprocess.run(command, shell=True, check=True, text=True)

# # joined skill obs recon (128 ld, 0.1 beta)
# command = 'python ase/run.py \
#     --task HumanoidAMP \
#     --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
#     --cfg_train ase/data/cfg/experiments/vae/vae_joined_skill_obs_128_beta.yaml \
#     --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
#     --headless'

# subprocess.run(command, shell=True, check=True, text=True)

# # joined skill recon
# command = 'python ase/run.py \
#     --task HumanoidAMP \
#     --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
#     --cfg_train ase/data/cfg/experiments/vae/vae_joined_skill.yaml \
#     --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
#     --headless'

# subprocess.run(command, shell=True, check=True, text=True)


# #seperate skill obs recon 
# command = 'python ase/run.py \
#     --task HumanoidAMP \
#     --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
#     --cfg_train ase/data/cfg/experiments/vae/vae_seperate_skill_obs.yaml \
#     --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
#     --headless'

# subprocess.run(command, shell=True, check=True, text=True)

# #joined skill recon
# command = 'python ase/run.py \
#     --task HumanoidAMP \
#     --cfg_env ase/data/cfg/humanoid_ase_sword_shield.yaml \
#     --cfg_train ase/data/cfg/experiments/vae/vae_seperate_skill.yaml \
#     --motion_file ase/data/motions/reallusion_sword_shield/reallusion_1.yaml\
#     --headless'

# subprocess.run(command, shell=True, check=True, text=True)