
import subprocess


# command = 'cd .. && python ase/run.py \
#     --task A1ASE \
#     --cfg_env ./ase/data/cfg/experiments/curiase/test1/env.yaml \
#     --cfg_train ./ase/data/cfg/experiments/curiase/test1/train.yaml \
#     --motion_file ./ase/data/motions/dog_mocap_processed/all_inv.yaml \
#    --headless'

command = 'cd .. && python ase/run.py \
    --task A1ASE \
    --cfg_env ./ase/data/cfg/experiments/tripASE/env.yaml \
    --cfg_train ./ase/data/cfg/experiments/tripASE/train.yaml \
    --motion_file ./ase/data/motions/all_inv.yaml\
   --headless'



subprocess.run(command, shell=True, check=True, text=True)
