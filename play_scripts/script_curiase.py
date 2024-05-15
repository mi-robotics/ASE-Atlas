import subprocess


command = 'cd .. && python ./ase/run.py \
    --test --task A1ASE \
    --noise_level 1.0 \
    --friction_overide 1 \
    --use_delay 0 \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/experiments/curiase/test1/env.yaml \
    --cfg_train ./ase/data/cfg/experiments/curiase/test1/train.yaml \
    --motion_file ./ase/data/motions/a1_complex_processed/handstand_1707757090.0902457_5.npy \
    --checkpoint ./output/IROS_LARGE_CURIASE_26-22-01-27/nn/IROS_LARGE_CURIASE.pth'



# /home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/data/motions/a1_complex_processed/biped_1707756804.4866664_3.npy
# /home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/data/motions/a1_complex_processed/handstand_1707757090.0902457_5.npy

subprocess.run(command, shell=True, check=True, text=True)