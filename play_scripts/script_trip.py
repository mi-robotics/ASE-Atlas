import subprocess


command = 'cd .. && python ./ase/run.py \
    --test --task A1ASE \
    --noise_level 1.0 \
    --friction_overide 1 \
    --use_delay 0 \
    --num_envs 1 \
    --cfg_env ./ase/data/cfg/experiments/tripASE/env.yaml \
    --cfg_train ./ase/data/cfg/experiments/tripASE/train.yaml \
    --motion_file ./ase/data/motions/a1_complex_processed/biped_1707756804.4866664_2.npy \
    --checkpoint ./output/TRIPASE_TEST_11-16-24-42/nn/TRIPASE_TEST.pth'



# /home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/data/motions/a1_complex_processed/biped_1707756804.4866664_3.npy
# /home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/data/motions/a1_complex_processed/handstand_1707757090.0902457_5.npy

subprocess.run(command, shell=True, check=True, text=True)