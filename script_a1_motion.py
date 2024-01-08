import subprocess


command = 'python ase/run.py --test \
    --task A1ViewMotion --num_envs 2 \
    --cfg_env ase/data/cfg/a1_ase_env.yaml \
    --cfg_train ase/data/cfg/train/rlg/ase_a1.yaml \
    --motion_file ase/data/motions/a1_recordings/demo_recording.npy'

subprocess.run(command, shell=True, check=True, text=True)