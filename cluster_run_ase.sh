#!/bin/bash -l

# Batch script to run a GPU job under SGE.

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=true
#$ gpu_type=rtx6000

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=2:0:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=24G
#$ -l tmem=16G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N ASE_RUN_01

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
# $ -wd /home/mcarroll/ase

# Change into temporary directory to run work
cd $TMPDIR
cd isaac_gym
bash run_isaac.sh
cd ase

# Run the application - the line below is just a random example.
python script_train_human.py

# 10. Preferably, tar-up (archive) all output files onto the shared scratch area
tar zcvf $HOME/Scratch/files_from_job_$JOB_ID.tar.gz $TMPDIR

# Make sure you have given enough time for the copy to complete!