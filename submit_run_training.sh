#!/usr/bin/env bash
#SBATCH --job-name DeepLearning
#SBATCH --time 40:00:00
#SBATCH --output DeepLearning-%j.txt
#SBATCH --error DeepLearning-%j.txt
#SBATCH --constraint=highmem

cd ~/skin_disorder
date

# Define the command to run the Python script
CMD="python3 ./run_training.py --classification_type mc --class_balance_type aug --basemodel VG --no_epochs 50 --fine_tune --output_dir results_23092024/VG_mc_aug_ft"

# Print the command to the output
echo "Running command: $CMD"

# Run the command
$CMD

date