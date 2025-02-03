#!/usr/bin/env bash
#SBATCH --job-name DeepLearning
#SBATCH --time 80:00:00
#SBATCH --output DeepLearning-%j.txt
#SBATCH --error DeepLearning-%j.txt
##SBATCH --constraint=highmem
#SBATCH --mem=32G  # Adjust as needed
cd ~/skin_disorder
date

# Define the command to run the Python script
#CMD="python3 ./run_training.py --classification_type kl --class_balance_type aug --custom custom --no_epochs 50 --output_dir all_results/results_24012025_a/Cust_kl_aug"

CMD="python3 ./run_training.py --classification_type kl --class_balance_type aug --fine_tune --basemodel MN --no_epochs 50 --output_dir all_results/results_24012025/MN_kl_aug_ft"
# Print the command to the output
echo "Running command: $CMD"

# Run the command
$CMD

date