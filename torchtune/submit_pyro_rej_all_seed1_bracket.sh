#!/bin/bash
#SBATCH --job-name=pyro-rej-all-s1-br
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=end
#SBATCH --mail-user=anonymous@example.com
cd ./torchtune
PYTHONPATH=${PWD}:$PYTHONPATH tune run custom_lora_answer_only.py \
    --config config_files/pyro_rej_lora_dist_all_seed1_bracket.yaml \
    >./archive/pyro_rej_all_seed1_bracket.out 2>&1
