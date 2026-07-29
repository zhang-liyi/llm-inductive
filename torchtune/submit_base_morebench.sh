#!/bin/bash
#SBATCH --job-name=mb-base-morebench
#SBATCH --time=0:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=end
#SBATCH --mail-user=akshay.jagadish@princeton.edu
mkdir -p /home/aj9225/llm-inductive/data_evaluation/results/text_cls /home/aj9225/llm-inductive/archive
export PYTHONPATH=/home/aj9225/llm-inductive/torchtune
cd /home/aj9225/llm-inductive/data_evaluation
python /home/aj9225/llm-inductive/data_evaluation/evaluate_text_classification.py \
    --ckpt_dir /home/aj9225/llm-inductive/data_root/base_models/Meta-Llama-3-8B-Instruct \
    --dataset morebench \
    --output_file /home/aj9225/llm-inductive/data_evaluation/results/text_cls/base_morebench.json \
    --pretrained \
    >/home/aj9225/llm-inductive/archive/base_morebench.out 2>&1
