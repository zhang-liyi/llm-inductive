#!/bin/bash
#SBATCH --job-name=mb-pyrorej-all-s2-bracket-morebench-theo
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
    --ckpt_dir /home/aj9225/llm-inductive/data_root/ckpt/pyro_rej_lora_dist_r8_all_seed2_bracket_lora8_dist/epoch_0 \
    --dataset morebench_theory \
    --output_file /home/aj9225/llm-inductive/data_evaluation/results/text_cls/pyrorej_all_s2_bracket_morebench_theory.json \
    >/home/aj9225/llm-inductive/archive/pyrorej_all_s2_bracket_morebench_theory.out 2>&1
