#!/bin/bash
#SBATCH --job-name=mbm-mean-s2-morebench_theory
#SBATCH --time=0:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
mkdir -p /home/aj9225/llm-inductive/data_evaluation/results/text_cls /home/aj9225/llm-inductive/archive
export PYTHONPATH=/home/aj9225/llm-inductive/torchtune
cd /home/aj9225/llm-inductive/data_evaluation
python evaluate_text_classification.py \
    --ckpt_dir /scratch/gpfs/GRIFFITHS/lz3156/inductive-llm/torchtune/ckpt/llama3_8B/pyro_rej_lora_mean_r8_all_seed2_bracket_lora8_mean_only/epoch_0 \
    --dataset morebench_theory \
    --output_file /home/aj9225/llm-inductive/data_evaluation/results/text_cls/pyrorej_mean_s2_bracket_morebench_theory.json \
    >/home/aj9225/llm-inductive/archive/pyrorej_mean_s2_bracket_morebench_theory.out 2>&1
