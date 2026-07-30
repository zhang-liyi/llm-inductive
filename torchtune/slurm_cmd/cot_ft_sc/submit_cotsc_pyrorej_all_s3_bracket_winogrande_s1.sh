#!/bin/bash
#SBATCH --job-name=sf3-winogra-s1
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=end
#SBATCH --mail-user=akshay.jagadish@princeton.edu
export PYTHONPATH=/home/aj9225/llm-inductive/torchtune
cd /home/aj9225/llm-inductive/data_evaluation
python /home/aj9225/llm-inductive/data_evaluation/evaluate_text_classification_cot.py \
    --ckpt_dir /home/aj9225/llm-inductive/data_root/ckpt/pyro_rej_lora_dist_r8_all_seed3_bracket_lora8_dist/epoch_0 \
    --sample_idx 1 \
    --sc_defaults \
    --seed 0 \
    --dataset winogrande \
    --subsample_n 500 \
    --subsample_seed 1234 \
    --start_idx 0 \
    --n_examples 500 \
    --max_new_tokens 512 \
    --output_file /home/aj9225/llm-inductive/data_evaluation/results/text_cls/cotsc_pyrorej_all_s3_bracket_winogrande_s1.json \
    >/home/aj9225/llm-inductive/archive/cotsc_pyrorej_all_s3_bracket_winogrande_s1.out 2>&1
