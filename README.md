# Inductive LLM

Code for fine-tuning and evaluating LLMs on probabilistic / inductive reasoning
tasks. This repo holds the source, configs, and data-prep / evaluation scripts;
trained checkpoints and external model weights are not included.

## Layout

| Directory | Contents |
| --- | --- |
| `torchtune/` | Fine-tuning recipes, configs (`config_files/`), and the `torchtune/torchtune/` Python package. |
| `posterior_sampling_pytorch/` | Pyro-based program family (rejection sampling and direct inference). |
| `forward_sampling/` | Forward-sampling baseline scripts. |
| `gemini_direct/` | Gemini-direct integer-answer baseline. |
| `data_processing/` | Dataset preparation for OpenEstimate and Bayesian Teaching. |
| `data_evaluation/` | Evaluation pipelines (OpenEstimate, Bayesian Teaching, text classification), the calibration-table aggregator, the temperature-scaling baseline, and the healthcare estimation eval. |
| `utils/`, `generation_prompts/` | Misc helpers and prompt templates. |

## Setup

Run scripts from the repo root — relative paths use `./` as the project root.

External data and model weights (Llama-3, Qwen-2, datasets) are referenced as
`<DATA_ROOT>/...`. Point this at the directory holding your downloaded
HuggingFace caches and other resources:

```bash
export DATA_ROOT=/path/to/external/data
sed -i "s|<DATA_ROOT>|$DATA_ROOT|g" *.sh torchtune/*.sh
```

Scripts that call external LLM APIs read their keys from the environment:

```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Running

**Train** a LoRA adapter with a torchtune recipe + config:

```bash
tune run --nproc_per_node 1 torchtune/custom_lora_answer_only.py \
    --config torchtune/config_files/pyro_rej_lora_dist_all_seed1_bracket.yaml
```

**Evaluate** a checkpoint on a benchmark (writes per-query JSON to
`data_evaluation/results/`):

```bash
# Open-ended estimation
python data_evaluation/evaluate_openestimate.py \
    --ckpt_dir torchtune/ckpt/llama3_8B/pyro_rej_lora_dist_r8_all_seed1_bracket/epoch_2 \
    --output_file data_evaluation/results/openestimate/pyrorej_all_s1_bracket_openestimate.json

# Bayesian Teaching
python data_evaluation/evaluate_bayesian_teaching.py \
    --ckpt_dir torchtune/ckpt/llama3_8B/pyro_rej_lora_dist_r8_all_seed1_bracket/epoch_2 \
    --output_file data_evaluation/results/bayesian_teaching/pyrorej_all_s1_bracket_bayesian_teaching_guided.json
```

**Aggregate** the result JSONs into calibration tables:

```bash
python data_evaluation/aggregate_pyrorej_all_bracket_calibration.py
```
