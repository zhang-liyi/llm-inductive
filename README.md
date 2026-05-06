# Inductive LLM — Paper Supplementary Material

## Layout

| Directory | Contents |
| --- | --- |
| `torchtune/` | Fine-tuning recipes, configs (`config_files/`), and the `torchtune/torchtune/` Python package. |
| `posterior_sampling_pytorch/` | Pyro-based program family (rejection sampling and MCMC inference) with a sample of programs. |
| `forward_sampling/` | Forward-sampling baseline scripts. |
| `gemini_direct/` | Code for Gemini-direct integer-answer baseline. |
| `data_processing/` | Dataset preparation for OpenEstimate, Bayesian Teaching evaluations. |
| `data_evaluation/` | Evaluation pipelines and seed-1 result JSONs for OpenEstimate, Bayesian Teaching, and text classification. |
| `benchmarking/` | LLM-vs-human comparison scripts and core inference results for two representative models. |
| `utils/`, `generation_prompts/` | Misc helpers, notes, and prompt templates. |

## Setup

These scripts assume the working directory is the repo root. Most relative paths
written into shell scripts and Python modules use `./` as the project root.

External data and model weights (Llama-3, Qwen-2, datasets) are referenced as
`<DATA_ROOT>/...` — set this to the parent directory containing your downloaded
HuggingFace caches and other resources, e.g.

```bash
# example: edit shell scripts that reference <DATA_ROOT>
export DATA_ROOT=/path/to/external/data
sed -i "s|<DATA_ROOT>|$DATA_ROOT|g" *.sh torchtune/*.sh
```

### API Keys

Several scripts call external LLM APIs. The actual key strings have been stripped
from the source. Replace the placeholders or, preferred, set environment variables:

```bash
export GEMINI_API_KEY=...   # for gemini_direct/, posterior_sampling/msa_*.py
export OPENAI_API_KEY=...   # if running the GPT pipelines
export ANTHROPIC_API_KEY=... # if running Claude pipelines
```

In source you may see literal placeholder tokens such as
`<GEMINI_API_KEY_PLACEHOLDER>` — replace these with your real key (or refactor the
caller to read `os.environ["GEMINI_API_KEY"]`).

## Reproducing Results

To reproduce the fine-tuning pipeline:

1. Download base models (Llama-3-8B-Instruct, Qwen2-7B-Instruct) from HuggingFace
   into `<DATA_ROOT>/resources/`.
2. Re-train with the provided torchtune recipes/configs:
   ```bash
   tune run --nproc_per_node 1 torchtune/custom_lora_answer_only.py \
       --config torchtune/config_files/pyro_rej_lora_dist_all_seed1_bracket.yaml
   ```
3. Run evaluation:
   ```bash
   # Open-ended estimation
   python data_evaluation/evaluate_openestimate.py \
       --ckpt_dir torchtune/ckpt/llama3_8B/pyro_rej_lora_dist_r8_all_seed1_bracket/epoch_0 \
       --output_file data_evaluation/results/openestimate/pyrorej_all_s1_bracket_openestimate.json

   # Bayesian Teaching
   python data_evaluation/evaluate_bayesian_teaching.py \
       --ckpt_dir torchtune/ckpt/llama3_8B/pyro_rej_lora_dist_r8_all_seed1_bracket/epoch_0 \
       --output_file data_evaluation/results/bayesian_teaching/pyrorej_all_s1_bracket_bayesian_teaching_guided.json
   ```
4. Aggregate calibration tables:
   ```bash
   python data_evaluation/aggregate_pyrorej_all_bracket_calibration.py
   ```

The included seed-1 result JSONs allow inspection of per-query model outputs
without re-running inference.

