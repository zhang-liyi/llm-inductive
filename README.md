# Inductive LLM — Paper Supplementary Material

This is anonymized supplementary material accompanying a paper submission. It is a
trimmed snapshot of the research codebase: source code, configs, and a small set of
representative result files. **Trained model checkpoints are NOT included** — see
"Reproducing Results" below.

## Layout

| Directory | Contents |
| --- | --- |
| `torchtune/` | Fine-tuning recipes, configs (`config_files/`), checkpoint metadata (`ckpt/*/torchtune_config.yaml`), and the `torchtune/torchtune/` Python package. |
| `posterior_sampling_pytorch/` | Pyro-based program family (rejection sampling and direct inference) with a stratified sample of programs. |
| `forward_sampling/` | Forward-sampling baseline scripts. |
| `gemini_direct/` | Code for Gemini-direct integer-answer baseline. |
| `data_processing/` | Dataset preparation for OpenEstimate and Bayesian Teaching evaluations. |
| `data_evaluation/` | Evaluation pipelines for OpenEstimate, Bayesian Teaching, and text classification, plus the calibration-table aggregator, temperature-scaling baseline, and the healthcare estimation eval. |
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
export GEMINI_API_KEY=...   # for gemini_direct/, posterior_sampling_pytorch/msa_*.py
export OPENAI_API_KEY=...   # if running the GPT pipelines
export ANTHROPIC_API_KEY=... # if running Claude pipelines
```

In source you may see literal placeholder tokens such as
`<GEMINI_API_KEY_PLACEHOLDER>` — replace these with your real key (or refactor the
caller to read `os.environ["GEMINI_API_KEY"]`).

## Reproducing Results

Trained LoRA adapters and base model weights are NOT shipped. To reproduce:

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
       --ckpt_dir torchtune/ckpt/llama3_8B/pyro_rej_lora_dist_r8_all_seed1_bracket/epoch_2 \
       --output_file data_evaluation/results/openestimate/pyrorej_all_s1_bracket_openestimate.json

   # Bayesian Teaching
   python data_evaluation/evaluate_bayesian_teaching.py \
       --ckpt_dir torchtune/ckpt/llama3_8B/pyro_rej_lora_dist_r8_all_seed1_bracket/epoch_2 \
       --output_file data_evaluation/results/bayesian_teaching/pyrorej_all_s1_bracket_bayesian_teaching_guided.json
   ```
4. Aggregate calibration tables:
   ```bash
   python data_evaluation/aggregate_pyrorej_all_bracket_calibration.py
   ```

The evaluation scripts write their per-query outputs as JSON under
`data_evaluation/results/`; the aggregator then reads those JSONs to build the
calibration tables. Result files are not shipped — run the steps above to
regenerate them.

## Anonymization Notes

This release is anonymized for double-blind review. Original absolute paths and
identifying information have been replaced with relative paths or placeholders
(`<DATA_ROOT>/`, `anonymous@example.com`, `<*_API_KEY_PLACEHOLDER>`). The full,
non-anonymized code will be released on acceptance.
