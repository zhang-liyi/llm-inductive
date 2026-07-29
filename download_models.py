"""
download_models.py

Downloads the two base models into ``data_root/base_models/``:

    Meta-Llama-3-8B-Instruct   (~15 GB)   revision e1945c40...
    Qwen2-7B-Instruct          (~15 GB)

Only the base models. The three fine-tuned checkpoints are not downloaded here
— they ship as ``finetuned_checkpoints.zip`` next to this file.

    python download_models.py                 # both
    python download_models.py --only llama    # or --only qwen

Llama-3 is a **gated** repo: accept the licence on its HuggingFace page once,
then authenticate before running this, either with

    huggingface-cli login          # or:  export HF_TOKEN=hf_...

The Llama revision is pinned to the exact one our results were produced with.
Leave it alone — a different revision is a different model.

### The tokenizer.model quirk

HuggingFace does not serve ``tokenizer.model`` at the top level of the Llama-3
repo; it lives under ``original/``. Some download paths therefore leave you
without it. This script fetches it explicitly and, if that fails, falls back to
``llama3_tokenizer.model`` shipped next to this file.

Neither of the two evaluations needs that file — they tokenize through
``tokenizer.json`` via ``AutoTokenizer`` — so a missing one will not block you.
It is only needed by the torchtune training configs and the healthcare eval.
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "data_root" / "base_models"

LLAMA_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_REV = "e1945c40cd546c78e41f1151f4db032b271faeaa"
LLAMA_DIR = "Meta-Llama-3-8B-Instruct"
LLAMA_FILES = [
    "config.json", "generation_config.json",
    "model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors",
    "model.safetensors.index.json", "special_tokens_map.json",
    "tokenizer_config.json", "tokenizer.json",
]

QWEN_REPO = "Qwen/Qwen2-7B-Instruct"
QWEN_DIR = "Qwen2-7B-Instruct"

# Keep 'qwen' out of the Llama directory name and in the Qwen one: the eval
# code picks the model architecture by looking for 'qwen' in the path.
TOKENIZER_BACKUP = HERE / "llama3_tokenizer.model"


def _download(repo, local_dir, revision=None, allow_patterns=None):
    from huggingface_hub import snapshot_download
    print(f"  downloading {repo}"
          f"{'' if revision is None else ' @ ' + revision[:8]} -> {local_dir}",
          flush=True)
    snapshot_download(repo_id=repo, revision=revision,
                      local_dir=str(local_dir),
                      allow_patterns=allow_patterns)


def get_llama():
    out = DEST / LLAMA_DIR
    _download(LLAMA_REPO, out, revision=LLAMA_REV, allow_patterns=LLAMA_FILES)

    tok = out / "tokenizer.model"
    if not tok.exists():
        try:
            from huggingface_hub import hf_hub_download
            src = hf_hub_download(repo_id=LLAMA_REPO, revision=LLAMA_REV,
                                  filename="original/tokenizer.model")
            shutil.copyfile(src, tok)
            print("  tokenizer.model  fetched from original/")
        except Exception as e:
            if TOKENIZER_BACKUP.exists():
                shutil.copyfile(TOKENIZER_BACKUP, tok)
                print(f"  tokenizer.model  HF fetch failed "
                      f"({type(e).__name__}); used the bundled backup")
            else:
                print(f"  tokenizer.model  NOT obtained ({type(e).__name__}). "
                      f"Harmless for these evals — see the note at the top of "
                      f"this file.")
    return out


def get_qwen():
    out = DEST / QWEN_DIR
    _download(QWEN_REPO, out)
    return out


def report(path: Path):
    if not path.exists():
        print(f"  MISSING {path}")
        return False
    files = sorted(p for p in path.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"  {path.name}: {len(files)} files, {total / 1e9:.1f} GB")
    has_cfg = (path / "config.json").exists()
    has_w = any(p.name.endswith(".safetensors") for p in files)
    if not (has_cfg and has_w):
        print(f"  INCOMPLETE — expected config.json and *.safetensors")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["llama", "qwen"], default=None)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    got = []
    if args.only in (None, "llama"):
        print("Llama-3-8B-Instruct")
        got.append(get_llama())
    if args.only in (None, "qwen"):
        print("Qwen2-7B-Instruct")
        got.append(get_qwen())

    print("\nResult")
    ok = all(report(p) for p in got)

    print(f"\nModels are under {DEST}.")
    print("run_morebench.sh already points at the Llama directory; change "
          "BASE_MODEL there if you want the Qwen row instead.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
