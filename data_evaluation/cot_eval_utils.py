"""cot_eval_utils.py

Shared helpers for the *chain-of-thought* (free-generation) baselines on the
three out-of-domain benchmark families:

    evaluate_text_classification_cot.py   (MMLU / TruthfulQA / HellaSwag /
                                           ARC-C / Winogrande / cls45 /
                                           chembench / legalbench)
    evaluate_bayesian_teaching_cot.py     (BT: flight / hotel / webshop)
    evaluate_openestimate_cot.py          (OE: 0-100 estimation)

Protocol
--------
The existing baselines score the answer token *immediately* after the
assistant header, i.e. the model never gets to reason.  The CoT baseline
inserts self-generated reasoning between the header and the answer:

    1. Rewrite the prompt so reasoning is explicitly permitted / requested.
    2. Free-generate up to ``max_new_tokens`` (greedy by default).
    3. Strip the model's *own* final answer from the generation (if it
       emitted one) so that exactly one answer slot exists.
    4. Append the canonical answer cue ``"\\n\\nThe answer is: <"`` and run a
       single forward pass; the restricted softmax at the last position is
       the model's answer distribution.

Step 4 is what makes CoT comparable to the non-CoT numbers: the answer is
read off the *same* restricted softmax (letters A..Z / choices 1-3 /
integers 0-100) as in the teacher-forced evals, so accuracy, NLL and ECE
are all defined and directly comparable.  The answer parsed straight out of
the free generation is also recorded (``parsed_*`` / ``valid``) as a
sanity check — the two agree on the overwhelming majority of examples.

This mirrors ``evaluate_bayesian_teaching_tf_reasoning.py``, which does the
same thing with *ground-truth* reasoning traces instead of self-generated
ones.
"""

from __future__ import annotations

import contextlib
import re
import traceback
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from torchtune.generation import generate as _tt_generate
from torchtune.modules.common_utils import disable_kv_cache


# The cue appended after the reasoning.  Kept byte-identical to the
# ``ASSISTANT_PREFILL`` used by evaluate_text_classification.py and to the
# repair suffix used by evaluate_bayesian_teaching.py so that the answer is
# read from the same token position/context as in the non-CoT evals.
ANSWER_CUE = "\n\nThe answer is: <"

# Phrases the model typically writes just before its own final answer; used
# to cut the self-generated answer off the end of the reasoning.
_LEAD_IN_RE = re.compile(
    r"(?:\*\*|##)?\s*"
    r"(?:so|and so|therefore|thus|hence|in conclusion|finally)?[\s,]*"
    r"(?:the\s+(?:final\s+)?answer\s+is|my\s+answer\s+is|answer\s*:)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)


# ── KV-cached generation ──────────────────────────────────────────────────────

def setup_generation_caches(model, dtype: torch.dtype, max_total_len: int,
                            device=None) -> bool:
    """Allocate KV caches for batch-size-1 incremental decoding.

    Returns True on success.  Generation is O(n) per token with caches and
    O(n^2) without, which matters a lot for 256-512-token CoT traces, but a
    failure here is not fatal — the caller falls back to the naive loop.

    The ``torch.device`` context is required: ``KVCache`` allocates its buffers
    with a bare ``torch.zeros(cache_shape, dtype=dtype)``, so without it the
    caches land on CPU while the model sits on GPU and every forward pass dies
    with "found at least two devices".
    """
    if device is None:
        device = next(model.parameters()).device
    try:
        with torch.device(device):
            model.setup_caches(
                batch_size=1, dtype=dtype, decoder_max_seq_len=max_total_len,
            )
        return bool(model.caches_are_enabled())
    except Exception as exc:  # pragma: no cover - device/arch dependent
        print(f"  [WARNING] KV-cache setup failed ({exc}); "
              f"falling back to uncached generation (slow).")
        return False


def _caches_are_setup(model) -> bool:
    try:
        return bool(model.caches_are_setup())
    except Exception:
        return False


@contextlib.contextmanager
def _no_cache(model):
    """Run a full-sequence forward pass without touching the KV caches.

    A model with caches *set up* rejects any forward pass that does not
    supply an explicit mask + input_pos, so every full-sequence call has to
    go through here — including the "uncached" generation fallback.
    """
    if _caches_are_setup(model) and model.caches_are_enabled():
        with disable_kv_cache(model):
            yield
    else:
        yield


def causal_mask_if_needed(model, seq_len: int, device) -> Optional[torch.Tensor]:
    """An explicit ``[1, s, s]`` causal mask, but only when one is required.

    ``MultiHeadAttention.forward`` decides whether to apply causal masking with

        is_causal = self.kv_cache is None and mask is None and self.is_causal

    so the moment KV caches have been *set up*, a ``mask=None`` forward pass
    silently becomes **non-causal**: every position attends to every other one,
    including future tokens.  ``disable_kv_cache`` does not save us — it clears
    ``cache_enabled``, not ``self.kv_cache``.

    This matters because the CoT evals set up caches for generation and then
    reuse the same model for full-sequence scoring passes.  Without an explicit
    mask those scoring passes let the answer position attend forward into
    nothing (it is last) but let every *earlier* position attend to the future,
    corrupting the hidden states the answer is read from.

    Returns None when no cache is set up, so the fast ``is_causal=True`` flash
    path is kept for the common case.
    """
    if not _caches_are_setup(model):
        return None
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    ).unsqueeze(0)


def _safe_pad_id(prompt_ids: Sequence[int], vocab_size: Optional[int]) -> int:
    """A token id that does **not** occur in *prompt_ids*.

    ``torchtune.generation.generate`` derives a padding mask from ``pad_id``
    (``generated_tokens != pad_id``) to support batches of ragged prompts.  At
    batch size 1 there is no padding, but if ``pad_id`` happens to occur in the
    prompt those positions are silently treated as padding: they are masked out
    of attention *and* every later position id is shifted down by one.  The
    obvious choice of ``eos_id`` is the worst possible one, because the chat
    template puts ``<|eot_id|>`` in the middle of every prompt.

    Reserved special-token ids are tried first; they are in-vocabulary but
    never produced by the tokenizer from ordinary text.
    """
    present = set(int(t) for t in prompt_ids)
    for cand in (128255, 128254, 128253, 0, 1, 2):
        if cand not in present and (vocab_size is None or cand < vocab_size):
            return cand
    for cand in range(vocab_size or 128256):
        if cand not in present:
            return cand
    raise ValueError("Prompt covers the entire vocabulary; no free pad id.")


@torch.no_grad()
def free_generate(
    model,
    prompt_ids: Sequence[int],
    device: str,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    caches_on: bool,
    temperature: float = 1.0,
    top_k: Optional[int] = 1,
    rng: Optional[torch.Generator] = None,
) -> List[int]:
    """Generate up to *max_new_tokens* continuation tokens.

    Defaults to greedy decoding (``top_k=1`` collapses the sampling
    distribution onto the argmax).  Returns the generated ids with any
    trailing EOS / padding removed.
    """
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)

    if caches_on:
        model.reset_caches()
        vocab = getattr(getattr(model, "tok_embeddings", None),
                        "num_embeddings", None)
        tokens, _ = _tt_generate(
            model,
            prompt,
            max_generated_tokens=max_new_tokens,
            pad_id=_safe_pad_id(prompt_ids, vocab),
            temperature=temperature,
            top_k=top_k,
            stop_tokens=list(eos_ids),
            rng=rng,
        )
        gen = tokens[0, len(prompt_ids):].tolist()
    else:
        gen = []
        generated = prompt
        eos_set = set(int(t) for t in eos_ids)
        with _no_cache(model):
            for _ in range(max_new_tokens):
                logits = model(
                    tokens=generated,
                    mask=causal_mask_if_needed(
                        model, generated.shape[1], device),
                )
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)
                last = logits[:, -1, :]
                if top_k == 1:
                    nxt = torch.argmax(last, dim=-1, keepdim=True)
                else:
                    scaled = last.float() / max(temperature, 1e-5)
                    if top_k is not None:
                        v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                        scaled = torch.where(scaled < v[:, -1:], -float("inf"), scaled)
                    nxt = torch.multinomial(F.softmax(scaled, dim=-1), 1,
                                            generator=rng)
                tok = int(nxt.item())
                if tok in eos_set:
                    break
                gen.append(tok)
                generated = torch.cat([generated, nxt], dim=1)
        return gen

    # Trim at the first stop token (generate() pads the tail with pad_id).
    eos_set = set(int(t) for t in eos_ids)
    for i, t in enumerate(gen):
        if int(t) in eos_set:
            return gen[:i]
    return gen


# ── restricted scoring ────────────────────────────────────────────────────────

@torch.no_grad()
def last_position_logits(
    model,
    input_ids: Sequence[int],
    device: str,
    max_seq_len: int,
) -> torch.Tensor:
    """Full forward pass over *input_ids*; return the logit vector that
    predicts the *next* token (i.e. logits at the final position)."""
    ids = list(input_ids)
    if len(ids) > max_seq_len:
        ids = ids[-max_seq_len:]
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    mask = causal_mask_if_needed(model, len(ids), device)
    with _no_cache(model):
        logits = model(tokens=inp, mask=mask)
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=1)
    return logits[0, -1, :].float()


def restricted_probs(logit_vec: torch.Tensor,
                     choice_ids: Sequence[int]) -> np.ndarray:
    """Softmax over a restricted set of token ids."""
    idx = torch.tensor(list(choice_ids), dtype=torch.long,
                       device=logit_vec.device)
    return F.softmax(logit_vec[idx], dim=-1).cpu().numpy().astype(np.float64)


# ── self-consistency ──────────────────────────────────────────────────────────

# Wang et al. (ICLR 2023) sample k reasoning paths at temperature and
# marginalise them out by majority vote over the final answers.  Defaults below
# are theirs.
SC_TEMPERATURE = 0.7
SC_TOP_K = 40


def sc_seed(base_seed: int, global_idx: int, k: int) -> int:
    """Deterministic seed for sample *k* of dataset example *global_idx*.

    Keyed on the absolute dataset index, not the position within a shard, so a
    given example draws the same reasoning paths no matter how the benchmark is
    split across jobs — otherwise re-sharding would silently change results.
    """
    return (int(base_seed) * 1_000_003 + int(global_idx) * 9_176 + int(k)) % (2 ** 31 - 1)


def marginalize(prob_list: Sequence[np.ndarray]) -> np.ndarray:
    """p(answer) = (1/k) Σ_k p(answer | reasoning path k).

    Averaging the per-path restricted softmaxes — rather than counting votes —
    is what makes NLL and ECE well defined.  Vote proportions are granular to
    1/k and assign exactly zero to an answer no path chose, which sends NLL to
    infinity; the mixture never does.  Where the per-path distributions are
    near one-hot (they are: agree_rate ≈ 0.99) the argmax of this mixture and
    the majority vote coincide anyway, so this keeps canonical behaviour while
    staying scoreable.
    """
    return np.mean(np.stack(list(prob_list), axis=0), axis=0)


def majority_vote(preds: Sequence[Optional[int]],
                  probs_mean: Optional[np.ndarray] = None) -> Optional[int]:
    """Canonical self-consistency decision: the answer most paths reached.

    *preds* are 0-based class indices; None entries (a path whose answer could
    not be parsed) are skipped rather than counted as a class, matching the
    original formulation. Ties are broken by the marginal probability so the
    decision stays deterministic. Returns None only if no path produced an
    answer.
    """
    votes: dict = {}
    for p in preds:
        if p is not None:
            votes[int(p)] = votes.get(int(p), 0) + 1
    if not votes:
        return None
    top = max(votes.values())
    tied = [c for c, v in votes.items() if v == top]
    if len(tied) == 1:
        return tied[0]
    if probs_mean is not None:
        return max(tied, key=lambda c: float(probs_mean[c]))
    return min(tied)


def vote_distribution(preds: Sequence[Optional[int]], n_classes: int) -> list:
    """Raw vote proportions over *n_classes* — reported alongside the mixture
    so the canonical self-consistency quantity is recoverable, but not used for
    NLL (it is zero-heavy by construction)."""
    counts = np.zeros(n_classes, dtype=np.float64)
    n = 0
    for p in preds:
        if p is not None:
            counts[int(p)] += 1.0
            n += 1
    return (counts / n).tolist() if n else counts.tolist()


def binned_ece(confidences: Sequence[float], correct: Sequence[float],
               n_bins: int = 15) -> float:
    """Standard equal-width binned ECE. Matches evaluate_text_classification."""
    conf = np.asarray(list(confidences), dtype=np.float64)
    corr = np.asarray(list(correct), dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = ((conf >= lo) & (conf <= hi)) if i == n_bins - 1 else \
            ((conf >= lo) & (conf < hi))
        if m.sum() == 0:
            continue
        ece += (m.sum() / conf.size) * abs(corr[m].mean() - conf[m].mean())
    return float(ece)


def sc_extras(items: List[dict]) -> dict:
    """Self-consistency diagnostics: how much do the sampled paths disagree?

    Also reports ``vote_ece`` — calibration using the *fraction of reasoning
    paths backing the chosen answer* as the confidence.  This is the
    calibration number that means something for self-consistency: the answer
    read off a single appended cue is a near-deterministic echo of what the
    model already wrote (agree_rate ~ 0.99), so its confidence carries almost
    no information, whereas how often k independent paths converge does.
    Granularity is 1/k, so k needs to be reasonably large for the bins to be
    informative.
    """
    if not items or not any("n_samples" in it for it in items):
        return {}
    agree_frac = [it["vote_margin"] for it in items if it.get("vote_margin") is not None]
    flipped = [it for it in items
               if it.get("vote_pred") is not None
               and it.get("pred_idx_marginal") is not None
               and it["vote_pred"] != it["pred_idx_marginal"]]
    scored = [it for it in items
              if it.get("vote_correct") is not None
              and it.get("vote_margin") is not None]
    out = {
        "n_samples": int(items[0].get("n_samples", 1)),
        # Fraction of sampled paths backing the winning answer: 1.0 means every
        # path agreed, 1/k means maximal disagreement.
        "mean_vote_margin": float(np.mean(agree_frac)) if agree_frac else float("nan"),
        # How often majority vote and the mixture argmax disagree. Large values
        # mean the two aggregation rules are not interchangeable here.
        "vote_vs_marginal_disagree": len(flipped) / len(items),
    }
    if scored:
        out["vote_ece"] = binned_ece(
            [it["vote_margin"] for it in scored],
            [float(bool(it["vote_correct"])) for it in scored],
        )
    return out


# ── reasoning post-processing ─────────────────────────────────────────────────

def strip_self_answer(
    text: str,
    answer_re: re.Pattern,
    tail_window: int = 80,
) -> Tuple[str, Optional[str]]:
    """Cut the model's own final answer off the end of its reasoning.

    Returns ``(reasoning, raw_answer_string_or_None)``.  The cut also removes
    an immediately preceding "the answer is" style lead-in so the cue we
    append afterwards is not duplicated.

    Only an answer that sits within *tail_window* characters of the end of
    the generation is treated as the model's final commitment.  Bracketed
    tokens that appear mid-reasoning — "option <A> says …", or a tentative
    guess the model then revises — are left in place, as is any reasoning
    that was cut off by the token budget before an answer was reached.
    """
    matches = list(answer_re.finditer(text))
    if not matches:
        return text.rstrip(), None

    m = matches[-1]
    if len(text.rstrip()) - m.end() > tail_window:
        return text.rstrip(), None

    cut = m.start()
    head = text[:cut]
    lead = _LEAD_IN_RE.search(head)
    if lead is not None and lead.end() == len(head):
        cut = lead.start()
    return text[:cut].rstrip(), m.group(0)


def build_scoring_context(
    tokenizer,
    prefix_ids: Sequence[int],
    reasoning: str,
    cue: str = ANSWER_CUE,
) -> List[int]:
    """prefix (chat header) + reasoning + answer cue, as token ids."""
    tail = tokenizer.encode(reasoning + cue, add_special_tokens=False)
    return list(prefix_ids) + list(tail)


class FailureTracker:
    """Print the first failure in full and abort a run that is failing wholesale.

    Per-example ``except`` blocks keep one bad example from killing a shard,
    but they also hide a systematic error: without this, a misconfigured run
    logs a one-line warning per example and exits with "No results", giving no
    way to tell a data problem from a broken model call.
    """

    def __init__(self, abort_after: int = 5):
        self.abort_after = abort_after
        self.n_failed = 0

    def record(self, i: int, exc: BaseException, n_ok: int) -> None:
        self.n_failed += 1
        if self.n_failed == 1:
            print(f"  [ERROR] eval failed on example {i}; traceback follows. "
                  f"Subsequent failures are logged as one-liners.")
            traceback.print_exc()
        else:
            print(f"  [WARNING] eval failed on example {i}: {exc}")
        if n_ok == 0 and self.n_failed >= self.abort_after:
            raise SystemExit(
                f"Aborting: the first {self.n_failed} examples all failed, so "
                f"this is a systematic error rather than bad data.")


def eos_token_ids(tokenizer) -> List[int]:
    """EOS ids to stop generation on.  Llama-3-Instruct emits ``<|eot_id|>``
    rather than ``<|end_of_text|>`` at the end of an assistant turn, so both
    are needed; Qwen-2 uses ``<|im_end|>``."""
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for name in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(name)
        # convert_tokens_to_ids falls back to the unk id for out-of-vocab
        # tokens on some tokenizers — round-trip to confirm the id is real.
        if tid is None or tid < 0 or int(tid) in ids:
            continue
        if tokenizer.convert_ids_to_tokens(int(tid)) != name:
            continue
        ids.append(int(tid))
    return ids or [0]
