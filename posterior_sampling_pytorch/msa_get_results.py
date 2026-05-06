import os
import json
import numpy as np


def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    s = ''
    for l in lines:
        s += l
    return s

def summarize_results(filenames, query_accumulative_list, len_query):
    """Return (means, stds) pooled across all runs and files."""

    sum_wx  = [0.0] * len_query
    sum_wx2 = [0.0] * len_query
    sum_w   = [0.0] * len_query

    for file_num, file in enumerate(filenames):
        for run in range(20):
            result_filename = f'inference_results/{file}-{run}.json'

            with open(result_filename, 'r') as f:
                data = json.load(f)

            results = {}
            for key in data['support'][0]:
                results[key] = {}

            for i in range(len(data['probs'])):
                for query in results:
                    query_value = data['support'][i][query]
                    if isinstance(query_value, int) or isinstance(query_value, float):
                        pass
                    else:
                        for key in query_value:
                            query_value = query_value[key]
                            break
                    if query_value not in results[query]:
                        results[query][query_value] = data['probs'][i]
                    else:
                        results[query][query_value] += data['probs'][i]

            for i, key in enumerate(results):
                slot = i + query_accumulative_list[file_num]
                for val, prob in results[key].items():
                    sum_wx[slot]  += prob * val
                    sum_wx2[slot] += prob * val * val
                    sum_w[slot]   += prob

    means, stds = [], []
    for slot in range(len_query):
        w    = sum_w[slot]
        mean = sum_wx[slot] / w
        var  = sum_wx2[slot] / w - mean ** 2
        means.append(mean)
        stds.append(np.sqrt(max(var, 0.0)))

    return means, stds


def summarize_pytorch_results(filenames, query_accumulative_list, len_query, num_runs=1):
    """
    Read PyTorch/Pyro inference result JSONs and return (means, stds) pooled across runs.

    Each result JSON has the format produced by pg-*-pytorch.py:
      {"query1": {"samples": [...], "mean": ..., "std": ...}, "query2": {...}, ...}

    filenames: list of base filenames (without .json and without run index)
    query_accumulative_list: starting slot index for each filename
    len_query: total number of query slots
    num_runs: if > 1, files are expected as {base}-{run}.json for run in range(num_runs);
              if 1, a single {base}.json is read per filename.
    """
    all_samples = [[] for _ in range(len_query)]

    for file_num, file in enumerate(filenames):
        if num_runs > 1:
            run_files = [f'inference_results/{file}-{run}.json' for run in range(num_runs)]
        else:
            run_files = [f'inference_results/{file}.json']

        for result_filename in run_files:
            with open(result_filename, 'r') as f:
                data = json.load(f)

            query_keys = sorted(k for k in data if k.startswith('query'))
            for i, key in enumerate(query_keys):
                slot = i + query_accumulative_list[file_num]
                all_samples[slot].extend(data[key]['samples'])

    means, stds = [], []
    for slot in range(len_query):
        s = np.array(all_samples[slot])
        means.append(float(s.mean()))
        stds.append(float(s.std()))

    return means, stds


query_accu_list = [0,8,12]
filenames = ['sports-3benchmarks/result-biathlon-2025-12-26-230750', 'sports-3benchmarks/result-canoe-2025-12-26-230750', 'sports-3benchmarks/result-tugofwar-2025-12-26-230750']
pg_means, pg_stds = summarize_results(filenames, query_accu_list, 18)
print(pg_means, pg_stds)

query_accu_list = [0,8,12]
filenames = ['sports-3benchmarks/result-benchmark-rejection-biathlon-2026-01-08-204305', 'sports-3benchmarks/result-benchmark-rejection-canoe-2026-01-08-204305', 'sports-3benchmarks/result-benchmark-rejection-tugofwar-2026-01-08-204305']
ben_means, ben_stds = summarize_results(filenames, query_accu_list, 18)
print(ben_means, ben_stds)

import matplotlib.pyplot as plt

def scatter_means_with_se_bins(
    x_means, x_stds,
    y_means, y_stds,
    bins=(0, 8, 12, 18),
    name='plot',
    ax=None,
    marker="o",
    capsize=3,
    alpha=0.9,
    error_alpha=0.3,
    s=40,
    x_label='Benchmark MSA',
    y_label='Dependent Variable',
    bin_labels=None,
    legend_title="Index bins",
    spine_lw=1.0,
    tick_lw=1.0,
    tick_length=4.0,
    label_fontsize=None,
    tick_fontsize=None,
    legend_fontsize=None,
    legend_title_fontsize=None,
    identity_lw=1.0,
):
    """
    x_means, x_stds: lists/arrays of length N
    y_means, y_stds: lists/arrays of length N
    bins: indices that partition points into groups, e.g. (0,8,12,18)
          meaning [0,8), [8,12), [12,18) (by index, not by value).
    name: filename stem used when saving the figure.
    """

    x_means = np.asarray(x_means, dtype=float)
    x_stds  = np.asarray(x_stds,  dtype=float)
    y_means = np.asarray(y_means, dtype=float)
    y_stds  = np.asarray(y_stds,  dtype=float)

    if not (len(x_means) == len(x_stds) == len(y_means) == len(y_stds)):
        raise ValueError("All input lists must have the same length.")

    N = len(x_means)
    if max(bins) > N:
        raise ValueError(f"bins max ({max(bins)}) cannot exceed number of points ({N}).")

    # ---- scale items with mean < 1 ----
    x_mask = x_means < 1
    y_mask = y_means < 1

    x_means[x_mask] *= 100
    x_stds[x_mask]  *= 100

    y_means[y_mask] *= 100
    y_stds[y_mask]  *= 100

    x_se = x_stds
    y_se = y_stds

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Colors per interval (distinct)
    # Use a simple colormap; you can replace with custom colors if you want.
    cmap = plt.get_cmap("tab10")
    interval_colors = [cmap(i) for i in range(len(bins) - 1)]
    interval_colors = ['blue', 'purple', 'red']

    # Plot each bin interval with its own color
    for k in range(len(bins) - 1):
        lo, hi = bins[k], bins[k + 1]
        if lo >= hi:
            continue

        label = bin_labels[k] if bin_labels is not None else f"[{lo},{hi})"
        eb = ax.errorbar(
            x_means[lo:hi],
            y_means[lo:hi],
            xerr=x_se[lo:hi],
            yerr=y_se[lo:hi],
            fmt=marker,
            linestyle="none",
            capsize=capsize,
            alpha=alpha,
            markersize=np.sqrt(s),  # roughly match scatter "s" feel
            color=interval_colors[k],
            label=label,
        )
        for capline in eb[1]:
            capline.set_alpha(error_alpha)
        for barline in eb[2]:
            barline.set_alpha(error_alpha)

    # Identity line (y = x), based on current data range with padding
    all_x = np.concatenate([x_means, y_means])
    finite = np.isfinite(all_x)
    if finite.any():
        lo = float(np.min(all_x[finite]))
        hi = float(np.max(all_x[finite]))
    else:
        lo, hi = 0.0, 1.0

    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo2, hi2 = lo - pad, hi + pad
    ax.plot([0, 100], [0, 100], linewidth=identity_lw, color='black')

    # Square plot + equal data scaling
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    leg = ax.legend(title=legend_title, frameon=False, fontsize=legend_fontsize)
    if legend_title_fontsize is not None and leg.get_title() is not None:
        leg.get_title().set_fontsize(legend_title_fontsize)
    ax.grid(True, alpha=0.2)

    # Spine + tick styling
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)
    ax.tick_params(axis='both', which='major', width=tick_lw, length=tick_length,
                   labelsize=tick_fontsize)

    fig.savefig(f'{name}.png', bbox_inches='tight')
    fig.savefig(f'{name}.pdf', bbox_inches='tight')
    return fig, ax

llama3_8b_mean = [85, 90, 70, 92, 88, 80, 55, 60, 50, 60, 40, 70, 85, 78, 60, 80, 55, 65]

llama3_8b_se = [4,3,5,2,3,4,5,6,5,10,5,5,10,12,8,5,15,18]

scatter_means_with_se_bins(ben_means, ben_stds, llama3_8b_mean, llama3_8b_se,
                           bins=[0, 8, 12, 18],
                           name='scatter_llama3_8b',
                           y_label='LLAMA3-8B')

scatter_means_with_se_bins(ben_means, ben_stds, pg_means, pg_stds,
                           bins=[0, 8, 12, 18],
                           name='scatter_our_msa',
                           y_label='Our MSA')

gemini_mean = [85, 55, 25, 35, 35, 75, 30, 48]
gemini_mean.extend([65, 90, 15, 85])
gemini_mean.extend([70, 20, 85, 85, 95, 98])

gemini_se = [10, 15, 10, 20, 20, 20, 15, 20]
gemini_se.extend([15, 10, 15, 15])
gemini_se.extend([10, 10, 15, 15, 5, 3])

scatter_means_with_se_bins(ben_means, ben_stds, gemini_mean, gemini_se,
                           bins=[0, 8, 12, 18],
                           name='scatter_gemini',
                           y_label='Gemini 3 Pro')

gpt5p1_mean = [85, 70, 75, 55, 50, 65, 40, 60]
gpt5p1_mean.extend([85, 35, 20, 70])
gpt5p1_mean.extend([72, 60, 65, 80, 65, 55])

gpt5p1_se = [5, 6, 6, 10, 10, 9, 12, 12]
gpt5p1_se.extend([7, 10, 12, 15])
gpt5p1_se.extend([10, 10, 15, 15, 20, 20])

scatter_means_with_se_bins(ben_means, ben_stds, gpt5p1_mean, gpt5p1_se,
                           bins=[0, 8, 12, 18],
                           name='scatter_gpt5p1',
                           y_label='GPT 5.1')

# ── PyTorch/Pyro program results vs benchmark MSA ─────────────────────────────
query_accu_list = [0, 8, 12]
pytorch_filenames = ['result-biathlon-pytorch', 'result-canoe-pytorch', 'result-tug-of-war-pytorch']
pytorch_means, pytorch_stds = summarize_pytorch_results(pytorch_filenames, query_accu_list, 18)
print(pytorch_means, pytorch_stds)

scatter_means_with_se_bins(ben_means, ben_stds, pytorch_means, pytorch_stds,
                           bins=[0, 8, 12, 18],
                           name='scatter_pytorch',
                           x_label='MSA',
                           y_label='meta-MSA',
                           bin_labels=['Biathlon', 'Canoe', 'Tug-of-War'],
                           legend_title='Scenario',
                           spine_lw=2.0,
                           tick_lw=2.5,
                           tick_length=8.0,
                           identity_lw=2.0,
                           label_fontsize=26,
                           tick_fontsize=20,
                           legend_fontsize=16,
                           legend_title_fontsize=18)