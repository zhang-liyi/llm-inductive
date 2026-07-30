# Fine-tuned + CoT + self-consistency

Same setup as the moral benchmark guide (`MOREBENCH_HANDOFF.md`) — same
dependencies, same `python download_data.py`, same models in `data_root/ckpt/`.
Nothing extra.

## Run it

**Seed 3 is the one to run.** Seed 1 is finished and seed 2 is likely covered
on our side, so the script defaults to seed 3 and asks for confirmation before
it will touch seed 2.

```bash
bash run_cot_sc.sh
```

40 jobs, up to ~12 h each, about a day of wall clock. Don't run a second seed
alongside it: 40 jobs fits the cluster's 44-job cap, 80 does not.

The script checks every path it needs and stops with a list if anything is
missing, before submitting.

Re-running is safe — jobs whose output already exists are skipped. If some
jobs die, just run the same command again.

## When the 40 jobs are done

```bash
python data_evaluation/combine_sc_samples.py --auto --tag cotsc_pyrorej_all_s3_bracket
```

If it warns *uneven path counts across examples*, some jobs did not finish.
Re-run the seed first, then combine.

## Don't change

`--seed 0` and the `cotsc_` tag are load-bearing (they keep this run paired
with the base-model run it is compared against). The script sets both.
