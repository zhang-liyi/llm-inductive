"""Apply Temperature Scaling to all 8 calibration eval JSONs for the
pyro-rej-all-bracket family × 3 seeds = 24 applies (CPU-only, fast).

Inputs (from the fit step):
  results/text_cls/ts_fit_{mmlu, hellaswag, winogrande, arc_challenge}_pyrorej_all_s{S}_bracket.json
  results/bayesian_teaching/ts_fit_bt_pyrorej_all_s{S}_bracket.json

Eval JSONs (from the calibration eval suite):
  results/text_cls/pyrorej_all_s{S}_bracket_{ds}.json    for ds in mmlu, hellaswag_h1, hellaswag_h2, winogrande, arc_challenge, truthfulqa
  results/bayesian_teaching/pyrorej_all_s{S}_bracket_{bayesian_teaching_base_guided, bt_base_tf}.json

Outputs (TS-applied):
  results/text_cls/ts_pyrorej_all_s{S}_bracket_{ds}.json
  results/bayesian_teaching/ts_pyrorej_all_s{S}_bracket_{...}.json

TruthfulQA uses T=1 (no fit).
HellaSwag h1+h2 share the same fit (ts_fit_hellaswag_<tag>.json).
BT-guided + BT-nonG share the same fit (ts_fit_bt_<tag>.json).
"""
import os
import subprocess
import sys

ROOT = './data_evaluation'
RES = f'{ROOT}/results'
SEEDS = (1, 2, 3)
APPLY = f'{ROOT}/temperature_scaling_apply.py'


def cmd_apply(eval_file, output_file, fit_file=None, T=None):
    args = ['python3', APPLY, '--eval_file', eval_file, '--output_file', output_file]
    if fit_file:
        args += ['--fit_file', fit_file]
    elif T is not None:
        args += ['--T', str(T)]
    return args


def main():
    failures = []
    for seed in SEEDS:
        tag = f'pyrorej_all_s{seed}_bracket'
        # text_cls datasets
        for ds, fit_ds in (
            ('mmlu',          'mmlu'),
            ('truthfulqa',    None),       # T=1
            ('hellaswag_h1',  'hellaswag'),
            ('hellaswag_h2',  'hellaswag'),
            ('winogrande',    'winogrande'),
            ('arc_challenge', 'arc_challenge'),
        ):
            eval_file = f'{RES}/text_cls/{tag}_{ds}.json'
            out_file  = f'{RES}/text_cls/ts_{tag}_{ds}.json'
            if fit_ds is None:
                args = cmd_apply(eval_file, out_file, T=1.0)
            else:
                fit_file = f'{RES}/text_cls/ts_fit_{fit_ds}_{tag}.json'
                args = cmd_apply(eval_file, out_file, fit_file=fit_file)
            print(f'  applying TS: {tag}  {ds}')
            try:
                subprocess.check_call(args)
            except subprocess.CalledProcessError as e:
                failures.append((tag, ds, e))
                print(f'    FAILED: {e}')

        # bayesian_teaching: 2 evals share one fit
        bt_fit = f'{RES}/bayesian_teaching/ts_fit_bt_{tag}.json'
        for eval_name in ('bayesian_teaching_base_guided', 'bt_base_tf'):
            eval_file = f'{RES}/bayesian_teaching/{tag}_{eval_name}.json'
            out_file  = f'{RES}/bayesian_teaching/ts_{tag}_{eval_name}.json'
            args = cmd_apply(eval_file, out_file, fit_file=bt_fit)
            print(f'  applying TS: {tag}  {eval_name}')
            try:
                subprocess.check_call(args)
            except subprocess.CalledProcessError as e:
                failures.append((tag, eval_name, e))
                print(f'    FAILED: {e}')

    if failures:
        print(f'\n{len(failures)} apply step(s) failed')
        sys.exit(1)
    print(f'\nAll TS applies done.')


if __name__ == '__main__':
    main()
