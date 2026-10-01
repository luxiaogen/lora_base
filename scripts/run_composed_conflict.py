"""Six real T10 experiments: cancellation-preserving S/P conflict gates, 3090."""
import argparse
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'scripts/sweeps/imgr10_two_expert_oracle_3090.json'
MODES = ('net', 'gross', 'magnitude', 'uniform', 'w_pre', 'random')


def settings_for(mode, smoke=False):
    settings = json.loads(SPEC.read_text())['common_overrides']
    settings.update(seed=[1993], dual_mask_composed_conflict=mode,
        dual_mask_anchor_reg_weight=2.5, wpre_distill_weight=0., ridge_fusion_enabled=False,
        two_expert_oracle_diagnostic=False, two_expert_calibration_holdout_mod=0,
        stage_audit=False, save_task_weights=False, task0_validation_enabled=False,
        incremental_holdout=False, wandb_group='imgr10_composed_conflict_3090')
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1, wandb_mode='offline')
    return settings


def command_for(mode, directory, smoke=False):
    settings = settings_for(mode, smoke)
    settings['prefix'] = 'imgr10_composed_' + mode + '_' + directory.parent.name + '_' + directory.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def read_metrics(path):
    content = path.read_text()
    totals = re.findall(r'\[trainer.py\] => CNN: (\{[^\n]+\})', content)
    averages = re.findall(r'\[trainer.py\] => Average Accuracy: ([\d.]+)', content)
    forgetting = re.findall(r'\[trainer.py\] => Forgetting:\s*([\d.-]+)', content)
    rows = [json.loads(row) for row in re.findall(r'ComposedConflict (\{[^\n]+\})', content)]
    result = dict(tasks_reported=len(totals), Average=None, Last=None, Old=None, New=None, Forgetting=None)
    if totals:
        last = ast.literal_eval(totals[-1])
        result.update(Last=last['total'], Old=last['old'], New=last['new'])
    if averages:
        result['Average'] = float(averages[-1])
    if forgetting:
        result['Forgetting'] = float(forgetting[-1])
    if rows:
        for key in ('density', 'removed_ratio', 'net_to_gross_norm',
                    'independent_amplified_coordinates', 'independent_reversed_coordinates',
                    'shared_amplified_coordinates', 'shared_reversed_coordinates'):
            result[key] = sum(row[key] for row in rows) / len(rows)
        result['max_branch_sum_error'] = max(row['branch_sum_error'] for row in rows)
    return result, rows


def summarize(directory, records):
    summaries, telemetry = [], []
    for record in records:
        if record['status'] not in ('completed', 'failed'):
            continue
        metrics, rows = read_metrics(directory / record['mode'] / 'training.log')
        summaries.append(dict(mode=record['mode'], status=record['status'], minutes=record['minutes'], **metrics))
        telemetry.extend(rows)
    for name, rows in (('results', summaries), ('gate_diagnostics', telemetry)):
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (directory / (name + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
    (directory / 'results.json').write_text(json.dumps(summaries, indent=2) + '\n')
    complete = [row for row in summaries if row['status'] == 'completed' and row['tasks_reported'] == 10]
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for row in complete:
            axes[0].scatter(row['Old'], row['New'])
            axes[0].annotate(row['mode'], (row['Old'], row['New']), xytext=(3, 4), textcoords='offset points')
        axes[0].set(xlabel='Final Old accuracy (%)', ylabel='Final New accuracy (%)', title='Old / New trade-off')
        axes[1].bar([r['mode'] for r in complete], [r['Average'] for r in complete])
        axes[1].set(ylabel='Average accuracy (%)', title='Real full-T10 results')
        axes[1].set_ylim(min(r['Average'] for r in complete) - .3, max(r['Average'] for r in complete) + .15)
        axes[1].tick_params(axis='x', rotation=35)
        axes[2].bar([r['mode'] for r in complete], [r['independent_reversed_coordinates'] for r in complete])
        axes[2].set(ylabel='Mean coordinates / layer-task', title='Independent-gate sign reversals\n(same-weight counterfactual, not accuracy)')
        axes[2].tick_params(axis='x', rotation=35)
        fig.savefig(directory / 'comparison.png', dpi=180)
        plt.close(fig)
    print('Measured results:', json.dumps(summaries), flush=True)


def run(mode, directory, revision, smoke=False, dry_run=False):
    command, settings = command_for(mode, directory, smoke)
    print('Starting:', mode, 'smoke' if smoke else 'REAL FULL T10', flush=True)
    print('Command:', shlex.join(command), flush=True)
    if dry_run:
        return 0, 0.
    directory.mkdir(parents=True)
    effective = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
    effective.update(settings)
    paths = ('models/attention.py', 'methods/dlora.py', 'scripts/run_composed_conflict.py',
             'scripts/sweeps/imgr10_two_expert_oracle_3090.json')
    (directory / 'run.json').write_text(json.dumps(dict(code_revision=revision,
        effective_config=effective, command=command,
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}), indent=2) + '\n')
    started = time.monotonic()
    with (directory / 'training.log').open('w') as stream:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
    minutes = (time.monotonic() - started) / 60
    print('Finished:', mode, 'exit_code=', code, 'minutes=', round(minutes, 1), flush=True)
    return code, minutes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'smoke', 't10', 'dry-run'), default='run')
    parser.add_argument('--hours', type=float, default=8., help='No new run after this time; never kills training')
    parser.add_argument('--only', nargs='+', choices=MODES)
    args = parser.parse_args()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = ROOT / 'logs/shell_logs/imgr10_composed_conflict_3090' / stamp
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('Code revision:', revision, '\nOutputs:', directory, flush=True)
    print('3090 ONLY; seed1993; anchor2.5; T10; 20 epochs; CA5; math-SDPA. '
          'No checkpoints, teacher, replay, holdout tuning or task-ID inference.', flush=True)
    print('All six jobs are trained candidates/controls, NOT a baseline rerun. '
          'Two 1-epoch smoke checks do not count as performance evidence.', flush=True)
    dry_run = args.mode == 'dry-run'
    started = time.monotonic()
    if not dry_run:
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps(dict(revision=revision,
            options=vars(args), modes=list(args.only or MODES)), indent=2) + '\n')
    if args.mode in ('run', 'smoke', 'dry-run'):
        for mode in ('net', 'uniform'):
            code, _ = run(mode, directory / ('smoke_' + mode), revision, smoke=True, dry_run=dry_run)
            if code:
                print('GPU smoke failed; full queue not started.', flush=True)
                return code
        if args.mode == 'smoke':
            return 0
    records = []
    for mode in args.only or MODES:
        if not dry_run and time.monotonic() - started >= args.hours * 3600:
            print('Time budget: not starting', mode, '(not a performance rejection)', flush=True)
            records.append(dict(mode=mode, status='time_budget_pending'))
        else:
            code, minutes = run(mode, directory / mode, revision, dry_run=dry_run)
            records.append(dict(mode=mode, status='completed' if code == 0 else 'failed', exit_code=code, minutes=minutes))
            if code:
                (directory / 'queue.json').write_text(json.dumps(records, indent=2))
                summarize(directory, records)
                return code
        if not dry_run:
            (directory / 'queue.json').write_text(json.dumps(records, indent=2))
            summarize(directory, records)
    print('Queue finished:', directory, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
