"""Time-budgeted finite branch-choice reference/proxy queue, seed1993."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def settings_for(spec, variant, smoke=False):
    settings = dict(spec['common_overrides'])
    settings.update(variant['overrides'])
    settings['seed'] = [1993]
    settings['branch_choice_scope'] = spec['scope']
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1,
                        wandb_mode='offline', branch_choice_interval=1)
    return settings


def command_for(spec, variant, stamp, smoke=False):
    settings = settings_for(spec, variant, smoke)
    settings['prefix'] = f"{spec['name']}_{variant['name']}_{stamp}"
    command = [sys.executable, 'main.py', '--config', spec['datasets'][0]['config']]
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def estimate_seconds(variant, observed, initial_minutes):
    tasks = variant['overrides']['max_tasks']
    interval = variant['overrides'].get('branch_choice_interval', 5)
    mode = variant['overrides']['branch_choice_mode']
    key = (tasks, interval, mode)
    if key in observed:
        return observed[key] * 1.15
    full = [seconds for (n, stride, _), seconds in observed.items() if n == 10 and stride == interval]
    if full:
        estimate = max(full) * (1. if tasks == 10 else .4)
    elif (3, 5, 'audit') in observed:
        estimate = observed[(3, 5, 'audit')] * (4. if tasks == 10 else 1.)
    else:
        estimate = initial_minutes * 60 * (1. if tasks == 10 else .35)
    if interval == 1:
        estimate *= 2.5  # reserve: five times as many counterfactual checks
    return estimate * 1.15


def fits_budget(elapsed, estimate, hours):
    return elapsed + estimate <= hours * 3600


def run(spec, variant, directory, smoke, dry_run):
    command, settings = command_for(spec, variant, directory.name, smoke)
    print(f"Starting {variant['name']}; mode={settings['branch_choice_mode']} "
          f"scope={spec['scope']} tasks={settings['max_tasks']}", flush=True)
    print('Command: ' + shlex.join(command), flush=True)
    if dry_run:
        return 0, 0.
    path = directory / f"{variant['name']}.log"
    started = time.monotonic()
    with path.open('w') as stream:
        stream.write('Command: ' + shlex.join(command) + '\n')
        stream.write('Settings: ' + json.dumps(settings) + '\n')
        stream.flush()
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        stream.write(f'ExitCode: {code}\n')
    duration = time.monotonic() - started
    print(f"Finished {variant['name']}: exit_code={code}; minutes={duration / 60:.1f}; log={path}", flush=True)
    return code, duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('machine', choices=('3090', '5090'))
    parser.add_argument('--hours', type=float, help='Start-next budget; never kills a running experiment')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--only', nargs='+', help='Run named variants, e.g. remaining jobs after a budget skip')
    args = parser.parse_args()
    spec = json.loads((ROOT / f'scripts/sweeps/imgr10_branch_choice_{args.machine}.json').read_text())
    hours = spec['hours'] if args.hours is None else args.hours
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = ROOT / spec['log_dir'] / stamp
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    print(f'Code revision: {revision}\n{status}\nOutputs: {directory}', flush=True)
    print(f'Budget: {hours} hours; Oracle/audit privileged, legal modes current-train-only; no checkpoints.', flush=True)
    if not args.dry_run:
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps(dict(
            revision=revision, status=status, spec=spec, options=vars(args)), indent=2))
    started, observed, records = time.monotonic(), {}, []
    smoke_jobs = [dict(name='smoke_' + mode, overrides=dict(branch_choice_mode=mode, max_tasks=2))
                  for mode in ('audit', 'oracle', 'logit', 'feature')]
    for job in smoke_jobs:
        code, duration = run(spec, job, directory, True, args.dry_run)
        if code:
            print('Smoke failed; full queue not started. No accuracy-equality assertion.', flush=True)
            return code
    if args.smoke:
        return 0
    jobs = [v for v in spec['variants'] if args.only is None or v['name'] in args.only]
    failed = 0
    for job in jobs:
        elapsed = time.monotonic() - started
        estimate = estimate_seconds(job, observed, spec['initial_t10_minutes'])
        if not args.dry_run and not fits_budget(elapsed, estimate, hours):
            print(f"Budget skip {job['name']}: estimated {estimate / 60:.1f} min; "
                  f"remaining {(hours * 3600 - elapsed) / 60:.1f} min. Not a performance rejection.", flush=True)
            records.append(dict(variant=job['name'], status='budget_skipped', estimate_seconds=estimate))
            continue
        code, duration = run(spec, job, directory, False, args.dry_run)
        records.append(dict(variant=job['name'], status='completed' if code == 0 else 'failed',
                            exit_code=code, seconds=duration))
        if code == 0 and not args.dry_run:
            settings = settings_for(spec, job)
            observed[(settings['max_tasks'], settings['branch_choice_interval'],
                      settings['branch_choice_mode'])] = duration
        failed = failed or code
        if not args.dry_run:
            (directory / 'queue.json').write_text(json.dumps(records, indent=2))
    skipped = [r['variant'] for r in records if r['status'] == 'budget_skipped']
    if not args.dry_run:
        (directory / 'queue.json').write_text(json.dumps(records, indent=2))
    if skipped:
        print('Continue later: bash scripts/10_01_imgr10_branch_choice_' + args.machine +
              '.sh --only ' + ' '.join(skipped), flush=True)
    if not args.dry_run:
        from scripts.analyze_branch_choice import analyze_directory
        analyze_directory(directory)
    print(f'Queue complete; FAILED={failed}; outputs={directory}', flush=True)
    return failed


if __name__ == '__main__':
    sys.exit(main())
