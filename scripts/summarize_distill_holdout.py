"""Print reserved-training-set results only; never rank by test accuracy."""
import argparse
import json
from pathlib import Path


def parse(text):
    runs = []
    run = None
    for line in text.splitlines():
        if line.startswith('Starting:'):
            run = {'name': line.split()[1], 'tasks': {}}
            runs.append(run)
        if 'IncrementalHoldout ' in line and run is not None:
            row = json.loads(line.split('IncrementalHoldout ', 1)[1])
            run['tasks'][row['task']] = row
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    args = parser.parse_args()
    runs = parse(Path(args.log).read_text())
    print('| Run | Task | Total | Old | New | Old margin | New margin |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for run in runs:
        for task, row in sorted(run['tasks'].items()):
            m = row['metrics']
            values = [m['total']['accuracy'], m['old']['accuracy'], m['new']['accuracy'],
                      m['old']['margin'], m['new']['margin']]
            print('| ' + ' | '.join([run['name'], str(task)] +
                                    ['—' if v is None else f'{v:.4f}' for v in values]) + ' |')
    print('\nTask0–2 must all be present. These are holdout results, not formal T10 scores.')


if __name__ == '__main__':
    main()
