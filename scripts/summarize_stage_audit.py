"""Check stage-report completeness and export directly attributable transitions."""
import argparse
import csv
import json
from pathlib import Path


def parse(path, expected_tasks):
    text = path.read_text(errors='replace')
    rows = [json.loads(line.split('StageAudit ', 1)[1]) for line in text.splitlines()
            if 'StageAudit {' in line]
    assert 'Traceback (most recent call last)' not in text, path
    assert '=> Last Accuracy:' in text, f'Incomplete training: {path}'
    assert len(rows) == expected_tasks * 3, (path, len(rows))
    result = []
    for task in range(expected_tasks):
        group = [r for r in rows if r['task'] == task]
        assert [r['stage'] for r in group] == ['pre_merge', 'post_merge', 'post_ca']
        assert len({r['sample_sha256'] for r in group}) == 1, 'Sample order changed'
        assert len({r['class_count'] for r in group}) == 1, 'Class range changed'
        assert [r['previous_stage'] for r in group] == [None, 'pre_merge', 'post_merge']
        for row in group:
            for partition, metrics in row['metrics'].items():
                result.append(dict(source=str(path), task=task, stage=row['stage'],
                                   partition=partition, **metrics))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('logs', type=Path, nargs='+')
    args = parser.parse_args()
    rows = [row for path in args.logs for row in parse(path, args.tasks)]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('Stage audit complete; test-report-only CSV:', args.output)


if __name__ == '__main__':
    main()
