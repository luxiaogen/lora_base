"""Run-level checks belong in the launcher, not the training method."""
import json
from pathlib import Path
import sys


def check(paths):
    reports = []
    for i, path in enumerate(paths):
        text = Path(path).read_text(errors='replace')
        rows = [json.loads(line.split('StageAudit ', 1)[1]) for line in text.splitlines()
                if 'StageAudit {' in line]
        reports.append(rows)
        source = 'gaussian' if i == 0 else 'real'
        assert f'CA feature source: new={source} old=gaussian per_class=256' in text
        assert 'old_competition_weighted ' not in text
    for task, stage in ((0, 'post_ca'), (1, 'post_merge')):
        pair = [next(r for r in rows if r['task'] == task and r['stage'] == stage) for rows in reports]
        assert pair[0]['sample_sha256'] == pair[1]['sample_sha256']
        assert pair[0]['metrics']['total']['accuracy'] == pair[1]['metrics']['total']['accuracy'], (task, stage)
    print('CA source smoke: matching pre-CA accuracy, real-feature route engaged; not performance proof.')


if __name__ == '__main__':
    check(sys.argv[1:])
