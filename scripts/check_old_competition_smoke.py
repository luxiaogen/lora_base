"""Launcher checks: candidate is active, Task0 is unchanged, no test feedback."""
import json
from pathlib import Path
import re
import sys


def check(paths):
    task0 = []
    for i, path in enumerate(paths):
        text = Path(path).read_text(errors='replace')
        reports = [json.loads(line.split('StageAudit ', 1)[1]) for line in text.splitlines() if 'StageAudit {' in line]
        task0.append(next(r['metrics']['total']['accuracy'] for r in reports
                          if r['task'] == 0 and r['stage'] == 'post_ca'))
        values = [float(v) for v in re.findall(r'old_competition_weighted ([0-9.eE+\-]+)', text)]
        if i == 0:
            assert not values, 'Baseline unexpectedly used competition loss'
        else:
            assert values and max(values) > 0, 'Candidate loss did not engage in smoke'
    assert len(task0) == 2 and task0[0] == task0[1], 'Task0 changed in a Task1-only experiment'
    print('Old competition smoke: Task0 matched and candidate loss engaged; not performance evidence.')


if __name__ == '__main__':
    check(sys.argv[1:])
