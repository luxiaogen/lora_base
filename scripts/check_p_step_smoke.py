"""Launcher checks only: require completed short runs and actual P updates."""
import sys
from pathlib import Path

from plot_p_step_direction import parse_logs, summary


def main():
    runs = parse_logs([Path(p) for p in sys.argv[1:]])
    assert len(runs) == 3, 'Expected all three smoke runs'
    for run in runs:
        row = summary(run)
        assert row['complete'] and row['tasks'] == 2, row
        assert row['sampled_layer_steps'] > 0, 'P step route was not exercised'
        if row['mode'] == 'baseline':
            assert row['applied'] == 0
        else:
            assert row['applied'] > 0, 'No candidate accepted in smoke; inspect before spending a full night'
        for step in run['steps']:
            if step['accepted']:
                assert step['effective_norm'] <= step['reference_norm'] * 1.00011 + 1e-11
                assert step['effective_norm'] >= step['reference_norm'] * .97999 - 1e-11
                if step['proposal'] == 'conflict':
                    assert step['conflict_norm'] <= step['reference_conflict_norm'] * 1.00011 + 1e-11
    assert len({r['stages'][0]['total'] for r in runs}) == 1, 'Task0 differs; inspect paired setup'
    print('GPU training smoke: all three routes completed; Task0 matched; both candidates applied.')
    print('Smoke accuracies are not performance results.')


if __name__ == '__main__':
    main()
