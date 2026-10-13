"""运行记录校验必须拒绝半成品和被异常中断的训练。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReproductionRecordTests(unittest.TestCase):
    def make_record(self, root):
        config = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
        config.update(init_epoch=1, epochs=1, ca_epochs=1, max_tasks=2)
        rows = []
        for task in (0, 1):
            controls = dict(alpha=[.5] * 12, beta=[.5] * 12, R_old=0.,
                            S_rank=[64] * 12, P_init_rank=[39 if task == 0 else 64] * 12,
                            P_active=task > 0, protection_active=task > 0,
                            coverage_center=None if task == 0 else .8944721110165117,
                            layer_coverage=[.9] * 12 if task == 0 else
                            [.8944721110165117 + .05 * (1. - 2. * i / 11.) for i in range(12)])
            rows.append(dict(task=task, top1=90. if task == 0 else 85., old=0. if task == 0 else 84.,
                             new=90. if task == 0 else 89., controls=controls))
        result = dict(status='completed', planned_tasks=2, tasks_completed=2, tasks=rows,
                      summary=dict(Average=87.5, Last=85., Old=84., New=89., StageOld=84., StageNew=89., Forgetting=6.))
        (root / 'effective_config.json').write_text(json.dumps(config))
        (root / 'run_info.json').write_text(json.dumps(dict(source_revision='a' * 40, source_dirty=False)))
        (root / 'result.json').write_text(json.dumps(result))
        lines = ['2026-10-14 01:00:00 [dlora.py] => Task 0, Epoch 1/1 => Loss 1.003, CE 1.000, anchor_reg 1.000e-03, anchor_reg_weighted 2.500e-03, Train_accy 70.00',
                 '2026-10-14 01:00:00 [dlora.py] => Task 1, Epoch 1/1 => Loss 1.000, CE 1.000, anchor_reg 0.000e+00, anchor_reg_weighted 0.000e+00, Train_accy 70.00',
                 '2026-10-14 01:00:00 [dlora.py] => CA Task 1, Epoch 1/1 => Loss 0.100']
        (root / 'train.log').write_text('\n'.join(lines) + '\n')

    def check(self, root, expected_revision=None):
        env = dict(os.environ)
        env.pop('EXPECTED_REVISION', None)
        if expected_revision is not None:
            env['EXPECTED_REVISION'] = expected_revision
        return subprocess.run([sys.executable, str(ROOT / 'scripts/check_reproduction.py'), str(root),
                               '--tasks', '2', '--epochs', '1'], capture_output=True, text=True, env=env)

    def test_complete_smoke_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_record(root)
            result = self.check(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_partial_record_cannot_unlock_formal_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_record(root)
            record = json.loads((root / 'result.json').read_text())
            record.update(status='running', tasks_completed=1, tasks=record['tasks'][:1])
            (root / 'result.json').write_text(json.dumps(record))
            self.assertEqual(self.check(root).returncode, 1)

    def test_exception_in_log_is_rejected_even_with_complete_JSON(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_record(root)
            with (root / 'train.log').open('a') as stream:
                stream.write('Traceback (most recent call last):\n')
            self.assertEqual(self.check(root).returncode, 1)

    def test_wrong_task0_P_rank_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_record(root)
            record = json.loads((root / 'result.json').read_text())
            record['tasks'][0]['controls']['P_init_rank'] = [64] * 12
            (root / 'result.json').write_text(json.dumps(record))
            self.assertEqual(self.check(root).returncode, 1)

    def test_uniform_coverage_cannot_impersonate_depth_B(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_record(root)
            record = json.loads((root / 'result.json').read_text())
            record['tasks'][1]['controls'].update(coverage_center=.9, layer_coverage=[.9] * 12)
            (root / 'result.json').write_text(json.dumps(record))
            self.assertEqual(self.check(root).returncode, 1)

    def test_CA_must_belong_to_the_incremental_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_record(root)
            log = root / 'train.log'
            log.write_text(log.read_text().replace('CA Task 1', 'CA Task 0'))
            self.assertEqual(self.check(root).returncode, 1)

    def test_actual_source_must_match_the_fixed_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_record(root)
            self.assertEqual(self.check(root, expected_revision='b' * 40).returncode, 1)

    def test_changed_learning_rate_is_not_old_B(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_record(root)
            config = json.loads((root / 'effective_config.json').read_text())
            config['lrate'] = .01
            (root / 'effective_config.json').write_text(json.dumps(config))
            self.assertEqual(self.check(root).returncode, 1)


if __name__ == '__main__':
    unittest.main()
