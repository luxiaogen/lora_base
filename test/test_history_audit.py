import tempfile
import unittest
import subprocess

import torch

from utils.history_audit import HistoryAudit, compare_prefix


class HistoryTests(unittest.TestCase):
    def test_launcher_recipe(self):
        output = subprocess.check_output(['bash', 'scripts/9_27_imgr10_history_audit_3090.sh', '--dry-run'], text=True)
        for setting in ('dual_mask_anchor_reg_weight=10', 'head_balance_weight=0',
                        'history_audit=true', 'ca_epochs=5', 'max_tasks=10', 'seed=[1993]'):
            self.assertIn(setting, output)
        self.assertNotIn('--set data_path=', output)

    def test_growing_prefix_multiple_references(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = HistoryAudit(directory)
            audit.record(0, 'post_ca', torch.eye(2), torch.arange(2), torch.arange(2))
            with self.assertLogs(level='INFO'):
                audit.record(1, 'post_ca', torch.eye(3), torch.arange(3), torch.arange(3))
                rows = audit.record(2, 'pre_merge', torch.eye(4), torch.arange(4), torch.arange(4))
            self.assertEqual([r['n'] for r in rows], [2, 3])
            self.assertEqual([r['fixed_broken'] for r in rows], [0, 0])

    def test_expansion_without_forgetting(self):
        ref = torch.eye(2)
        result = compare_prefix(torch.tensor([[1., 0., 2.], [0., 1., 0.]]),
                                torch.arange(2), ref)
        self.assertEqual(result['fixed_broken'], 0)
        self.assertEqual(result['expansion_only_errors'], 1)

    def test_break_recovery_and_exact_decomposition(self):
        labels = torch.tensor([0, 1, 0])
        ref = torch.tensor([[2., 0.], [0., 2.], [0., 2.]])
        now = torch.tensor([[0., 2., 3.], [0., 2., 3.], [2., 0., 0.]])
        r = compare_prefix(now, labels, ref)
        self.assertEqual((r['fixed_broken'], r['fixed_corrected'], r['expansion_only_errors']), (1, 1, 1))
        self.assertEqual(r['reference_correct'] - r['full_correct'],
                         r['fixed_broken'] - r['fixed_corrected'] + r['expansion_only_errors'])

    def test_alignment_persistence_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = HistoryAudit(directory)
            audit.record(0, 'post_ca', torch.eye(2), torch.arange(2), torch.tensor([8, 9]))
            state = torch.get_rng_state().clone()
            with self.assertLogs(level='INFO'):
                rows = audit.record(1, 'pre_merge', torch.tensor([[0., 1., 0.], [1., 0., 2.]]),
                                    torch.tensor([1, 0]), torch.tensor([9, 8]))
            self.assertEqual(rows[0]['expansion_only_errors'], 1)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            saved = torch.load(f'{directory}/task01_pre_merge.pt', weights_only=True)
            self.assertEqual(saved['indices'].tolist(), [9, 8])
            self.assertEqual(list(audit.references), [0])
