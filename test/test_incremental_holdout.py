import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.data_manager import DataManager
from utils.incremental_holdout import record_holdout
from test.test_stage_audit import Network
from scripts.summarize_distill_holdout import parse


class IncrementalHoldoutTests(unittest.TestCase):
    def test_init_default_preserves_data_and_enabled_reserves_split(self):
        def setup(manager, *args):
            manager._train_data = np.arange(40)
            manager._train_targets = np.repeat(np.arange(4), 10)
            manager._class_order = [0, 1, 2, 3]
        with patch.object(DataManager, '_setup_data', setup):
            default = DataManager('fake', False, 1993, 2, 2)
            enabled = DataManager('fake', False, 1993, 2, 2,
                                  {'incremental_holdout': True, 'incremental_holdout_mod': 5})
        self.assertEqual(len(default._train_data), 40)
        self.assertFalse(hasattr(default, '_holdout_data'))
        self.assertEqual(len(enabled._train_data), 32)
        self.assertEqual(len(enabled._holdout_data), 8)
        self.assertEqual(default._increments, enabled._increments)

    def test_summary_ignores_test_reporting(self):
        text = 'Starting: weight0 description\nCNN: fake_test_metrics\nIncrementalHoldout ' + json.dumps({'task': 0, 'metrics': {}})
        runs = parse(text)
        self.assertEqual(len(runs), 1)
        self.assertEqual(list(runs[0]['tasks']), [0])

    def manager(self):
        m = DataManager.__new__(DataManager)
        m._train_data = np.arange(40)
        m._train_targets = np.repeat(np.arange(4), 10)
        m._test_data = np.arange(100, 108)
        m._test_targets = np.repeat(np.arange(4), 2)
        m._train_trsf = m._test_trsf = m._common_trsf = []
        m.use_path = False
        m._reserve_incremental_holdout(5)
        return m

    def test_all_train_consumers_exclude_holdout_and_test_is_untouched(self):
        a, b = self.manager(), self.manager()
        self.assertTrue(np.array_equal(a._holdout_data, b._holdout_data))
        self.assertEqual(np.bincount(a._holdout_targets).tolist(), [2]*4)
        for indices in ([0], [1], [0, 1], [0, 1, 2]):
            held = a.get_incremental_holdout(indices)
            for mode in ('train', 'test'):
                data, labels, dataset = a.get_dataset(indices, source='train', mode=mode, ret_data=True)
                self.assertFalse(set(data) & set(a._holdout_data))
                self.assertEqual(set(data) | set(held.images),
                                 {i for i in range(40) if i//10 in indices})
        self.assertEqual(a.get_dataset([0, 1, 2, 3], 'test', 'test').images.tolist(), list(range(100,108)))

    def test_evaluation_preserves_parameters_modes_rng_and_reports_global_groups(self):
        net = Network('cpu').train()
        loader = DataLoader(TensorDataset(torch.arange(8), torch.randn(8, 3), torch.tensor([0,1,2,3]*2)),
                            batch_size=4, generator=torch.Generator().manual_seed(1993))
        state = torch.get_rng_state().clone()
        weights = {k: v.clone() for k,v in net.state_dict().items()}
        result = record_holdout(net, loader, 'cpu', 1, 2)
        self.assertEqual(result['source'], 'train_holdout')
        self.assertEqual(result['metrics']['old']['n'], 4)
        self.assertEqual(result['metrics']['new']['n'], 4)
        self.assertTrue(net.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, net.state_dict()[k]) for k,v in weights.items()))

    def test_three_short_runs_only_weight_differs(self):
        spec = json.loads(Path('scripts/sweeps/imgr10_distill_holdout_t3_3090.json').read_text())
        self.assertEqual([v['overrides'] for v in spec['variants']],
                         [{'old_model_distill_weight': w} for w in (0, .25, 1)])
        common = spec['common_overrides']
        self.assertEqual(common['max_tasks'], 3)
        self.assertEqual(common['dual_mask_anchor_reg_weight'], 10)
        self.assertTrue(common['incremental_holdout'])
        self.assertFalse(common['task0_validation_enabled'])
        self.assertFalse(common['plora_train_a'])
        self.assertNotIn('data_path', common)
        result = subprocess.check_output(['bash', 'scripts/9_27_imgr10_distill_holdout_t3_3090.sh', '--dry-run'], text=True)
        self.assertEqual(result.count('main.py --config'), 3)
        self.assertEqual(result.count('--set max_tasks=3'), 3)
