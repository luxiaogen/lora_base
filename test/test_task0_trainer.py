"""Check the real trainer's save/resume task boundaries without a GPU/dataset."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import trainer


class SmallLearner:
    calls = []

    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = self._total_classes = 0
        self._network = torch.nn.Linear(2, 2)
        self.acc_matrix = np.zeros((10, 10))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes += 20
        self.calls.append(self._cur_task)

    def eval_task(self):
        score = 96. - self._cur_task
        self.acc_matrix[:self._cur_task + 1, self._cur_task] = score
        return {'top1': score, 'grouped': {'total': score}}, {'top1': 96.}, None, score / 100

    def after_task(self):
        self._known_classes = self._total_classes


class Task0TrainerTests(unittest.TestCase):
    def test_resume_runs_only_tasks_one_two_and_keeps_task0_curve(self):
        args = json.loads(Path('exps/dlora/imgr10.json').read_text())
        args.update(seed=1993, device=['-1'], dual_mask_track_w0_metrics=False, max_tasks=1,
                    task0_checkpoint_save='task0.pt', dual_mask_s_task0_only=False)
        dm = SimpleNamespace(nb_tasks=10, _class_order=[0, 1], _train_data=['a'], _train_targets=[0],
                             _test_data=['b'], _test_targets=[1])
        previous = os.getcwd()
        SmallLearner.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with patch.object(trainer, 'DataManager', return_value=dm), \
                     patch.object(trainer.factory, 'get_model', side_effect=lambda name, cfg: SmallLearner(cfg)), \
                     patch.object(trainer.logging, 'basicConfig'), \
                     patch.object(trainer.logging, 'FileHandler'), \
                     patch.object(trainer, '_log_experiment_summary') as summary:
                    trainer._train(dict(args))
                    self.assertTrue(Path('task0.pt').exists())
                    self.assertEqual(SmallLearner.calls, [0])
                    resume_args = dict(args, device=['-1'], max_tasks=3, task0_checkpoint_save='',
                                       task0_checkpoint_resume='task0.pt', dual_mask_s_task0_only=True)
                    trainer._train(resume_args)
                    self.assertEqual(SmallLearner.calls, [0, 1, 2])
                    self.assertEqual(summary.call_args.args[1], [96., 95., 94.])
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
