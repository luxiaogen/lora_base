import json
from pathlib import Path
import shlex
import subprocess
import unittest


class AnchorFullTrainQueueTests(unittest.TestCase):
    def commands(self, *args):
        text = subprocess.check_output(['bash', 'scripts/9_28_imgr10_anchor_fulltrain_t3_3090.sh',
                                        '--dry-run', *args], text=True)
        result = []
        for line in text.splitlines():
            if 'main.py --config' not in line:
                continue
            tokens = shlex.split(line)
            result.append(dict(tokens[i+1].split('=', 1) for i, t in enumerate(tokens) if t == '--set'))
        return result

    def test_four_full_training_candidates_no_baseline_repeat(self):
        commands = self.commands()
        self.assertEqual([c['dual_mask_anchor_reg_weight'] for c in commands], ['0','2.5','5','20'])
        for c in commands:
            self.assertEqual(c['incremental_holdout'], 'false')
            self.assertEqual(c['task0_validation_enabled'], 'false')
            self.assertEqual(c['max_tasks'], '3')
            self.assertEqual(c['total_sessions'], '10')
            self.assertEqual(c['init_cls'], '20')
            self.assertEqual(c['increment'], '20')
            self.assertEqual(c['init_epoch'], '20')
            self.assertEqual(c['epochs'], '20')
            self.assertEqual(c['ca_epochs'], '5')
            self.assertEqual(c['seed'], '[1993]')
            self.assertEqual(c['head_start_epochs'], '0')
            self.assertEqual(c['head_start_init'], 'random')
            self.assertEqual(c['dual_mask_anchor_reg_task0_only'], 'true')
            self.assertNotIn('data_path', c)
        stripped = [{k:v for k,v in c.items() if k not in ('prefix','dual_mask_anchor_reg_weight')}
                    for c in commands]
        self.assertTrue(all(c == stripped[0] for c in stripped))

    def test_smoke_does_not_enable_holdout(self):
        for c in self.commands('--smoke'):
            self.assertEqual(c['max_tasks'], '2')
            self.assertEqual(c['epochs'], '1')
            self.assertEqual(c['incremental_holdout'], 'false')

    def test_no_auto_selection(self):
        spec = json.loads(Path('scripts/sweeps/imgr10_anchor_fulltrain_t3_3090.json').read_text())
        self.assertNotIn('reference', spec)
        text = Path('scripts/9_28_imgr10_anchor_fulltrain_t3_3090.sh').read_text()
        self.assertNotIn('screen_result', text)
        self.assertNotIn('run_head_start_night', text)
