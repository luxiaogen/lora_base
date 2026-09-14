import json
from pathlib import Path
import re
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def commands(path):
    source = path.read_text()
    blocks = re.findall(r'(?ms)^    python main\.py.*?2>&1 \| tee', source)
    parsed = []
    for block in blocks:
        tokens = shlex.split(block.replace('\\\n', ' '))
        options = {tokens[i + 1].split('=', 1)[0]: tokens[i + 1].split('=', 1)[1]
                   for i, token in enumerate(tokens[:-1]) if token == '--set'}
        parsed.append(options)
    return source, parsed


class OvernightSoftQKScriptsTests(unittest.TestCase):
    def test_full_ten_task_matrix_and_matched_controls(self):
        soft_script = ROOT / 'scripts/9_14_imgr10_soft_qk_5090.sh'
        qk_script = ROOT / 'scripts/9_14_imgr10_qk_all_3090.sh'
        source_5090, runs_5090 = commands(soft_script)
        source_3090, runs_3090 = commands(qk_script)
        self.assertEqual(len(runs_5090), 6)
        self.assertEqual(len(runs_3090), 3)
        self.assertNotIn('cd "$(dirname', source_5090 + source_3090)
        for script in (soft_script, qk_script):
            subprocess.run(['bash', '-n', str(script)], check=True)
        for seed in (1993, 1996, 1997):
            matched = [run for run in runs_5090 if run['seed'] == f'[{seed}]']
            self.assertEqual(len(matched), 2)
            qkv = next(run for run in matched if run['dual_mask_qk_all_tasks'] == 'false')
            qk = next(run for run in matched if run['dual_mask_qk_all_tasks'] == 'true')
            self.assertEqual(qkv['dual_mask_p_conflict_diagnostics'], 'true')
            self.assertEqual(qk['dual_mask_p_conflict_diagnostics'], 'true')
            ignored = {'prefix', 'dual_mask_qk_all_tasks', 'wandb_group', 'wandb_tags'}
            self.assertEqual({k: v for k, v in qkv.items() if k not in ignored},
                             {k: v for k, v in qk.items() if k not in ignored})
            replica = next(run for run in runs_3090 if run['seed'] == f'[{seed}]')
            ignored_replica = {'wandb_group', 'wandb_tags', 'dual_mask_p_conflict_diagnostics'}
            self.assertEqual({k: v for k, v in qk.items() if k not in ignored_replica},
                             {k: v for k, v in replica.items() if k not in ignored_replica})
            for run in (qkv, qk, replica):
                self.assertEqual(run['max_tasks'], '10')
                self.assertEqual(run['rank'], '64')
                self.assertEqual(run['init_epoch'], '20')
                self.assertEqual(run['epochs'], '20')
                self.assertEqual(run['ca'], 'true')
                self.assertEqual(run['ca_epochs'], '5')
                self.assertEqual(run['task0_checkpoint_resume'], '')
                self.assertEqual(run['task0_checkpoint_save'], '')
                self.assertNotIn('data_path', run)

    def test_specs_record_same_matrix_as_scripts(self):
        for name, expected_runs in (('imgr10_soft_qk_5090', 6), ('imgr10_qk_all_3090', 3)):
            spec = json.loads((ROOT / f'scripts/sweeps/{name}.json').read_text())
            self.assertEqual(spec['seeds'], [1993, 1996, 1997])
            self.assertEqual(len(spec['seeds']) * len(spec['variants']), expected_runs)
            self.assertEqual(spec['datasets'][0]['config'], 'exps/dlora/imgr10.json')


if __name__ == '__main__':
    unittest.main()
