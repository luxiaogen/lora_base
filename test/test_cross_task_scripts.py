import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CrossTaskScriptTests(unittest.TestCase):
    def test_paired_matrix_and_fixed_protocol(self):
        protocols = []
        for gpu, count in [('3090', 6), ('5090', 9)]:
            text = (ROOT / 'scripts' / ('9_10_cross_task_' + gpu + '.sh')).read_text()
            self.assertNotRegex(text, r'^cd\s')
            blocks = re.findall(r'    python main.py (.*?)        "\$@"', text, re.S)
            self.assertEqual(len(blocks), count)
            pairs = []
            for block in blocks:
                args = shlex.split(block.replace('\\\n', ''))
                settings = dict(args[i + 1].split('=', 1) for i, value in enumerate(args) if value == '--set')
                pairs.append((settings.pop('seed'), settings.pop('classification_training_mode')))
                settings.pop('prefix')
                settings.pop('wandb_group')
                settings.pop('wandb_tags')
                protocols.append(settings)
                self.assertEqual(settings['ca_epochs'], '5')
                self.assertEqual(settings['dual_mask_private_rank'], '0')
                self.assertEqual(settings['dual_mask_conflict_merge_mode'], 'suppress')
            expected = [(f'[{s}]', m) for s in (1993, 1996, 1997) for m in ('task_local', 'all_seen_replay')]
            if gpu == '5090':
                expected += [(f'[{s}]', 'all_seen') for s in (1993, 1996, 1997)]
            self.assertEqual(pairs, expected)
        self.assertTrue(all(protocol == protocols[0] for protocol in protocols))

    def test_shell_runs_continue_after_failure_and_forward_path_override(self):
        for gpu, count in [('3090', 6), ('5090', 9)]:
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                (work / 'main.py').touch()
                stub = work / 'python'
                stub.write_text('#!/bin/sh\n'
                                'if [ "$1" = main.py ]; then\n'
                                '  echo "$*" >> "$RUN_RECORD"\n'
                                '  case "$*" in *A_local_seed1993*) exit 1 ;; esac\n'
                                'fi\nexit 0\n')
                stub.chmod(0o755)
                record = work / 'calls.txt'
                env = dict(os.environ, PATH=directory + os.pathsep + os.environ['PATH'], RUN_RECORD=str(record))
                script = ROOT / 'scripts' / ('9_10_cross_task_' + gpu + '.sh')
                result = subprocess.run(['bash', str(script), '--set', 'data_path=/data/with spaces'],
                                        cwd=work, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                calls = record.read_text().splitlines()
                self.assertEqual(len(calls), count)
                self.assertTrue(all('data_path=/data/with spaces' in call for call in calls))
                self.assertIn('FAILED=1', result.stdout)


if __name__ == '__main__':
    unittest.main()
