import json
from pathlib import Path
import re
import shlex
import subprocess
import types
import unittest

import torch

from models.attention import Attention_LoRA


ROOT = Path(__file__).resolve().parents[1]


def make_module(mode, layer):
    args = {
        'use_slora': True,
        'use_plora': True,
        'dual_mask_importance': 'svd',
        'dual_mask_general_ratio': 0.4,
        'dual_mask_coverage_mode': 'ratio',
        'dual_mask_layerwise_ratio_mode': mode,
        'dual_mask_competence_adaptive': True,
        'lora_A_init': 'kaiming',
    }
    module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
    module._init_params(args)
    module.layer_idx = layer
    score = torch.arange(module.qkv.weight.numel()).reshape_as(module.qkv.weight)
    module._combined_importance = types.MethodType(lambda self: score, module)
    module.rebuild_dual_masks()
    return module


def commands(path):
    source = path.read_text()
    blocks = re.findall(r'(?ms)^    python main\.py.*?2>&1 \| tee', source)
    parsed = []
    for block in blocks:
        tokens = shlex.split(block.replace('\\\n', ' '))
        parsed.append({tokens[i + 1].split('=', 1)[0]: tokens[i + 1].split('=', 1)[1]
                       for i, token in enumerate(tokens[:-1]) if token == '--set'})
    return source, parsed


class LayerwiseBudgetTests(unittest.TestCase):
    def test_shallow_high_reallocates_the_same_average_budget(self):
        uniform = [make_module('none', layer).general_mask.float().mean().item()
                   for layer in range(12)]
        shallow = [make_module('shallow_high', layer).general_mask.float().mean().item()
                   for layer in range(12)]
        self.assertTrue(all(value == uniform[0] for value in uniform))
        self.assertGreater(shallow[0], uniform[0])
        self.assertLess(shallow[-1], uniform[-1])
        self.assertTrue(all(a >= b for a, b in zip(shallow, shallow[1:])))
        self.assertLessEqual(abs(sum(shallow) / 12 - sum(uniform) / 12), 1 / 48)

    def test_seed1993_script_is_a_matched_two_run_ablation(self):
        script = ROOT / 'scripts/9_15_imgr10_layerwise_budget_5090.sh'
        source, runs = commands(script)
        self.assertEqual(len(runs), 2)
        self.assertNotIn('cd "$(dirname', source)
        subprocess.run(['bash', '-n', str(script)], check=True)
        uniform = next(run for run in runs if run['dual_mask_layerwise_ratio_mode'] == 'none')
        shallow = next(run for run in runs if run['dual_mask_layerwise_ratio_mode'] == 'shallow_high')
        ignored = {'prefix', 'dual_mask_layerwise_ratio_mode', 'wandb_tags'}
        self.assertEqual({k: v for k, v in uniform.items() if k not in ignored},
                         {k: v for k, v in shallow.items() if k not in ignored})
        for run in runs:
            self.assertEqual(run['seed'], '[1993]')
            self.assertEqual(run['max_tasks'], '10')
            self.assertEqual(run['dual_mask_coverage_mode'], 'ratio')
            self.assertEqual(run['dual_mask_general_ratio'], '0.4')
            self.assertEqual(run['dual_mask_conflict_energy_adaptive'], 'true')
            self.assertEqual(run['dual_mask_conflict_energy_ratio_floor'], 'true')
            self.assertEqual(run['dual_mask_conflict_ratio'], '0.1')
            self.assertEqual(run['dual_mask_qv_all_tasks'], 'false')
            self.assertEqual(run['task0_checkpoint_resume'], '')
            self.assertNotIn('data_path', run)

    def test_spec_records_one_seed_and_two_variants(self):
        spec = json.loads((ROOT / 'scripts/sweeps/imgr10_layerwise_budget_5090.json').read_text())
        self.assertEqual(spec['seeds'], [1993])
        self.assertEqual(len(spec['variants']), 2)


if __name__ == '__main__':
    unittest.main()
