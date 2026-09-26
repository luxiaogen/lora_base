import json
from pathlib import Path
import shlex
import subprocess
import unittest

import torch

from models.attention import Attention_LoRA, _exact_top_ratio_mask, _top_ratio_mask
from test.test_dual_mask_core import make_args


class FixedCoverageTests(unittest.TestCase):
    def module(self, ratio, exact=True):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params(make_args(
            dual_mask_conflict_exact_topk=exact,
            dual_mask_conflict_ratio=ratio,
            dual_mask_conflict_old_overlap_adaptive=False,
            dual_mask_task0_gate_mode='unmasked',
        ))
        module.cur_task = 1
        module.w0_importance.fill_(1)
        module.general_mask[:, ::2] = 1
        module.general_mask[:, 1::2] = 0
        return module

    def test_exact_counts_with_ties_and_zero_scores(self):
        for score in (torch.zeros(10, 10), torch.ones(10, 10), torch.arange(100.).reshape(10, 10) % 3):
            for ratio in (0, .1, .2, .4, .6, 1):
                before = torch.get_rng_state()
                mask = _exact_top_ratio_mask(score, ratio)
                self.assertEqual(mask.sum().item(), int(100 * ratio))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertTrue(torch.equal(mask, _exact_top_ratio_mask(score, ratio)))
                if 0 < ratio < 1:
                    self.assertGreaterEqual(score[mask.bool()].min(), score[~mask.bool()].max())

    def test_valid_region_and_floor_rounding(self):
        score = torch.ones(3, 3)
        valid = torch.eye(3)
        mask = _exact_top_ratio_mask(score, .5, valid)
        self.assertEqual(mask.sum().item(), 1)
        self.assertEqual((mask * (1 - valid)).sum().item(), 0)
        self.assertTrue(torch.equal(_exact_top_ratio_mask(score, 1, valid), valid))
        self.assertEqual(_exact_top_ratio_mask(score, 1, valid * 0).sum().item(), 0)

    def test_legacy_selection_unchanged(self):
        module = self.module(.2, exact=False)
        for delta in (torch.ones_like(module.qkv.weight), torch.arange(48.).reshape_as(module.qkv.weight)):
            score, mask = module._joint_conflict(delta)
            self.assertTrue(torch.equal(mask, _top_ratio_mask(score, .2)))

    def test_forward_merge_and_endpoints(self):
        delta = torch.arange(1., 49.).reshape(12, 4)
        for ratio in (0, .1, .2, .4, .6, 1):
            module = self.module(ratio)
            module.effective_protect_strength = .3
            for isolated in (False, True):
                actual = module._safe_delta(delta, isolated=isolated)
                merged = module._compose_merge_delta(delta, isolated, ratio, .5)
                torch.testing.assert_close(actual, merged, rtol=0, atol=0)
                _, mask = module._joint_conflict(delta)
                self.assertEqual(mask.sum().item(), int(48 * ratio))
                base = (1 - module.general_mask) if isolated else (1 - .3 * module.general_mask)
                torch.testing.assert_close(actual, delta * base * (1 - .5 * mask))
            module.cur_task = 0
            torch.testing.assert_close(module._safe_delta(delta, True), delta)
            torch.testing.assert_close(module._safe_delta(delta, False), delta)

    def test_machine_specs_and_commands(self):
        commons = []
        for gpu, ratios in (('3090', [.1, 0, .4, .6]), ('5090', [.1, .2, 1])):
            spec = json.loads(Path(f'scripts/sweeps/imgr10_fixed_coverage_{gpu}.json').read_text())
            common = spec['common_overrides']
            self.assertEqual(spec['seeds'], [1993])
            self.assertEqual([v['overrides'] for v in spec['variants']],
                             [{'dual_mask_conflict_ratio': r} for r in ratios])
            for key in ('dual_mask_conflict_energy_adaptive', 'dual_mask_conflict_old_overlap_adaptive', 'ca_real_new_features'):
                self.assertFalse(common[key])
            self.assertTrue(common['dual_mask_conflict_exact_topk'])
            self.assertEqual(common['dual_mask_conflict_strength'], .5)
            self.assertEqual(common['dual_mask_conflict_budget_multiplier'], 1)
            self.assertEqual(common['dual_mask_conflict_granularity'], 'layer')
            self.assertEqual(common['dual_mask_private_conflict_mode'], 'global')
            commons.append({k: v for k, v in common.items() if k != 'wandb_group'})
            script = f'scripts/9_26_imgr10_fixed_coverage_{gpu}.sh'
            source = Path(script).read_text()
            for check in ('assert', 'preflight', 'unittest'):
                self.assertNotIn(check, source)
            subprocess.run(['bash', '-n', script], check=True)
            commands = subprocess.check_output(['bash', script, '--dry-run'], text=True).split('    python main.py')[1:]
            self.assertEqual(len(commands), len(ratios))
            for command, variant in zip(commands, spec['variants']):
                tokens = shlex.split(command.replace('\\\n', ' '))
                settings = dict(tokens[i+1].split('=', 1) for i, token in enumerate(tokens) if token == '--set')
                for key, value in {**common, **variant['overrides']}.items():
                    self.assertEqual(settings[key], value if isinstance(value, str) else json.dumps(value, separators=(',', ':')))
                self.assertNotIn('data_path', settings)
        self.assertEqual(*commons)

    def test_attention_training_and_single_merge(self):
        for ratio in (0, .1, .2, .4, .6, 1):
            torch.manual_seed(1993)
            module = self.module(ratio)
            module.before_task(1)
            module.train()
            inputs = torch.randn(2, 3, 4)
            optimizer = torch.optim.SGD((p for p in module.parameters() if p.requires_grad), lr=.02)
            for _ in range(2):
                optimizer.zero_grad()
                loss = module(inputs, task=1).square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
            module.eval()
            with torch.no_grad():
                before = module(inputs, task=1)
                module.after_task(1)
                after = module(inputs, task=1)
                torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
