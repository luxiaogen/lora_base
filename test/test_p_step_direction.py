"""Tests of actual-update constraints and unchanged baseline training."""
import copy
import ast
import logging
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

from test.test_global_conflict_budget import GlobalBudgetSelectionTests
from utils.p_step_direction import (
    finish_step, gram_preconditioner, prepare_step, projection_norms,
    propose_steps, split_effective_step, probe_directions,
)


class ToyGate:
    def __init__(self, a, b, dynamic=False):
        self.cur_task = 1
        self.layer_idx = 0
        self.plora_gamma = .75
        self.P_lora = [None, SimpleNamespace(A_weight=a, B_weight=b)]
        self.dynamic = dynamic

    def _safe_delta(self, raw, isolated, return_details=False):
        if self.dynamic:
            mask = (raw.abs() >= raw.abs().median()).to(raw)
        else:
            mask = torch.zeros_like(raw)
            mask[:, 0] = 1
        gate = 1 - .5 * mask
        gate[:, -1] = 0  # P plastic boundary is retained.
        result = raw * gate
        return (result, gate, mask) if return_details else result


class DirectionTests(unittest.TestCase):
    def test_training_hook_with_probes_matches_original_sgd(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_backward_and_step')
        namespace = {'torch': torch, 'logging': logging}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     '<training-hook>', 'exec'), namespace)

        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = GlobalBudgetSelectionTests._make_attention('layer')
                self.attention.before_task(1)
                self.attention.set_task_and_stage(1, 0)
                self.head = torch.nn.Linear(12, 3)

            def forward(self, inputs):
                return {'logits': self.head(self.attention._contrib_from_units(inputs, 1))}

        torch.manual_seed(31)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = Network().to(device)
        initial = copy.deepcopy(network.state_dict())
        inputs, targets = torch.randn(4, 4, device=device), torch.tensor([0, 1, 2, 0], device=device)
        results = []
        for diagnostic in (False, True):
            network.load_state_dict(initial)
            params = [p for p in network.parameters() if p.requires_grad]
            opt = torch.optim.SGD(params, lr=.02, momentum=.9)
            learner = SimpleNamespace(_network=network, _cur_task=1, _p_step_counts={},
                                      _iter_lora_modules=lambda: [network.attention],
                                      _p_step_probe=(inputs, torch.nn.functional.cross_entropy))
            rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone() if device.type == 'cuda' else None
            for step in range(3):
                learner._p_step_context = ('baseline', 0, step, True) if diagnostic else None
                output = network(inputs)
                loss = torch.nn.functional.cross_entropy(output['logits'], targets)
                with self.assertLogs(level='INFO') if diagnostic else self.subTest(step=step):
                    namespace['_backward_and_step'](learner, loss, None, opt, output, targets)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            if cuda_rng is not None:
                self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
            results.append(([p.detach().clone() for p in params],
                            [opt.state[p]['momentum_buffer'].clone() for p in params]))
        for xs, ys in zip(*results):
            for x, y in zip(xs, ys):
                self.assertTrue(torch.equal(x, y))

    def test_probe_restores_parameters_modes_and_rng(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(4, 3, bias=False)
                self.dropout = torch.nn.Dropout(.2)

            def forward(self, x):
                return {'logits': self.head(self.dropout(x))}

        torch.manual_seed(8)
        network = Network().train()
        network.head.eval()  # Preserve mixed module modes too.
        original = network.head.weight.detach().clone()
        inputs, targets = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])
        records = [(network.head.weight, original, original + .1, original - .1, original)]
        rng = torch.get_rng_state().clone()
        metrics = probe_directions(network, records, inputs, targets, torch.nn.functional.cross_entropy)
        self.assertEqual(set(metrics), {'reference', 'norm', 'conflict'})
        self.assertTrue(torch.equal(original, network.head.weight))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(network.training)
        self.assertFalse(network.head.training)
        def fail(logits, targets):
            raise RuntimeError('probe test')
        with self.assertRaisesRegex(RuntimeError, 'probe test'):
            probe_directions(network, records, inputs, targets, fail)
        self.assertTrue(torch.equal(original, network.head.weight))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_dynamic_gate_decomposition_is_exact(self):
        torch.manual_seed(5)
        x, y = torch.randn(9, 4), torch.randn(9, 4)
        m = ToyGate(None, None, dynamic=True)
        sx, gx, _ = m._safe_delta(x, True, True)
        sy, gy, _ = m._safe_delta(y, True, True)
        fixed, switch = split_effective_step(x, y, gx, gy, .75)
        torch.testing.assert_close(fixed + switch, .75 * (sy - sx))
        self.assertGreater(float(switch.norm()), 0)

    def test_candidates_are_feasible_and_can_be_nontrivial(self):
        accepted = 0
        for dynamic in (False, True):
            for seed in range(12):
                torch.manual_seed(seed)
                a = torch.tensor([[1., .1, 0., .2], [.8, .4, .1, 0.]])
                b = torch.randn(9, 2) * .1
                gradient = torch.randn_like(b)
                after = b - .01 * gradient
                module = ToyGate(a, after, dynamic)
                ref, choices, old_mask = propose_steps(module, b, after, gradient, gram_preconditioner(a))
                for name, candidate in choices.items():
                    if candidate is ref:
                        continue
                    accepted += 1
                    self.assertGreater(candidate['gain'], max(ref['gain'], 0.))
                    n, r = projection_norms(candidate['effective']), projection_norms(ref['effective'])
                    self.assertTrue(bool((n <= r * 1.0001 + 1e-12).all()))
                    self.assertTrue(bool((n >= r * .98 - 1e-12).all()))
                    self.assertEqual(float(candidate['effective'][:, -1].abs().max()), 0)
                    if name == 'conflict':
                        mask = old_mask | ref['mask'] | candidate['mask']
                        self.assertTrue(bool((projection_norms(candidate['effective'] * mask)
                                              <= projection_norms(ref['effective'] * mask) * 1.0001 + 1e-12).all()))
        self.assertGreater(accepted, 0)

    def test_zero_step_falls_back_without_nan(self):
        a, b = torch.eye(2), torch.zeros(6, 2)
        module = ToyGate(a, b)
        ref, choices, _ = propose_steps(module, b, b.clone(), torch.ones_like(b), gram_preconditioner(a))
        self.assertIs(choices['norm'], ref)
        self.assertIs(choices['conflict'], ref)

    def test_real_attention_baseline_diagnostic_preserves_training(self):
        torch.manual_seed(9)
        module = GlobalBudgetSelectionTests._make_attention('layer')
        module.before_task(1)
        module.set_task_and_stage(1, 0)
        x = torch.randn(2, 3, 4)
        state = copy.deepcopy(module.state_dict())
        endings = []
        for diagnostic in (False, True):
            module.load_state_dict(state)
            params = [module.S_lora[1].B_weight, module.P_lora[1].B_weight]
            opt = torch.optim.SGD(params, lr=.02, momentum=.9)
            rng = torch.get_rng_state().clone()
            with self.assertLogs(level='INFO') if diagnostic else self.subTest(diagnostic=False):
                for step in range(4):
                    loss = (module._contrib_from_units(x, 1) - .5).square().mean()
                    snapshots = prepare_step([module], loss) if diagnostic else None
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    if diagnostic:
                        finish_step(snapshots, 'baseline', 0, step, True)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            endings.append(([p.detach().clone() for p in params],
                            [opt.state[p]['momentum_buffer'].clone() for p in params]))
        for left, right in zip(*endings):
            for a, b in zip(left, right):
                self.assertTrue(torch.equal(a, b))

    def test_details_match_regular_forward_and_only_p_changes(self):
        torch.manual_seed(7)
        module = GlobalBudgetSelectionTests._make_attention('layer')
        module.before_task(1)
        module.set_task_and_stage(1, 0)
        raw = torch.randn(12, 4)
        safe, gate, _ = module._safe_delta(raw, True, return_details=True)
        self.assertTrue(torch.equal(safe, module._safe_delta(raw, True)))
        torch.testing.assert_close(safe, raw * gate)
        s = module.S_lora[1].B_weight.detach().clone()
        a = module.P_lora[1].A_weight.detach().clone()
        p = module.P_lora[1].B_weight
        loss = (p - .2).square().sum()
        snapshots = prepare_step([module], loss)
        loss.backward()
        torch.optim.SGD([p], lr=.02).step()
        finish_step(snapshots, 'conflict', 0, 0, False)
        self.assertTrue(torch.equal(s, module.S_lora[1].B_weight))
        self.assertTrue(torch.equal(a, module.P_lora[1].A_weight))


if __name__ == '__main__':
    unittest.main()
