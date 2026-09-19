import ast
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations"))
from dualmask_transfer.core import DualMask, select
from dualmask_transfer.hosts import cl_effective, install_cl, install_sd, sd_kernel


@contextmanager
def host_modules(modules):
    # Restore only these names, not lazy Torch imports loaded inside the context.
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def official_module(host, relative, names):
    root = ROOT / ".external" / ("dualmask_" + host)
    if not root.exists():
        root = Path("/tmp/dualmask-" + host + "-lora-source")
    path = root / relative
    if not path.exists():
        raise RuntimeError("Download pinned upstream sources with run.py --host " + host + " --setup")
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    module = types.ModuleType("official_test_" + host)
    module.__dict__.update(torch=torch, nn=nn, math=math, Parameter=nn.Parameter)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


class CoreTests(unittest.TestCase):
    def test_energy_floor_zero_and_complement(self):
        self.assertEqual(select(torch.zeros(10), .5, .1).sum(), 0)
        mask = select(torch.arange(100).float(), .5, .1)
        self.assertGreaterEqual(mask.sum(), 10)
        self.assertGreaterEqual((mask * torch.arange(100)).sum(), torch.arange(100).sum() * .5)
        gate = DualMask(torch.randn(8, 8), "private")
        delta = torch.randn(8, 8, requires_grad=True)
        output = gate(delta)
        self.assertEqual(torch.count_nonzero(output * gate.protect), 0)
        output.sum().backward()
        self.assertEqual(torch.count_nonzero(delta.grad * gate.protect), 0)

    def test_no_randomness_and_anchor_immutable(self):
        anchor = torch.randn(8, 8)
        rng = torch.get_rng_state().clone()
        gate = DualMask(anchor, "single")
        torch.testing.assert_close(torch.get_rng_state(), rng)
        score = gate.importance.clone()
        anchor.zero_()
        torch.testing.assert_close(score, gate.importance)
        delta = torch.randn(8, 8, requires_grad=True)
        gate(delta).sum().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertEqual(sum(p.numel() for p in gate.parameters()), 0)


class CLTests(unittest.TestCase):
    def test_off_task0_shared_boundary_private_copy(self):
        official = official_module("cl", "backbone/vit_cllora.py", {"Adapter_lora"})
        Adapter = official.Adapter_lora
        def adapter():
            return Adapter(types.SimpleNamespace(d_model=8, attn_bn=2), init_option="lora")
        class Vision(nn.Module):
            def __init__(self):
                super().__init__()
                self.general_pos, self.specfic_pos, self.adapt_pos = [0], [1], [0, 1]
                self.blocks = nn.ModuleList([nn.Module(), nn.Module()])
                for block in self.blocks:
                    block.attn = nn.Module()
                    for name in ("q_proj", "k_proj", "v_proj"):
                        setattr(block.attn, name, nn.Linear(8, 8))
                self.cur_adapter = nn.ModuleList([nn.ModuleList([adapter(), nn.Identity(), adapter()]) for _ in range(2)])
                self.saved = []
            def add_adapter_to_list(self):
                self.saved.append(copy.deepcopy(self.cur_adapter))
                self.cur_adapter[1] = nn.ModuleList([adapter(), nn.Identity(), adapter()])
        class Learner:
            def __init__(self, args):
                self._network = types.SimpleNamespace(backbone=Vision())
        official.VisionTransformer = Vision
        learner_module = types.ModuleType("models.cllora")
        learner_module.Learner = Learner
        modules = {"backbone": types.ModuleType("backbone"), "backbone.vit_cllora": official,
                   "models": types.ModuleType("models"), "models.cllora": learner_module}
        x = torch.randn(2, 3, 8)
        with host_modules(modules):
            original = Adapter.forward
            install_cl()
            net = Learner({})._network.backbone
            shared = net.cur_adapter[0][0]
            with torch.no_grad():
                shared.lora_A.weight.normal_()
            torch.testing.assert_close(shared(x), original(shared, x), rtol=0, atol=0)
            before = shared(x).detach()
            net.add_adapter_to_list()
            torch.testing.assert_close(shared(x), before)
            teacher = net.saved[-1][0][0]
            torch.testing.assert_close(teacher(x), before)
            with torch.no_grad():
                shared.lora_A.weight.add_(.1)
            changed = shared(x).detach()
            self.assertFalse(torch.allclose(changed, before))
            torch.testing.assert_close(teacher(x), before)
            private = net.cur_adapter[1][0]
            with torch.no_grad():
                private.lora_A.weight.normal_()
            self.assertEqual(torch.count_nonzero(cl_effective(private) * private.dm_gate.protect), 0)
            private_before = private(x).detach()
            net.add_adapter_to_list()
            torch.testing.assert_close(shared(x), changed)
            torch.testing.assert_close(net.saved[-1][1][0](x), private_before)
            # A zero task increment still has a learning gradient.
            shared(x).square().mean().backward()
            self.assertGreater(shared.lora_A.weight.grad.norm(), 0)


class SDTests(unittest.TestCase):
    def test_task0_identity_and_identity_gate_reference_gradients(self):
        official = official_module("sd", "backbone/lora.py", {"_LoRA_qkv_timm_train", "ParameterWrapper"})
        original_forward = official._LoRA_qkv_timm_train.forward
        modules = {"backbone": types.ModuleType("backbone"), "backbone.lora": official}
        with host_modules(modules), patch.object(nn.Module, "cuda", lambda self: self):
            install_sd()
            def make(task, saved_a, saved_b):
                linears = [nn.Linear(8, 2, bias=False), nn.Linear(2, 8, bias=False),
                           nn.Linear(8, 2, bias=False), nn.Linear(2, 8, bias=False)]
                current = nn.ModuleList([official.ParameterWrapper(nn.Parameter(torch.tensor([.8])))])
                old = nn.ModuleList([official.ParameterWrapper(nn.Parameter(torch.tensor([.7]))) for _ in range(3)])
                return official._LoRA_qkv_timm_train(nn.Linear(8, 24), *linears, task, saved_a, saved_b, 0, 2, current, old)
            x = torch.randn(2, 3, 8)
            zero = make(0, {}, {})
            torch.testing.assert_close(zero(x), original_forward(zero, x), rtol=0, atol=0)
            saved_a = {"saved_A_0": [zero.linear_a_q, zero.linear_a_v]}
            saved_b = {"saved_B_0": [zero.linear_b_q, zero.linear_b_v]}
            net = make(1, saved_a, saved_b)
            reference = copy.deepcopy(net)
            net.linear_b_q.dm_gate = nn.Identity()
            net.linear_b_v.dm_gate = nn.Identity()
            rng = torch.get_rng_state().clone()
            output = net(x)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(rng)
            expected = original_forward(reference, x)
            torch.testing.assert_close(torch.get_rng_state(), after)
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
            output.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(net.linear_a_q.weight.grad, reference.linear_a_q.weight.grad)
            torch.testing.assert_close(net.scaling_factor_prev[0].param.grad, reference.scaling_factor_prev[0].param.grad)
            self.assertIsNone(zero.linear_a_q.weight.grad)

    def test_saved_direction_gate_not_lost_or_applied_twice(self):
        a, b = nn.Linear(8, 2, bias=False), nn.Linear(2, 8, bias=False)
        b.dm_gate = DualMask(torch.randn(8, 8), "single")
        expected = sd_kernel(a, b).detach()
        import io
        stream = io.BytesIO()
        torch.save((a, b), stream)
        stream.seek(0)
        a2, b2 = torch.load(stream, weights_only=False)
        torch.testing.assert_close(sd_kernel(a2, b2), expected)
        torch.testing.assert_close(b2.weight, b.weight)


class LauncherTests(unittest.TestCase):
    def run_queue(self, fail_smoke=False, smoke_only=False):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            trace = folder / "trace.txt"
            fake = folder / "python"
            fake.write_text(f"#!{sys.executable}\nimport os, sys\n"
                            "with open(os.environ['TRACE'], 'a') as f: f.write(' '.join(sys.argv[1:])+'\\n')\n"
                            "sys.exit(1 if os.environ.get('FAIL_SMOKE') == '1' and '--smoke' in sys.argv and 'cl' in sys.argv and 'on' in sys.argv else 0)\n")
            fake.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(fake), CL_PYTHON=str(fake), SD_PYTHON=str(fake),
                       TRACE=str(trace), FAIL_SMOKE=str(int(fail_smoke)), SMOKE_ONLY=str(int(smoke_only)))
            result = subprocess.run(["bash", str(ROOT / "scripts/9_19_dualmask_transfer_3090.sh")], cwd=folder, env=env, capture_output=True, text=True)
            runs = [line for line in trace.read_text().splitlines() if "--mode" in line]
            return result, runs

    def test_four_full_runs_and_four_smokes(self):
        result, runs = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(runs), 8)
        self.assertEqual(sum("--smoke" in run for run in runs), 4)
        self.assertTrue(all("--data-path" not in run for run in runs))

    def test_failed_smoke_skips_only_its_full_pair(self):
        result, runs = self.run_queue(fail_smoke=True)
        self.assertNotEqual(result.returncode, 0)
        full = [line for line in runs if "--smoke" not in line]
        self.assertEqual(len(full), 2)
        self.assertTrue(all("--host sd" in line for line in full))

    def test_smoke_only_does_not_start_full_training(self):
        result, runs = self.run_queue(smoke_only=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(runs), 4)
        self.assertTrue(all("--smoke" in line for line in runs))


if __name__ == "__main__":
    unittest.main()
