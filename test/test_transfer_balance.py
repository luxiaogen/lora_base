import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from test.test_dualmask_transfer import official_module, host_modules
from dualmask_transfer.core import DualMask
from dualmask_transfer.hosts import install_sd_checkpoint


class BalanceTests(unittest.TestCase):
    def test_strength_changes_shared_only(self):
        weight, delta = torch.randn(8, 8), torch.randn(8, 8)
        strong, weak = DualMask(weight, "shared"), DualMask(weight, "shared", .25)
        torch.testing.assert_close(weak(delta) * weak.protect, 1.5 * strong(delta) * strong.protect)
        torch.testing.assert_close(weak(delta) * (1-weak.protect), strong(delta) * (1-strong.protect))
        torch.testing.assert_close(DualMask(weight, "private", .25)(delta), DualMask(weight, "private")(delta))

    def test_sd_task5_checkpoint_output_gradients_and_rng(self):
        official = official_module("sd", "backbone/lora.py", {"_LoRA_qkv_timm_train", "ParameterWrapper"})
        original = official._LoRA_qkv_timm_train.forward
        with host_modules({"backbone.lora": official}), patch.object(nn.Module, "cuda", lambda self: self):
            install_sd_checkpoint()
            saved_a = {"saved_A_"+str(t): [nn.Linear(8, 2, bias=False) for _ in range(2)] for t in range(5)}
            saved_b = {"saved_B_"+str(t): [nn.Linear(2, 8, bias=False) for _ in range(2)] for t in range(5)}
            def scales(count):
                return nn.ModuleList([official.ParameterWrapper(nn.Parameter(torch.tensor([.7]))) for _ in range(count)])
            linears = [nn.Linear(8, 2, bias=False), nn.Linear(2, 8, bias=False), nn.Linear(8, 2, bias=False), nn.Linear(2, 8, bias=False)]
            candidate = official._LoRA_qkv_timm_train(nn.Linear(8, 24), *linears, 5, saved_a, saved_b, 0, 2, scales(1), scales(5))
            reference = copy.deepcopy(candidate)
            x = torch.randn(2, 3, 8, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            state = torch.get_rng_state().clone()
            expected = original(reference, xr)
            expected.square().sum().backward()
            after = torch.get_rng_state().clone()
            torch.set_rng_state(state)
            result = candidate(x)
            result.square().sum().backward()
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, xr.grad)
            for (name, p), (_, pr) in zip(candidate.named_parameters(), reference.named_parameters()):
                if pr.grad is not None:
                    torch.testing.assert_close(p.grad, pr.grad, msg=name)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
