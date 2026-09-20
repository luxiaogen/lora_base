import copy
import unittest
import torch
from test.test_p_region_training import make_attention
from utils.p_input_subspace import update_basis, conflict_subspace_loss, collect_input_bases


class InputSubspaceTests(unittest.TestCase):
    def test_collection_preserves_rng_parameters_and_training_mode(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = make_attention()
                self.attn.before_task(0)
                self.attn.after_task(0)

            def interface(self, x):
                return self.attn(x, 0)

        network = Network().train()
        x = torch.randn(4, 3, 4)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(4), x, torch.zeros(4)), batch_size=2)
        before = {key: value.clone() for key, value in network.state_dict().items()}
        state = torch.get_rng_state().clone()
        collect_input_bases(network, [network.attn], loader, torch.device('cpu'), rank=2)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(network.training and network.attn.training)
        for key, value in network.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(len(network.attn._forward_pre_hooks), 0)

    def test_basis_and_rng(self):
        module = make_attention()
        state = torch.get_rng_state().clone()
        update_basis(module, torch.diag(torch.tensor([4., 3., 2., 1.])), 2, 0, 1993)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for basis in (module._p_input_basis, module._p_random_basis):
            torch.testing.assert_close(basis.T @ basis, torch.eye(2))
        torch.testing.assert_close(module._p_input_basis @ module._p_input_basis.T, torch.diag(torch.tensor([1., 1., 0., 0.])))

    def test_only_p_gradients_and_no_forward_change(self):
        module = make_attention()
        module.before_task(0); module.after_task(0); module.before_task(1)
        module.general_mask.zero_()
        module.dual_mask_conflict_strength = .5
        with torch.no_grad():
            module.P_lora[1].A.weight.normal_(0, .1)
            module.P_lora[1].B.weight.normal_(0, .1)
        reference = copy.deepcopy(module)
        update_basis(module, torch.eye(4), 2, 0, 1993)
        x = torch.randn(3, 2, 4)
        module.eval(); reference.eval()
        torch.testing.assert_close(module(x, 1), reference(x, 1), rtol=0, atol=0)
        for mode in ("old", "random"):
            module.zero_grad()
            loss = conflict_subspace_loss(module, mode)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(module.P_lora[1].B.weight.grad.norm().item(), 0)
            self.assertIsNone(module.S_lora[1].B.weight.grad)
        module.after_task(1); reference.after_task(1)
        torch.testing.assert_close(module.qkv.weight, reference.qkv.weight, rtol=0, atol=0)

    def test_zero_b_and_task0(self):
        module = make_attention()
        module.before_task(0)
        self.assertIsNone(conflict_subspace_loss(module, "old"))
        module.after_task(0); module.before_task(1)
        update_basis(module, torch.eye(4), 2, 0, 1993)
        loss = conflict_subspace_loss(module, "old")
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(module.P_lora[1].B.weight.grad).all())
        self.assertIsNone(conflict_subspace_loss(module, "baseline"))
