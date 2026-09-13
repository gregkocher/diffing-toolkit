"""CPU tests for intervention restoration and recurrence identity; no downloads."""
import unittest
from types import SimpleNamespace
import torch
from .native import recurrence_adapter_mask, recurrence_states, select_positions


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = {}
        self.disable_adapters = False
        self.merged = False

    def enable_adapters(self, enabled):
        self.disable_adapters = not enabled

    def forward(self, x):
        return x if self.disable_adapters else x + 1


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()

    def forward(self, x, current_ut=None):
        return self.adapter(x)


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer()])
        self.total_ut_steps = 4

    def forward(self, input_ids, use_cache=False):
        x = input_ids.float()
        states = []
        for loop in range(4):
            x = self.layers[0](x, current_ut=loop)
            states.append(x)
        return None, states, [None]*4


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.model = Core()

    def test_loop_mask_preserves_all_executions_and_restores(self):
        inputs = {"input_ids": torch.tensor([0])}
        with recurrence_adapter_mask(self.model, [True, False, True, False]) as visits:
            states = recurrence_states(self.model, inputs)
        self.assertEqual(visits, [0,1,2,3])
        self.assertEqual([x.item() for x in states], [1,1,2,2])
        self.assertEqual(recurrence_states(self.model, inputs)[-1].item(), 4)
        self.assertFalse(self.model.model.layers[0]._forward_pre_hooks)

    def test_exception_restores_original_disabled_state(self):
        self.model.model.layers[0].adapter.enable_adapters(False)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with recurrence_adapter_mask(self.model, [True]*4):
                self.model.model.layers[0](torch.tensor([0]), current_ut=0)
                raise RuntimeError("test failure")
        self.assertTrue(self.model.model.layers[0].adapter.disable_adapters)
        self.assertFalse(self.model.model.layers[0]._forward_pre_hooks)

    def test_incomplete_forward_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "complete uncached"):
            with recurrence_adapter_mask(self.model, [True]*4):
                self.model.model.layers[0](torch.tensor([0]), current_ut=0)
        self.assertFalse(self.model.model.layers[0].adapter.disable_adapters)

    def test_positions_have_next_token_and_no_duplicates(self):
        self.assertEqual(select_positions(2,8), [0])
        positions = select_positions(128,8)
        self.assertEqual(len(positions),8)
        self.assertEqual((positions[0],positions[-1]),(0,126))
        self.assertEqual(len(set(positions)),8)


if __name__ == "__main__":
    unittest.main()
