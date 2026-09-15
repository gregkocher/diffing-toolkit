"""Protocol adapter for the reference Jacobian estimator on native Ouro.

A residual block is shared across recurrence calls. Separate identity modules
provide hook boundaries without replacing the native recurrent computation.
Boundary r is the final block output of recurrence r, before the recurrent
normalization. In particular, downstream Jacobians include every intervening
normalization and subsequent recurrence.
"""
from __future__ import annotations

import torch
from torch import nn


class OuroLensModel:
    def __init__(self, hf_model, tokenizer):
        self.hf_model = hf_model.eval()
        self.tokenizer = tokenizer
        for p in hf_model.parameters():
            p.requires_grad_(False)
        self.text = hf_model.model
        self.n_layers = int(self.text.total_ut_steps)
        if self.n_layers != 4:
            raise ValueError("This campaign requires the native four-pass Ouro model")
        self.d_model = int(hf_model.config.hidden_size)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(self.n_layers)])
        self.visits = []
        self._hook = self.text.layers[-1].register_forward_hook(self._boundary, with_kwargs=True)

    def _boundary(self, module, args, kwargs, output):
        recurrence = int(kwargs["current_ut"])
        if recurrence != len(self.visits):
            raise RuntimeError(f"Unexpected recurrence order: {self.visits}, {recurrence}")
        self.visits.append(recurrence)
        if not torch.is_tensor(output):
            raise TypeError("Expected native Ouro block to return a residual tensor")
        return self.layers[recurrence](output)

    def encode(self, text, *, max_length=128):
        return self.tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=max_length).input_ids.to(self.text.embed_tokens.weight.device)

    def forward(self, input_ids):
        self.visits = []
        result = self.text(input_ids=input_ids, use_cache=False)
        if self.visits != list(range(self.n_layers)):
            raise RuntimeError(f"Incomplete native recurrence: {self.visits}")
        return result

    def unembed(self, residual):
        return self.hf_model.lm_head(self.text.norm(residual))

    def close(self):
        self._hook.remove()
