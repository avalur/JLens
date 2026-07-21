"""Delta-injection hooks + raw final-residual capture.

Mechanics (verified in-env on transformers 5.13.0):
- Decoder layers return a BARE Tensor (not a tuple) in transformers 5.x, so the hook is
  `return output + delta`. A tuple branch is kept only as defensive fallback.
- Each `delta_l` is a zero-initialized leaf created ON device (else `.to(mps)` makes it a
  non-leaf and `.grad` never populates). Zero-init => the forward pass is unperturbed.
- `h_L` (the RAW, pre-final-norm residual the paper calls h_L) is captured as the *input*
  to `model.model.norm` via a forward_pre_hook, kept IN-GRAPH so we can backprop through it.
  (Note: output_hidden_states[-1] is POST final-norm, so it is NOT h_L.)
"""

from __future__ import annotations

import torch


class JHooks:
    def __init__(self, lm):
        self.lm = lm
        self.deltas: list[torch.Tensor] = []
        self.h_L: torch.Tensor | None = None          # in-graph, [B, S, d]
        self.layer_out: list[torch.Tensor | None] = []  # detached raw residual per layer
        self._handles: list = []

    def __enter__(self) -> "JHooks":
        d, dev, dt = self.lm.d, self.lm.device, self.lm.dtype
        L = self.lm.num_layers
        self.deltas = [
            torch.zeros(d, device=dev, dtype=dt, requires_grad=True) for _ in range(L)
        ]
        self.layer_out = [None] * L

        def make(i):
            delta = self.deltas[i]

            def hook(_mod, _args, out):
                if isinstance(out, tuple):  # defensive; not taken for Qwen2/3, Gemma2/3
                    new0 = out[0] + delta
                    self.layer_out[i] = new0.detach()
                    return (new0,) + tuple(out[1:])
                new = out + delta
                self.layer_out[i] = new.detach()
                return new

            return hook

        for i, lyr in enumerate(self.lm.layers):
            self._handles.append(lyr.register_forward_hook(make(i)))

        def norm_pre(_mod, args):
            self.h_L = args[0]  # raw residual entering the final norm, kept in-graph
            return None

        self._handles.append(self.lm.final_norm.register_forward_pre_hook(norm_pre))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def zero_grads(self):
        for dl in self.deltas:
            dl.grad = None

    def set_delta(self, layer: int, vec: torch.Tensor | None):
        """Set a layer's delta value in-place (for finite-difference / steering).
        Pass None to reset to zero."""
        with torch.no_grad():
            if vec is None:
                self.deltas[layer].zero_()
            else:
                self.deltas[layer].copy_(vec.to(self.deltas[layer]))
