"""J-vector steering / patching (the post's main causal tool).

Patch applies, at each selected layer l and every position, the directional swap
    h <- h + alpha * relu(<h, u_from_l>) * (u_to_l - u_from_l)
where u_from_l, u_to_l are UNIT directions in layer l's residual space (normalized J-vectors).
This removes the component the state was pushing toward `from` and injects a `to` push,
scaled by how strongly `from` was present -- the "France -> China" swap.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Patch:
    def __init__(self, lm, layers, u_from: dict, u_to: dict, alpha: float):
        self.lm = lm
        self.layers = list(layers)
        self.u_from = u_from  # layer -> [d] unit tensor (device, dtype)
        self.u_to = u_to
        self.alpha = alpha
        self._handles = []

    def __enter__(self):
        def make(l):
            uf, ut = self.u_from[l], self.u_to[l]
            delta = ut - uf
            a = self.alpha

            def hook(_mod, _args, out):
                is_tuple = isinstance(out, tuple)
                h = out[0] if is_tuple else out
                coef = F.relu(h @ uf)  # [B, S]
                h = h + a * coef.unsqueeze(-1) * delta
                return (h,) + tuple(out[1:]) if is_tuple else h

            return hook

        for l in self.layers:
            self._handles.append(self.lm.layers[l].register_forward_hook(make(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class ProjectOut:
    """At each given layer, remove the residual's component along an orthonormal basis:
    h <- h - (h @ B^T) @ B, where B[layer] is [k, d] with orthonormal rows. Used to ablate the
    J-space subspace (or a random control subspace of equal dimension)."""

    def __init__(self, lm, bases: dict):
        self.lm = lm
        self.bases = bases  # layer -> [k, d] orthonormal (device, dtype)
        self._handles = []

    def __enter__(self):
        def make(l):
            B = self.bases[l]

            def hook(_mod, _args, out):
                is_tuple = isinstance(out, tuple)
                h = out[0] if is_tuple else out
                h = h - (h @ B.t()) @ B
                return (h,) + tuple(out[1:]) if is_tuple else h

            return hook

        for l in self.bases:
            self._handles.append(self.lm.layers[l].register_forward_hook(make(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def unit_jvectors(V: dict, layers, row: int, device, dtype) -> dict:
    """From compute_jvectors output V[l] (rows = tokens), take `row` at each layer, unit-normalize."""
    out = {}
    for l in layers:
        v = V[l][row]
        out[l] = (v / (v.norm() + 1e-12)).to(device).to(dtype)
    return out
