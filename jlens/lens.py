"""Lens read-outs, family-aware.

The J-lens is *defined* as softmax(W_U . norm([J] h)); `logit_lens`/`j_lens` return the PRE-SOFTMAX
LOGITS of that expression (apply torch.softmax, or use `top_tokens`, for probabilities — the top-k
ranking is identical either way, and argmax works directly on the logits).

`norm` is the model's OWN final RMSNorm module (lm.final_norm), so the learned gain,
the Gemma (1+weight) parameterization, the float32 upcast, and eps are all inherited for
free. The optional Gemma-2 final logit soft-cap (30*tanh(x/30)) is monotonic: it changes
probabilities but NOT top-k ordering, so it is off by default and only matters for faithful
probabilities.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _readout(lm, resid: torch.Tensor, apply_softcap: bool = False) -> torch.Tensor:
    """resid: [..., d] residual in FINAL-residual space -> logits [..., V]."""
    normed = lm.final_norm(resid.to(lm.final_norm.weight.dtype).to(lm.device))
    logits = F.linear(normed, lm.W_U)
    cap = lm.final_logit_softcap
    if apply_softcap and cap:
        logits = cap * torch.tanh(logits / cap)
    return logits


def logit_lens(lm, h_l: torch.Tensor, apply_softcap: bool = False) -> torch.Tensor:
    """Standard logit lens: read a raw intermediate residual as if it were final."""
    return _readout(lm, h_l, apply_softcap)


def j_lens(lm, h_l: torch.Tensor, J_l: torch.Tensor, apply_softcap: bool = False) -> torch.Tensor:
    """J-lens: map h_l through the averaged linearization J_l, then read.

    J_l[i, j] has row index i = output/final-residual coord, so (J_l @ h)_i = sum_j J[i,j] h[j].
    """
    J = J_l.to(h_l.dtype).to(h_l.device)
    Jh = torch.einsum("ij,...j->...i", J, h_l.to(J.dtype))
    return _readout(lm, Jh, apply_softcap)


def top_tokens(lm, logits: torch.Tensor, k: int = 8):
    """Return list of (token_str, prob) for the top-k of a 1-D logits vector."""
    probs = torch.softmax(logits.float(), dim=-1)
    vals, idx = probs.topk(k)
    return [(lm.tokenizer.decode([int(i)]), float(v)) for v, i in zip(vals, idx)]
