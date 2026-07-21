"""Averaged-Jacobian ("J-lens") computation via reverse-mode VJPs.

The trick (verified correct):
  Add a zero leaf delta_l (broadcast over positions) at each layer l's output.
  With G_i = sum_{t'} h_{L,t',i} on the RAW final residual,
      G_i.backward()  =>  delta_l.grad == sum_t sum_{t'>=t} d h_{L,t',i} / d h_{l,t}
                       == row i of the pair-summed Jacobian, for EVERY layer at once.
  Causal masking auto-zeros t'<t. Orientation: J_l[i, :] = delta_l.grad  (row = output coord).

d backward passes give the full d x d J_l for all layers. For a targeted token y,
  s = sum_{t'} (W_U[y] . h_{L,t'})  ;  s.backward()  =>  delta_l.grad == v_y (one VJP per token).
"""

from __future__ import annotations

import torch

from .hooks import JHooks


def _causal_pairs(seq_len: int) -> int:
    # number of (t, t') with t' >= t for one sequence
    return seq_len * (seq_len + 1) // 2


def compute_jacobian(
    lm,
    batches,
    rows=None,
    layers=None,
    normalize: bool = True,
    verbose: bool = True,
):
    """Corpus-averaged Jacobian.

    Args:
        batches: iterable of input_ids tensors [B, S] (unpadded; fixed S recommended).
        rows: output coords i to compute (default: all d). Restricting rows is cheaper.
        layers: layer indices to keep (default: all). Cost is independent of this.
        normalize: divide by total causal pairs to get the average (paper's E[.]).
    Returns:
        (J, rows, total_pairs) where J[l] is fp32 [len(rows), d] on CPU.
    """
    d, L = lm.d, lm.num_layers
    rows = list(range(d)) if rows is None else list(rows)
    layers = list(range(L)) if layers is None else list(layers)
    J = {l: torch.zeros(len(rows), d, dtype=torch.float32) for l in layers}
    total_pairs = 0

    with JHooks(lm) as hk:
        for b_idx, ids in enumerate(batches):
            ids = ids.to(lm.device)
            hk.zero_grads()
            lm.base(input_ids=ids, use_cache=False)  # base stack; norm pre-hook grabs h_L
            h_L = hk.h_L                              # [B, S, d], in-graph
            G = h_L.sum(dim=(0, 1))                   # [d]
            B, S = ids.shape[0], ids.shape[1]
            total_pairs += B * _causal_pairs(S)

            step = max(1, len(rows) // 8)
            for ri, i in enumerate(rows):
                hk.zero_grads()
                G[i].backward(retain_graph=True)
                for l in layers:
                    g = hk.deltas[l].grad
                    if g is not None:
                        J[l][ri] += g.detach().float().cpu()
                if verbose and len(rows) > 64 and ri % step == 0:
                    print(f"    row {ri}/{len(rows)}", flush=True)

            del G, h_L
            if lm.device == "mps":
                torch.mps.empty_cache()
            if verbose:
                print(f"  batch {b_idx + 1}: rows={len(rows)} pairs+={B * _causal_pairs(S)}")

    if normalize:
        denom = max(total_pairs, 1)
        for l in layers:
            J[l] /= denom
    return J, rows, total_pairs


def compute_jvectors(lm, batches, token_ids, layers=None, normalize: bool = True):
    """Per-token J-vectors v_y = row y of (W_U J_l), via one VJP per token.

    Returns (V, total_pairs) where V[l] is fp32 [len(token_ids), d] on CPU.
    Note: this treats the final norm as pass-through (matches the paper's v_y definition,
    which omits the final-norm Jacobian).
    """
    L = lm.num_layers
    layers = list(range(L)) if layers is None else list(layers)
    token_ids = list(token_ids)
    V = {l: torch.zeros(len(token_ids), lm.d, dtype=torch.float32) for l in layers}
    total_pairs = 0
    WU = lm.W_U  # [V, d]

    with JHooks(lm) as hk:
        for ids in batches:
            ids = ids.to(lm.device)
            hk.zero_grads()
            lm.base(input_ids=ids, use_cache=False)
            h_L = hk.h_L  # [B, S, d]
            B, S = ids.shape[0], ids.shape[1]
            total_pairs += B * _causal_pairs(S)
            for yi, y in enumerate(token_ids):
                hk.zero_grads()
                s = (h_L * WU[y].to(h_L.dtype)).sum()  # sum over B, S, d
                s.backward(retain_graph=True)
                for l in layers:
                    g = hk.deltas[l].grad
                    if g is not None:
                        V[l][yi] += g.detach().float().cpu()
            del h_L
            if lm.device == "mps":
                torch.mps.empty_cache()

    if normalize:
        denom = max(total_pairs, 1)
        for l in layers:
            V[l] /= denom
    return V, total_pairs
