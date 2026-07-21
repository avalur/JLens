"""Correctness checks for the J-lens core. Run before trusting any result.

[1] structural         : len(hidden_states)==L+1 AND logit_lens(raw h_L) == model logits.
[2] jlens_identity     : j_lens(h, I) == logit_lens(h)  (einsum orientation sanity).
[3] j_last_identity    : normalized J_{L-1} ~ (2/(S+1)) * I  (hook order + normalization).
[4] finite_difference  : J_l @ v == central-difference directional derivative of sum_t' h_L.
                         This is the GOLD test: it checks both the VALUE and the ORIENTATION
                         of J (a transpose would fail), and because a silent all-zero MPS
                         Jacobian would make J@v==0 while the finite difference is nonzero,
                         it also guards against the documented torch/MPS silent-zero bug.
"""

from __future__ import annotations

import torch

from . import lens
from .hooks import JHooks
from .jacobian import compute_jacobian


def _capture_hL(lm, ids, want_logits=False):
    with JHooks(lm) as hk:
        if want_logits:
            out = lm.model(input_ids=ids.to(lm.device), use_cache=False,
                           output_hidden_states=True)
            return hk.h_L.detach(), out
        lm.base(input_ids=ids.to(lm.device), use_cache=False)
        return hk.h_L.detach(), None


def structural_check(lm, ids):
    h_L, out = _capture_hL(lm, ids, want_logits=True)
    n_hs = len(out.hidden_states)
    cap = lm.family == "gemma2"
    ll = lens.logit_lens(lm, h_L, apply_softcap=cap)
    diff = (ll - out.logits).abs().max().item()
    return {
        "n_hidden_states": n_hs,
        "expected": lm.num_layers + 1,
        "ok_len": n_hs == lm.num_layers + 1,
        "logit_lens_vs_model_maxdiff": diff,
        "ok_readout": diff < 1e-2,
    }


def jlens_identity_check(lm, ids):
    h_L, _ = _capture_hL(lm, ids)
    I = torch.eye(lm.d)
    a = lens.logit_lens(lm, h_L)
    b = lens.j_lens(lm, h_L, I)
    diff = (a - b).abs().max().item()
    return {"maxdiff": diff, "ok": diff < 1e-3}


def j_last_identity_check(lm, ids, n_rows=8):
    last = lm.num_layers - 1
    rows = list(range(n_rows))
    J, rows, pairs = compute_jacobian(
        lm, [ids], rows=rows, layers=[last], normalize=True, verbose=False
    )
    Jl = J[last]  # [n_rows, d]
    S = ids.shape[1]
    expected = 2.0 / (S + 1)
    diag = torch.stack([Jl[r, rows[r]] for r in range(n_rows)])
    off = Jl.clone()
    for r in range(n_rows):
        off[r, rows[r]] = 0.0
    return {
        "seq_len": S,
        "expected_diag": expected,
        "diag_mean": diag.mean().item(),
        "diag_min": diag.min().item(),
        "max_offdiag_abs": off.abs().max().item(),
        "ok": abs(diag.mean().item() - expected) < 0.15 * expected
        and off.abs().max().item() < 0.05 * expected,
    }


def finite_difference_check(lm, ids, layer=None, n_rows=16, eps=1e-2):
    d = lm.d
    layer = lm.num_layers // 2 if layer is None else layer
    g = torch.Generator().manual_seed(0)
    v = torch.randn(d, generator=g)
    v = (v / v.norm()).to(lm.dtype)

    with JHooks(lm) as hk:
        def sum_hL(scale):
            hk.set_delta(layer, (scale * v).to(lm.device))
            with torch.no_grad():
                lm.base(input_ids=ids.to(lm.device), use_cache=False)
                return hk.h_L.detach().sum(dim=(0, 1)).float().cpu()

        hp = sum_hL(+eps)
        hm = sum_hL(-eps)
        hk.set_delta(layer, None)
    fd = (hp - hm) / (2 * eps)  # [d] ~= (J_l^sum @ v)

    rows = list(range(n_rows))
    J, rows, pairs = compute_jacobian(
        lm, [ids], rows=rows, layers=[layer], normalize=False, verbose=False
    )
    Jl = J[layer]  # [n_rows, d]
    pred = Jl @ v.float().cpu()  # [n_rows]
    target = fd[rows]
    denom = target.norm().item() or 1.0
    rel = (pred - target).norm().item() / denom
    return {
        "layer": layer,
        "rel_err": rel,
        "pred_norm": pred.norm().item(),
        "fd_norm": target.norm().item(),
        "ok": rel < 5e-2 and pred.norm().item() > 1e-6,
    }


def run_all(lm, ids, verbose=True, fd_eps=1e-2):
    # fd_eps: central-difference step for the FD-JVP check. 1e-2 suits small models; larger
    # models (bigger residual/h_L magnitudes) need a larger step to avoid fp32 cancellation
    # (e.g. Gemma-2-9b is cancellation-dominated at 1e-2 but ~0.5% at 1e-1).
    results = {
        "structural": structural_check(lm, ids),
        "jlens_identity": jlens_identity_check(lm, ids),
        "j_last_identity": j_last_identity_check(lm, ids),
        "finite_difference": finite_difference_check(lm, ids, eps=fd_eps),
    }
    if verbose:
        for name, r in results.items():
            ok = r.get("ok", r.get("ok_len") and r.get("ok_readout"))
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {name}: {r}")
    all_ok = all(
        r.get("ok", (r.get("ok_len") and r.get("ok_readout"))) for r in results.values()
    )
    return all_ok, results
