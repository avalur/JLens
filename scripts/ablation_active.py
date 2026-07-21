"""Faithful J-space ablation / selectivity (Nikolenko's post, property 5).

Per-input ACTIVE J-space: at each position and workspace layer, find the top-k J-vectors the state
is pushing toward (largest positive <h, v_hat_y> over the dictionary V_l = W_U @ J_l) and ZERO those
projections: h <- h - sum_{active} relu(<h,v_hat>) v_hat. Applied throughout multi-token GENERATION so
intermediate write->read routing is disrupted. Control = a fixed RANDOM set of k J-vectors (equal size).

Expectation (post): J-space ablation collapses MULTI-STEP REASONING (analogy chains, transitive, logic,
translation) while sparing AUTOMATIC abilities (recall, antonyms, induction, morphology); the random
control does neither. Grade = does the correct answer appear in the greedy generation.

Uses the cached full Jacobian (scripts/ablation.py builds it); no new backprop. Model scale matters --
the paper's result is on much larger models, so reproduction at 1.5B/7B is genuinely uncertain.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens import models  # noqa: E402
from jlens.jacobian import compute_jacobian  # noqa: E402
from jlens.corpus import wiki_texts, stream_batches  # noqa: E402

NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEV = os.environ.get("JLENS_DEVICE", "mps")
K = int(os.environ.get("JLENS_K", "25"))              # active J-vectors per position (paper ~25)
M = int(os.environ.get("JLENS_M", "50000"))           # J-vector dictionary size (vocab subset)
NGEN = int(os.environ.get("JLENS_NGEN", "24"))        # tokens to greedily generate
BAND = (float(os.environ.get("JLENS_D0", "0.35")), float(os.environ.get("JLENS_D1", "0.75")))

REASONING = [
    ("analogy-chain", "Paris is to France as Tokyo is to Japan as Cairo is to", "egypt"),
    ("two-hop", "The capital of the country where the Eiffel Tower is located is", "paris"),
    ("transitive", "Anna is taller than Bob. Bob is taller than Carl. So the shortest person is", "carl"),
    ("day-logic", "If today is Monday, then the day after tomorrow will be", "wednesday"),
    ("translate", "Translate the phrase 'good morning' into Spanish:", "buenos"),
]
AUTOMATIC = [
    ("recall", "The capital of Japan is", "tokyo"),
    ("antonym", "The opposite of the word 'big' is", "small"),
    ("induction", "red blue green red blue green red blue", "green"),
    ("plural", "One mouse, two", "mice"),
    ("recall2", "The largest planet in our solar system is", "jupiter"),
]

print(f"Loading {NAME} on {DEV} ...", flush=True)
lm = models.load(NAME, dtype=torch.float32, device=DEV)
L, d = lm.num_layers, lm.d
layers = [l for l in range(L) if BAND[0] <= l / (L - 1) <= BAND[1]]
print(f"  family={lm.family} layers={L} d={d} | ablate layers {layers[0]}-{layers[-1]} k={K} M={M}\n",
      flush=True)

cache = f"/tmp/jlens_J_{NAME.split('/')[-1]}.pt"
if os.path.exists(cache):
    print(f"loading cached Jacobian {cache}", flush=True)
    J = torch.load(cache)
else:
    avg = torch.cat(stream_batches(lm.tokenizer, wiki_texts(120), n_frag=16, seq_len=32,
                                   micro_bs=16, device=DEV), dim=0)
    print(f"computing full Jacobian over {avg.shape} ({d} backward passes) ...", flush=True)
    J, _, _ = compute_jacobian(lm, [avg], rows=None, layers=None, normalize=True, verbose=True)
    torch.save(J, cache)

# Per-layer unit J-vector dictionary V_hat_l = normalize_rows(W_U[:M] @ J_l).
WU = lm.W_U.detach().float()[:M].to(DEV)  # [M, d]
Vhat = {}
for l in layers:
    Vl = WU @ J[l].float().to(DEV)                 # [M, d]
    Vhat[l] = Vl / (Vl.norm(dim=-1, keepdim=True) + 1e-8)
g = torch.Generator().manual_seed(0)
rand_idx = {l: torch.randperm(M, generator=g)[:K].to(DEV) for l in layers}


def make_hook(l, mode):
    Vd = Vhat[l]
    eye = torch.eye(K)

    def hook(_mod, _args, out):
        is_t = isinstance(out, tuple)
        h = out[0] if is_t else out                       # [B, S, d]
        B, S, _ = h.shape
        if mode == "jspace":                              # per-position active set (top-k by corr)
            _, idx = (h @ Vd.t()).topk(K, dim=-1)         # [B, S, K]
            Vsel = Vd[idx]                                # [B, S, K, d]
        else:                                             # fixed random control subspace
            Vsel = Vd[rand_idx[l]].view(1, 1, K, -1).expand(B, S, K, -1)
        # exact orthogonal projection of h onto span(Vsel), then remove it
        rhs = torch.einsum("bskd,bsd->bsk", Vsel, h)                       # <v_i, h>
        G = torch.einsum("bskd,bsjd->bskj", Vsel, Vsel)                    # Gram [B,S,K,K]
        c = torch.linalg.solve((G + 1e-3 * eye.to(G)).cpu(), rhs.cpu().unsqueeze(-1))
        proj = torch.einsum("bsk,bskd->bsd", c.squeeze(-1).to(h), Vsel)   # [B,S,d]
        h = h - proj
        return (h,) + tuple(out[1:]) if is_t else h

    return hook


class Ablate:
    def __init__(self, mode):
        self.mode = mode
        self._h = []

    def __enter__(self):
        if self.mode:
            for l in layers:
                self._h.append(lm.layers[l].register_forward_hook(make_hook(l, self.mode)))
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h = []


@torch.no_grad()
def generate(prompt, mode=None):
    ids = lm.tokenizer(prompt, return_tensors="pt").input_ids.to(DEV)
    start = ids.shape[1]
    with Ablate(mode):
        for _ in range(NGEN):
            nxt = lm.model(input_ids=ids, use_cache=False).logits[0, -1].argmax()
            if int(nxt) == lm.tokenizer.eos_token_id:
                break
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return lm.tokenizer.decode(ids[0, start:], skip_special_tokens=True)


def run(group, title):
    print(f"=== {title} ===")
    hits = {"clean": 0, "jspace": 0, "control": 0}
    for tname, prompt, ans in group:
        outs = {m or "clean": generate(prompt, m) for m in (None, "jspace", "control")}
        for key in hits:
            if ans in outs[key].lower():
                hits[key] += 1
        mark = {k: ("OK" if ans in outs[k].lower() else "--") for k in hits}
        print(f"[{tname}] answer='{ans}'")
        for k in ("clean", "jspace", "control"):
            print(f"    {mark[k]} {k:<8}: {outs[k].strip()[:70]!r}")
    n = len(group)
    print(f"  ANSWER-PRESENT RATE:  clean={hits['clean']}/{n}  "
          f"J-space-ablated={hits['jspace']}/{n}  random-control={hits['control']}/{n}\n")


run(REASONING, "MULTI-STEP REASONING (expect J-space ablation to collapse, control to spare)")
run(AUTOMATIC, "AUTOMATIC ABILITIES (expect all three to survive)")
