"""Reproduce the Spanish->French SELECTIVITY result (Nikolenko's post, property 5, via patching).

Patch the language J-vector (Spanish -> French) on the workspace layers, then show ONE patch:
  - BREAKS the meta-question "what language is this text?" (answer flips Spanish -> French), but
  - LEAVES the continuation of the Spanish text coherent Spanish (unaffected).
Mechanism = the selectivity: the patch fires only where the state pushes toward the TOKEN 'Spanish'
(the language-ID answer), not while generating Spanish prose (which pushes toward the next Spanish word).

J-vectors are read from the cached full Jacobian (v_y[l] = W_U[y] @ J_l); no new backprop.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens import models  # noqa: E402
from jlens.jacobian import compute_jacobian  # noqa: E402
from jlens.steering import Patch  # noqa: E402
from jlens.corpus import wiki_texts, stream_batches  # noqa: E402
from jlens.lens import top_tokens  # noqa: E402

NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEV = os.environ.get("JLENS_DEVICE", "mps")
ALPHAS = [float(a) for a in os.environ.get("JLENS_ALPHAS", "0,2,3,4,6,8").split(",")]
PATCH_LAYERS = list(range(9, 20))
NGEN = 20

SPANISH = ("El sol brillaba sobre las montañas mientras los pájaros cantaban en los "
           "árboles. María caminaba despacio por el sendero disfrutando del aire fresco")

print(f"Loading {NAME} on {DEV} ...", flush=True)
lm = models.load(NAME, dtype=torch.float32, device=DEV)
L, d = lm.num_layers, lm.d
print(f"  family={lm.family} layers={L} d={d}\n", flush=True)

# cached full Jacobian (built by the ablation scripts); compute if absent
cache = f"/tmp/jlens_J_{NAME.split('/')[-1]}.pt"
if os.path.exists(cache):
    J = torch.load(cache)
else:
    avg = torch.cat(stream_batches(lm.tokenizer, wiki_texts(120), n_frag=16, seq_len=32,
                                   micro_bs=16, device=DEV), dim=0)
    print(f"computing full Jacobian ({d} backward passes) ...", flush=True)
    J, _, _ = compute_jacobian(lm, [avg], rows=None, layers=None, normalize=True, verbose=True)
    torch.save(J, cache)


def tid(word):
    return lm.tokenizer(" " + word, add_special_tokens=False).input_ids[0]


sp_id, fr_id = tid("Spanish"), tid("French")
WU = lm.W_U.detach().float()

# language J-vectors from cached J: v_y[l] = W_U[y] @ J_l ; unit-normalized
u_sp, u_fr = {}, {}
for l in PATCH_LAYERS:
    Jl = J[l].float()
    vs = WU[sp_id].cpu() @ Jl
    vf = WU[fr_id].cpu() @ Jl
    u_sp[l] = (vs / vs.norm()).to(DEV).to(lm.dtype)
    u_fr[l] = (vf / vf.norm()).to(DEV).to(lm.dtype)


def patch_ctx(alpha):
    return Patch(lm, PATCH_LAYERS, u_sp, u_fr, alpha) if alpha > 0 else None


def chat_lang_ids():
    msg = [{"role": "user", "content":
            f'What language is the following text written in? Answer with a single word.\n\n"{SPANISH}"'}]
    out = lm.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                           return_tensors="pt")
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return ids.to(DEV)


LANG_IDS = chat_lang_ids()


@torch.no_grad()
def logits_of(ids, alpha):
    ctx = patch_ctx(alpha)
    if ctx:
        ctx.__enter__()
    out = lm.model(input_ids=ids, use_cache=False).logits[0, -1]
    if ctx:
        ctx.__exit__()
    return out


@torch.no_grad()
def generate(text, alpha):
    ids = lm.tokenizer(text, return_tensors="pt").input_ids.to(DEV)
    start = ids.shape[1]
    ctx = patch_ctx(alpha)
    if ctx:
        ctx.__enter__()
    for _ in range(NGEN):
        nxt = lm.model(input_ids=ids, use_cache=False).logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    if ctx:
        ctx.__exit__()
    return lm.tokenizer.decode(ids[0, start:], skip_special_tokens=True)


@torch.no_grad()
def continuation_agreement(text, alpha):
    """Per-position top-1 agreement (clean vs patched) over the Spanish text: high = undisrupted."""
    ids = lm.tokenizer(text, return_tensors="pt").input_ids.to(DEV)
    clean = lm.model(input_ids=ids, use_cache=False).logits[0].argmax(-1)
    ctx = patch_ctx(alpha)
    ctx.__enter__()
    patched = lm.model(input_ids=ids, use_cache=False).logits[0].argmax(-1)
    ctx.__exit__()
    return float((clean == patched).float().mean())


print("SELECTIVITY SWEEP: same Spanish->French patch on two tasks.")
print("  TASK A = 'what language is this?'  (want: flips Spanish->French)")
print("  TASK B = continue the Spanish prose (want: stays Spanish => high agreement)\n")
print(f"{'alpha':>5} | {'A p(Spanish)':>12} {'A p(French)':>11} {'A top-1':>12} | {'B agreement':>11}")
print("-" * 62)
for a in ALPHAS:
    lg = logits_of(LANG_IDS, a)
    p = torch.softmax(lg.float(), 0)
    a_top = top_tokens(lm, lg, 1)[0][0].strip()
    b_agr = 1.0 if a == 0 else continuation_agreement(SPANISH, a)
    print(f"{a:>5g} | {float(p[sp_id]):>12.3f} {float(p[fr_id]):>11.3f} {a_top!r:>12} | {b_agr:>10.1%}")

print("\nSample continuations of the Spanish text (Task B):")
print(f"  clean:          {generate(SPANISH, 0)!r}")
for a in [ALPHAS[len(ALPHAS) // 2], ALPHAS[-1]]:
    print(f"  patched a={a:g}:   {generate(SPANISH, a)!r}")
