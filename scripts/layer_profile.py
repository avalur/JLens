"""Reproduce the "layer profile / three zones" structural result (Nikolenko's post).

For each layer, measure how often the lens's top-1 token matches the MODEL's own final top-1
prediction, on held-out text. Expected shape: first third unreadable (~sensory zone), a slow rise
(~workspace), and sharp late convergence (~motor zone). Compare logit lens vs J-lens; the post notes
the logit lens can match the J-lens (or beat it) in the middle on small models, with the J-lens
pulling ahead in the mid-late layers on larger ones.

Requires the FULL averaged d x d Jacobian (the J-lens is read at many positions), so this computes
J_l via d reverse-mode backward passes over a small averaging corpus (the M2 step).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens import models  # noqa: E402
from jlens.hooks import JHooks  # noqa: E402
from jlens.jacobian import compute_jacobian  # noqa: E402

NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEV = os.environ.get("JLENS_DEVICE", "mps")
S = int(os.environ.get("JLENS_S", "16"))
N_AVG = int(os.environ.get("JLENS_NAVG", "8"))   # fragments to average J over (cost ~ N_AVG*S)
N_EVAL = int(os.environ.get("JLENS_NEVAL", "24"))

# Disjoint generic text pools: AVG_POOL for the Jacobian average, EVAL_POOL held out for scoring.
AVG_POOL = [
    "The history of ancient civilizations spans many thousands of years of profound change.",
    "Modern computers rely on billions of tiny transistors etched onto small silicon chips.",
    "Chemical reactions can release or absorb energy in the form of heat and light.",
    "Rivers carry sediment downstream and slowly reshape the surrounding valleys and plains.",
    "Economic policy often balances the rate of inflation against the level of employment.",
    "The orchestra carefully tuned their many instruments before the evening performance began.",
    "Volcanic eruptions can send enormous clouds of ash high into the upper atmosphere.",
    "Farmers rotate their crops each season to keep the soil fertile and highly productive.",
]
EVAL_POOL = [
    "Many species of migratory birds travel south together during the coldest winter months.",
    "The old novel describes a difficult journey across a vast and nearly empty desert.",
    "Scientists patiently collected detailed data from the large telescope over several clear nights.",
    "The city council debated the controversial new housing proposal for nearly four long hours.",
    "The suspension bridge was carefully engineered to withstand strong winds and heavy daily traffic.",
    "A balanced diet generally includes proteins, carbohydrates, healthy fats, and many vitamins.",
    "Light from the most distant stars takes many millions of years to reach our small planet.",
    "Young children gradually learn language by listening to and imitating the people around them.",
]


def chunk_stream(tok, texts, seq_len, n_max):
    ids = []
    for t in texts:
        ids += tok(t, add_special_tokens=False).input_ids
    chunks = [torch.tensor(ids[i:i + seq_len]).unsqueeze(0)
              for i in range(0, len(ids) - seq_len + 1, seq_len)]
    return chunks[:n_max]


print(f"Loading {NAME} on {DEV} ...", flush=True)
lm = models.load(NAME, dtype=torch.float32, device=DEV)
L, d = lm.num_layers, lm.d
print(f"  family={lm.family} layers={L} d={d}\n", flush=True)

# --- full averaged Jacobian over a small corpus ---------------------------------------------
avg_chunks = chunk_stream(lm.tokenizer, AVG_POOL, S, N_AVG)
avg_batch = torch.cat(avg_chunks, dim=0).to(DEV)  # [N_AVG, S]
print(f"computing full d x d Jacobian over {avg_batch.shape[0]} x {S} tokens "
      f"({d} backward passes) ...", flush=True)
t0 = time.time()
J, _, pairs = compute_jacobian(lm, [avg_batch], rows=None, layers=None, normalize=True, verbose=True)
print(f"  J computed in {time.time() - t0:.1f}s\n", flush=True)

# --- held-out eval: collect residuals + model's own top-1 per position ----------------------
eval_chunks = chunk_stream(lm.tokenizer, EVAL_POOL, S, N_EVAL)
H = {l: [] for l in range(L)}
model_top1 = []
with JHooks(lm) as hk:
    for ids in eval_chunks:
        ids = ids.to(DEV)
        with torch.no_grad():
            out = lm.model(input_ids=ids, use_cache=False)
        model_top1.append(out.logits[0].argmax(-1).cpu())     # [S]
        for l in range(L):
            H[l].append(hk.layer_out[l][0].float().cpu())      # [S, d]
model_top1 = torch.cat(model_top1)                             # [P]
for l in range(L):
    H[l] = torch.cat(H[l], 0)                                  # [P, d]
P = model_top1.shape[0]
print(f"scoring on {len(eval_chunks)} held-out fragments = {P} positions\n", flush=True)

WU = lm.W_U.detach()  # [V, d] on device
CH = 128


@torch.no_grad()
def lens_top1(resid_Pd):  # [P, d] on device -> [P] argmax over vocab (cpu)
    normed = lm.final_norm(resid_Pd)
    outs = []
    for s in range(0, normed.shape[0], CH):
        outs.append((normed[s:s + CH] @ WU.t()).argmax(-1))
    return torch.cat(outs).cpu()


print(f"{'lyr':>3} {'depth':>6} | {'logit-lens match':>16} | {'J-lens match':>13}")
print("-" * 48)
for l in range(L):
    Hl = H[l].to(DEV).to(lm.dtype)
    Jl = J[l].to(DEV).to(lm.dtype)
    ll = lens_top1(Hl)
    jl = lens_top1(Hl @ Jl.t())
    ll_m = (ll == model_top1).float().mean().item()
    jl_m = (jl == model_top1).float().mean().item()
    print(f"{l:>3} {l / (L - 1):>6.2f} | {ll_m:>16.3f} | {jl_m:>13.3f}")
    del Hl, Jl
    if DEV == "mps":
        torch.mps.empty_cache()
