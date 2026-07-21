"""Reproduce the "Eiffel two-hop" result (Nikolenko's post, Qwen section) -- FAITHFUL version.

Prompt: "...the capital of the country where the Eiffel Tower is located is the city of" -> Paris.
Claim: over the FULL vocabulary, the LOGIT lens does not rank Paris near the top until the very
late layers, while the J-LENS surfaces Paris (and France) in the MIDDLE layers -- a latent
"France -> Paris" two-hop computed before it is verbalized.

Faithful + cheap method: for the probe's residual h_l at each layer, compute the corpus-averaged
J_l @ h_l directly as a Jacobian-vector product via central finite differences (perturb a
broadcast delta_l = eps * unit(h_l), measure the change in sum_{t'} h_L over a generic corpus).
This equals (full J_l) @ h_l without ever forming the d x d matrix. Then read the FULL vocab:
  J-lens logits = W_U . norm(J_l @ h_l)   vs   logit-lens logits = W_U . norm(h_l).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens import models  # noqa: E402
from jlens.hooks import JHooks  # noqa: E402
from jlens.lens import top_tokens  # noqa: E402

NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEV = os.environ.get("JLENS_DEVICE", "mps")
# Central-difference step for the FD-JVP. 1e-2 suits small models; large models (bigger h_L
# magnitudes) need a larger step to avoid fp32 cancellation -- pass JLENS_EPS=0.1 for Gemma-2-9b.
EPS = float(os.environ.get("JLENS_EPS", "1e-2"))

PROMPT = "The capital of the country where the Eiffel Tower is located is the city of"

# Generic corpus to average the linearization over (NOT France-specific).
CORPUS = [
    "The history of ancient civilizations spans many thousands of years of change.",
    "Modern computers rely on transistors etched onto small silicon chips.",
    "Many species of birds migrate south during the cold winter months.",
    "The novel describes a long journey across a vast and empty desert.",
    "Chemical reactions can release or absorb energy in the form of heat.",
    "Economic policy often balances inflation against the level of employment.",
    "The orchestra tuned their instruments before the evening performance began.",
    "Rivers carry sediment downstream and slowly reshape the surrounding land.",
    "Scientists collected data from the telescope over several clear nights.",
    "The committee debated the new proposal for nearly three long hours.",
    "Bacteria can reproduce rapidly under warm and moist laboratory conditions.",
    "The bridge was engineered to withstand strong winds and heavy traffic.",
    "Students practiced the difficult passage until the melody sounded smooth.",
    "The market opened higher after the company reported strong quarterly sales.",
    "Volcanic eruptions can send ash high into the upper atmosphere.",
    "The museum acquired several paintings from a private collection last year.",
    "Farmers rotate their crops to keep the soil fertile and productive.",
    "The algorithm sorts the list by repeatedly comparing adjacent elements.",
    "A balanced diet includes proteins, carbohydrates, fats, and many vitamins.",
    "The treaty was signed after months of careful and difficult negotiation.",
    "Light from distant stars takes many years to reach our small planet.",
    "The factory installed new robots to assemble the delicate components.",
    "Children learn language by listening and imitating the people around them.",
    "The mountain trail became steep and rocky near the windy summit.",
]

print(f"Loading {NAME} on {DEV} ...", flush=True)
lm = models.load(NAME, dtype=torch.float32, device=DEV)
L, d = lm.num_layers, lm.d
print(f"  family={lm.family} layers={L} d={d}\n", flush=True)

paris_id = lm.tokenizer(" Paris", add_special_tokens=False).input_ids[0]
france_id = lm.tokenizer(" France", add_special_tokens=False).input_ids[0]

# Build a single corpus batch [N, S] (fixed length, no padding).
S = 10
corpus_ids = torch.cat(
    [lm.tokenizer(t, return_tensors="pt").input_ids[:, :S]
     for t in CORPUS if lm.tokenizer(t, return_tensors="pt").input_ids.shape[1] >= S],
    dim=0,
).to(DEV)
N = corpus_ids.shape[0]
total_pairs = N * S * (S + 1) // 2
print(f"corpus for averaging: {N} fragments x {S} tokens ({total_pairs} causal pairs)\n", flush=True)

WU = lm.W_U.detach().float().cpu()  # [V, d]


def rank_and_prob(logits_cpu, tok_id):
    rank = int((logits_cpu > logits_cpu[tok_id]).sum().item())  # 0 = top
    prob = float(torch.softmax(logits_cpu, 0)[tok_id])
    return rank, prob


def readout(resid_d):  # resid_d: [d] cpu -> full-vocab logits [V] cpu
    normed = lm.final_norm(resid_d.to(DEV).to(lm.dtype)).float().cpu()
    return WU @ normed


# 1) probe: capture per-layer residual (all positions) + model's own top-5
with JHooks(lm) as hk:
    ids = lm.tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEV)
    lm.base(input_ids=ids, use_cache=False)
    probe_layers = [hk.layer_out[l][0].float().cpu() for l in range(L)]  # [S_p, d] each
with torch.no_grad():
    model_logits = lm.model(input_ids=ids, use_cache=False).logits[0, -1]
print(f"probe: {PROMPT!r}")
print(f"  model top-5: {top_tokens(lm, model_logits, 5)}\n", flush=True)

# 2) per-layer faithful J-lens via finite-difference JVP, full-vocab readout
print(f"{'lyr':>3} | {'LOGIT-lens top-3':<34} {'LLpar#':>6} | "
      f"{'J-lens top-3':<34} {'JLpar#':>6} {'JLfr#':>6}")
print("-" * 108)

with JHooks(lm) as hk:
    for l in range(L):
        h_probe = probe_layers[l][-1]  # [d] last position
        n = h_probe.norm()
        if n < 1e-8:
            continue
        u = (h_probe / n).to(DEV).to(lm.dtype)

        def sum_hL(scale):
            hk.set_delta(l, scale * u)
            with torch.no_grad():
                lm.base(input_ids=corpus_ids, use_cache=False)
                return hk.h_L.sum(dim=(0, 1)).float().cpu()

        hp, hm = sum_hL(+EPS), sum_hL(-EPS)
        hk.set_delta(l, None)
        Jl_u = (hp - hm) / (2 * EPS) / total_pairs  # (avg J_l) @ unit
        Jl_h = n * Jl_u                              # (avg J_l) @ h_probe

        ll = readout(h_probe)   # logit lens (full vocab)
        jl = readout(Jl_h)      # J-lens (full vocab)
        ll_rank, _ = rank_and_prob(ll, paris_id)
        jl_rank, _ = rank_and_prob(jl, paris_id)
        jl_fr_rank, _ = rank_and_prob(jl, france_id)

        ll_top = ", ".join(t.strip() for t, _ in top_tokens(lm, ll, 3))
        jl_top = ", ".join(t.strip() for t, _ in top_tokens(lm, jl, 3))
        print(f"{l:>3} | {ll_top:<34} {ll_rank:>6} | {jl_top:<34} {jl_rank:>6} {jl_fr_rank:>6}")

print("\n(#: rank of Paris/France over the full ~152k vocab; 0 = top. "
      "Effect = J-lens ranks Paris/France far higher than the logit lens in the middle layers.)")
