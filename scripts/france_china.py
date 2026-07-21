"""Reproduce the "France -> China" patching / broadcast result (Nikolenko's post).

Compute per-layer J-vectors for ' France' and ' China', unit-normalize, then patch the residual
stream on layers 9-19 with h <- h + alpha * relu(<h, v_F>) * (v_C - v_F). A SINGLE swap should
redirect many downstream "functions" at once: capital, language, continent, currency. A Germany
control checks selectivity (should ideally hold, especially at 7B).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from jlens import models  # noqa: E402
from jlens.jacobian import compute_jvectors  # noqa: E402
from jlens.steering import Patch, unit_jvectors  # noqa: E402
from jlens.lens import top_tokens  # noqa: E402

# Generic corpus to average the linearization over (NOT France/China-specific).
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

NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEV = os.environ.get("JLENS_DEVICE", "mps")
ALPHAS = [float(a) for a in os.environ.get("JLENS_ALPHAS", "1,2,4").split(",")]
PATCH_LAYERS = list(range(9, 20))

PROBES = [
    ("capital",           "The capital of France is",                  "Paris",  "Beijing"),
    ("language",          "The official language of France is",        "French", "Chinese"),
    ("continent",         "France is located on the continent of",     "Europe", "Asia"),
    ("currency",          "The currency of France is the",             "Euro",   "Yuan"),
    ("CONTROL (Germany)", "The capital of Germany is",                 "Berlin", "Beijing"),
]

print(f"Loading {NAME} on {DEV} ...", flush=True)
lm = models.load(NAME, dtype=torch.float32, device=DEV)
print(f"  family={lm.family} layers={lm.num_layers} d={lm.d}\n", flush=True)


def tid(word):
    ids = lm.tokenizer(" " + word, add_special_tokens=False).input_ids
    return ids[0], len(ids)


france_id, _ = tid("France")
china_id, _ = tid("China")

# per-layer J-vectors for France (row 0) and China (row 1)
S = 10
corpus = torch.cat(
    [lm.tokenizer(t, return_tensors="pt").input_ids[:, :S]
     for t in CORPUS if lm.tokenizer(t, return_tensors="pt").input_ids.shape[1] >= S],
    dim=0,
).to(DEV)
print(f"averaging J-vectors (France, China) over {corpus.shape[0]} fragments, layers {PATCH_LAYERS[0]}-{PATCH_LAYERS[-1]} ...", flush=True)
V, _ = compute_jvectors(lm, [corpus], [france_id, china_id], layers=PATCH_LAYERS)
u_from = unit_jvectors(V, PATCH_LAYERS, 0, DEV, lm.dtype)  # France
u_to = unit_jvectors(V, PATCH_LAYERS, 1, DEV, lm.dtype)    # China
print("  done.\n", flush=True)


def probe(prompt, alpha):
    ids = lm.tokenizer(prompt, return_tensors="pt").input_ids.to(DEV)
    ctx = Patch(lm, PATCH_LAYERS, u_from, u_to, alpha) if alpha > 0 else None
    if ctx:
        ctx.__enter__()
    with torch.no_grad():
        logits = lm.model(input_ids=ids, use_cache=False).logits[0, -1]
    if ctx:
        ctx.__exit__()
    return logits


for name, prompt, expected, swapped in PROBES:
    e_id, _ = tid(expected)
    s_id, _ = tid(swapped)
    print(f"### {name}: {prompt!r}   (expected='{expected}'  swap-target='{swapped}')")
    for a in [0.0] + ALPHAS:
        logits = probe(prompt, a)
        p = torch.softmax(logits.float(), 0)
        top = ", ".join(f"{t.strip()}={pr:.2f}" for t, pr in top_tokens(lm, logits, 3))
        tag = "baseline" if a == 0 else f"alpha={a:g}"
        print(f"   {tag:<10} p({expected})={float(p[e_id]):.3f} p({swapped})={float(p[s_id]):.3f}  | top3: {top}")
    print()
