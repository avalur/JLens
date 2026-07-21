"""Reproduce the "selectivity / ablation" result (Nikolenko's post, property 5).

Ablate J-space (zero the residual's projection onto the J-space directions) at the workspace-zone
layers and show it collapses MULTI-STEP REASONING while barely touching AUTOMATIC abilities
(single-hop recall, pattern completion). A RANDOM subspace of equal dimension is the control:
if reasoning breaks under J-space ablation but NOT under the random control, the effect is
specific to J-space, not to perturbation magnitude.

J-space directions at layer l := top-k right singular vectors of the corpus-averaged Jacobian J_l
(the residual-stream directions that most drive downstream verbalizable computation).

The full d x d Jacobian is cached to /tmp so k / layer-band can be tuned without recomputing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens import models  # noqa: E402
from jlens.jacobian import compute_jacobian  # noqa: E402
from jlens.steering import ProjectOut  # noqa: E402
from jlens.corpus import wiki_texts, stream_batches  # noqa: E402

NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEV = os.environ.get("JLENS_DEVICE", "mps")
K = int(os.environ.get("JLENS_K", "40"))                 # J-space dims to ablate per layer
BAND = (float(os.environ.get("JLENS_D0", "0.35")),
        float(os.environ.get("JLENS_D1", "0.80")))        # workspace depth band to ablate
JSPACE = os.environ.get("JLENS_JSPACE", "svd")           # "svd" (top sing. vecs of J_l) | "concept"
# J-averaging corpus size. Default 120/16/32 (=512 tokens) suits small models; large models retain the
# forward graph across d backward passes, so shrink this for a 9b (e.g. 8/16 = 128 tokens) to fit memory.
J_DOCS = int(os.environ.get("JLENS_JDOCS", "120"))
J_NFRAG = int(os.environ.get("JLENS_JNFRAG", "16"))
J_SEQ = int(os.environ.get("JLENS_JSEQ", "32"))

# Concept dictionary for the "concept" J-space: dominant directions of the J-vectors of these words.
CONCEPTS = (
    "time year people way day man thing woman life child world school state family student group "
    "country problem hand part place case week company system program question work government number "
    "night point home water room mother area money story fact month lot right study book eye job word "
    "business issue side kind head house service friend father power hour game line end member law car "
    "city community name president team minute idea body information back parent face others level office "
    "door health person art war history party result change morning reason research girl guy moment air "
    "teacher force education foot boy age policy music market sense nation plan college interest death "
    "experience effect use class control care field development role effort rate heart drug show leader "
    "light voice wife police mind price report decision son view relationship town road arm difference "
    "value building action model season society tax director position player record paper space ground"
).split()

REASONING = [  # multi-step: should collapse under J-space ablation
    ("two-hop capital", "The capital of the country where the Eiffel Tower is located is the city of", "Paris"),
    ("analogy", "Paris is to France as Tokyo is to", "Japan"),
    ("transitive", "Anna is older than Bob. Bob is older than Carl. So the oldest person is", "Anna"),
    ("two-hop language", "The Eiffel Tower is in a country whose main language is", "French"),
]
AUTOMATIC = [  # single-step / pattern: should survive
    ("single-hop fact", "The capital of France is the city of", "Paris"),
    ("antonym recall", "The opposite of the word hot is", "cold"),
    ("induction/copy", "cat dog cat dog cat", "dog"),
    ("continuation", "Once upon a", "time"),
]

print(f"Loading {NAME} on {DEV} ...", flush=True)
lm = models.load(NAME, dtype=torch.float32, device=DEV)
L, d = lm.num_layers, lm.d
ablate_layers = [l for l in range(L) if BAND[0] <= l / (L - 1) <= BAND[1]]
print(f"  family={lm.family} layers={L} d={d} | ablate layers {ablate_layers[0]}-{ablate_layers[-1]} "
      f"(depth {BAND[0]}-{BAND[1]}), k={K}\n", flush=True)

# --- full averaged Jacobian (cached) --------------------------------------------------------
cache = f"/tmp/jlens_J_{NAME.split('/')[-1]}.pt"
if os.path.exists(cache):
    print(f"loading cached Jacobian {cache}", flush=True)
    J = torch.load(cache)
else:
    avg = torch.cat(stream_batches(lm.tokenizer, wiki_texts(J_DOCS), n_frag=J_NFRAG, seq_len=J_SEQ,
                                   micro_bs=16, device=DEV), dim=0)  # one [J_NFRAG, J_SEQ] batch
    print(f"computing full Jacobian over {avg.shape} ({d} backward passes) ...", flush=True)
    J, _, _ = compute_jacobian(lm, [avg], rows=None, layers=None, normalize=True, verbose=True)
    torch.save(J, cache)
    print(f"  cached -> {cache}", flush=True)

# --- J-space basis + random control per layer -----------------------------------------------
# "svd":     top-k right singular vectors of J_l (directions that most drive downstream state).
# "concept": top-k principal directions of the concept J-vectors v_y = W_U[y] @ J_l (the
#            residual directions that most vary "what content token the state pushes toward").
g = torch.Generator().manual_seed(0)
jspace, control = {}, {}
WU_cpu = lm.W_U.detach().float().cpu()
if JSPACE == "concept":
    cids = list(dict.fromkeys(
        lm.tokenizer(" " + w, add_special_tokens=False).input_ids[0] for w in CONCEPTS))
    WU_c = WU_cpu[cids]  # [C, d]
    print(f"  concept J-space from {len(cids)} content tokens", flush=True)
for l in ablate_layers:
    if JSPACE == "concept":
        Vmat = WU_c @ J[l].float()                       # [C, d] concept J-vectors
        Vmat = Vmat - Vmat.mean(0, keepdim=True)         # center -> principal directions
        Vh = torch.linalg.svd(Vmat, full_matrices=False).Vh
    else:
        Vh = torch.linalg.svd(J[l].float(), full_matrices=False).Vh  # rows = input dirs
    jspace[l] = Vh[:K].to(DEV).to(lm.dtype)
    Q, _ = torch.linalg.qr(torch.randn(d, K, generator=g))           # random orthonormal control
    control[l] = Q.t().to(DEV).to(lm.dtype)


def correct_prob(prompt, word, ctx=None):
    ids = lm.tokenizer(prompt, return_tensors="pt").input_ids.to(DEV)
    tok = lm.tokenizer(" " + word, add_special_tokens=False).input_ids[0]
    if ctx:
        ctx.__enter__()
    with torch.no_grad():
        logits = lm.model(input_ids=ids, use_cache=False).logits[0, -1]
    if ctx:
        ctx.__exit__()
    p = torch.softmax(logits.float(), 0)
    top = lm.tokenizer.decode([int(logits.argmax())]).strip()
    return float(p[tok]), top


def run(group, name):
    print(f"=== {name} ===")
    print(f"{'task':<20} {'answer':<8} | {'clean':>7} {'ablated':>8} {'control':>8} | top-1 clean/abl")
    drops_abl, drops_ctl = [], []
    for tname, prompt, word in group:
        pc, tc = correct_prob(prompt, word)
        pa, ta = correct_prob(prompt, word, ProjectOut(lm, jspace))
        pk, _ = correct_prob(prompt, word, ProjectOut(lm, control))
        drops_abl.append((pc - pa) / (pc + 1e-9))
        drops_ctl.append((pc - pk) / (pc + 1e-9))
        print(f"{tname:<20} {word:<8} | {pc:>7.3f} {pa:>8.3f} {pk:>8.3f} | {tc!r}/{ta!r}")
    print(f"  mean relative drop:  J-space={sum(drops_abl)/len(drops_abl):+.2%}   "
          f"random-control={sum(drops_ctl)/len(drops_ctl):+.2%}\n")


run(REASONING, "MULTI-STEP REASONING (expect J-space ablation to collapse it)")
run(AUTOMATIC, "AUTOMATIC ABILITIES (expect both to survive)")
