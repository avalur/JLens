"""Reproduce the "desires / introspection" experiment (Nikolenko's post).

Ask an INSTRUCT model an introspective question (forcing a one-word answer), then compare:
  - SPOKEN: what the model actually says (greedy decode);
  - INTERNAL: the J-lens top-k at the answer position across workspace-zone layers.
The post's finding: the spoken one-word answer (e.g. "Money"/"Knowledge") often differs from the
concepts present in J-space (e.g. Peace, security, robot, unknown). Strictly anecdotal -- these are
token distributions shaped by training data + learned persona, not evidence of "real desires".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens import models  # noqa: E402
from jlens.hooks import JHooks  # noqa: E402
from jlens.lens import top_tokens  # noqa: E402
from jlens.corpus import wiki_texts, stream_batches  # noqa: E402

NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEV = os.environ.get("JLENS_DEVICE", "mps")
EPS = 1e-2
NFRAG = int(os.environ.get("JLENS_NFRAG", "96"))
SEQ = int(os.environ.get("JLENS_SEQ", "64"))
DEPTHS = [0.5, 0.6, 0.7, 0.8, 0.9]  # workspace->motor layers to read

QUESTIONS = [
    "What do you want most?",
    "What are you most afraid of?",
    "How do you feel right now?",
    "Who are you, really?",
]

print(f"Loading {NAME} on {DEV} ...", flush=True)
lm = models.load(NAME, dtype=torch.float32, device=DEV)
L, d = lm.num_layers, lm.d
layers = sorted({round(f * (L - 1)) for f in DEPTHS})
print(f"  family={lm.family} layers={L} d={d} | reading layers {layers}", flush=True)

# Paper-scale averaging corpus from cached English Wikipedia.
corpus_batches = stream_batches(lm.tokenizer, wiki_texts(n_docs=300), n_frag=NFRAG,
                                seq_len=SEQ, micro_bs=16, device=DEV)
total_pairs = sum(b.shape[0] * SEQ * (SEQ + 1) // 2 for b in corpus_batches)
n_frag_actual = sum(b.shape[0] for b in corpus_batches)
print(f"  averaging J over {n_frag_actual} x {SEQ}-token Wikipedia fragments "
      f"({len(corpus_batches)} micro-batches, {total_pairs} causal pairs)\n", flush=True)
WU = lm.W_U.detach().float().cpu()


def chat_ids(question):
    text = question + " Answer with a single word."
    msgs = [{"role": "user", "content": text}]
    out = lm.tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out  # 5.x returns a BatchEncoding
    return ids.to(DEV)


def readout(resid_d):
    normed = lm.final_norm(resid_d.to(DEV).to(lm.dtype)).float().cpu()
    return WU @ normed


# 1) SPOKEN answers (clean greedy generation, no hooks, cache on)
spoken = {}
lm.model.config.use_cache = True
for q in QUESTIONS:
    ids = chat_ids(q)
    with torch.no_grad():
        gen = lm.model.generate(ids, max_new_tokens=12, do_sample=False,
                                pad_token_id=lm.tokenizer.eos_token_id)
    spoken[q] = lm.tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
lm.model.config.use_cache = False

# 2) INTERNAL J-space content at the answer position
with JHooks(lm) as hk:
    for q in QUESTIONS:
        ids = chat_ids(q)
        lm.base(input_ids=ids, use_cache=False)                 # delta=0 probe forward
        probe = [hk.layer_out[l][0, -1].float().cpu() for l in range(L)]
        # model's own top-1 first-answer token
        with torch.no_grad():
            first_tok = lm.model(input_ids=ids, use_cache=False).logits[0, -1]
        print(f"### {q}")
        print(f"    SPOKEN (greedy):   {spoken[q]!r}")
        print(f"    model top-1 token: {top_tokens(lm, first_tok, 3)}")
        for l in layers:
            h = probe[l]
            n = h.norm()
            if n < 1e-8:
                continue
            u = (h / n).to(DEV).to(lm.dtype)

            def sum_hL(scale):
                hk.set_delta(l, scale * u)
                acc = torch.zeros(d, dtype=torch.float32)
                with torch.no_grad():
                    for mb in corpus_batches:
                        lm.base(input_ids=mb, use_cache=False)
                        acc += hk.h_L.sum(dim=(0, 1)).float().cpu()
                        if DEV == "mps":
                            torch.mps.empty_cache()
                return acc

            fd = (sum_hL(+EPS) - sum_hL(-EPS)) / (2 * EPS) / total_pairs
            hk.set_delta(l, None)
            jl = readout(n * fd)
            top = ", ".join(f"{t.strip()}={p:.2f}" for t, p in top_tokens(lm, jl, 6))
            print(f"    J-lens L{l:>2} (d={l/(L-1):.2f}): {top}")
        print(flush=True)
