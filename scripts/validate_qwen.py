"""M1 validation: prove the J-lens core is correct on Qwen2.5-1.5B before scaling / Gemma.

Usage:
    JLENS_MODEL=Qwen/Qwen2.5-1.5B-Instruct JLENS_DEVICE=mps python3 scripts/validate_qwen.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens import models, validate  # noqa: E402
from jlens.corpus import fixed_batches  # noqa: E402

NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEV = os.environ.get("JLENS_DEVICE", "mps")

print(f"Loading {NAME} on {DEV} (fp32, eager) ...", flush=True)
t0 = time.time()
lm = models.load(NAME, dtype=torch.float32, device=DEV)
print(
    f"  loaded in {time.time() - t0:.1f}s | family={lm.family} "
    f"layers={lm.num_layers} d={lm.d} vocab={lm.W_U.shape[0]} "
    f"tie={lm.model.config.tie_word_embeddings}",
    flush=True,
)

batches = fixed_batches(lm.tokenizer, seq_len=12, device=DEV)
if not batches:  # fallback: full tokenization of the first sentence
    from jlens.corpus import DEFAULT_TEXTS

    ids = lm.tokenizer(DEFAULT_TEXTS[0], return_tensors="pt").input_ids.to(DEV)
else:
    ids = batches[0]
print(f"  validation seq: {lm.tokenizer.decode(ids[0])!r} (S={ids.shape[1]})\n", flush=True)

t1 = time.time()
all_ok, _ = validate.run_all(lm, ids, verbose=True)
print(f"\nvalidation took {time.time() - t1:.1f}s")
print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
