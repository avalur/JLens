"""Family-parametrized model loading for the J-lens.

Verified constraints baked in (transformers 5.13.0 / torch 2.5.1 / MPS):
- attn_implementation='eager'  (clean grads on MPS; dodges Gemma-2 sdpa+padding NaN)
- dtype=  (NOT torch_dtype=, which is deprecated in transformers 5.x)
- use_cache=False; model.eval(); requires_grad_(False)  (freeze weights, only deltas need grad)
- W_U = get_output_embeddings().weight  (handles tied 1.5B/2B and untied 7B uniformly)
- final norm = model.model.norm  (the model's OWN RMSNorm; inherits gain/(1+w)/eps per family)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_family(config) -> str:
    mt = getattr(config, "model_type", "") or ""
    if mt.startswith("gemma2"):
        return "gemma2"
    if mt.startswith("gemma"):
        return "gemma3"  # gemma3 and future gemma variants
    if mt.startswith("qwen"):
        return "qwen"
    return mt or "unknown"


@dataclass
class LoadedModel:
    name: str
    family: str
    model: torch.nn.Module            # the CausalLM
    tokenizer: object
    device: str
    dtype: torch.dtype

    # --- structural accessors -------------------------------------------------
    @property
    def base(self) -> torch.nn.Module:
        """Base decoder stack (no lm_head). Calling this for the Jacobian avoids
        materializing the huge lm_head logits in the autograd graph."""
        return self.model.model

    @property
    def layers(self):
        return self.model.model.layers

    @property
    def final_norm(self) -> torch.nn.Module:
        return self.model.model.norm

    @property
    def W_U(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight  # [V, d]

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def d(self) -> int:
        return self.model.config.hidden_size

    @property
    def final_logit_softcap(self):
        return getattr(self.model.config, "final_logit_softcapping", None)


def load(
    name: str,
    dtype: torch.dtype = torch.float32,
    device: str = "mps",
    attn: str = "eager",
) -> LoadedModel:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation=attn
    )
    model.config.use_cache = False
    model.to(device)
    model.eval()
    model.requires_grad_(False)  # freeze: only the injected deltas carry grad
    fam = detect_family(model.config)
    return LoadedModel(
        name=name, family=fam, model=model, tokenizer=tok, device=device, dtype=dtype
    )
