"""J-lens: averaged-Jacobian lens ("global workspace / J-space") reproduction.

Core pieces:
- models.load(name)            -> LoadedModel (eager attn, fp32, frozen weights, MPS)
- hooks.JHooks(lm)             -> per-layer delta leaves + in-graph raw h_L capture
- jacobian.compute_jacobian    -> corpus-averaged d x d J_l via reverse-mode VJPs
- jacobian.compute_jvectors    -> per-token v_y via 1 VJP each (cheap, scales to big models)
- lens.logit_lens / j_lens     -> readout softmax(W_U . norm([J] h)), family-aware
"""

from . import models, hooks, jacobian, lens  # noqa: F401
