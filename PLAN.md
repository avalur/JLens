# Reproducing the "Global Workspace / J-space" (J-lens) on Apple M3 Max

Reproduction plan for the method in Sergey Nikolenko's post
*"Global workspace in the J-space"* (analyzing Anthropic's July-2026 paper
*Verbalizable Representations Form a Global Workspace in Language Models*,
`transformer-circuits.pub/2026/workspace`). We mirror **Nikolenko's own scope** — he reproduced
the method on Qwen2.5-1.5B/7B in "a couple hundred lines of PyTorch" (no public repo). We reimplement
it and add **Gemma-2**.

> This plan is built on a research+verification pass (6 topics, each adversarially re-checked). Facts
> below marked "(verified)" were confirmed against the actually-installed libraries / live HF configs.

---

## 0. What the J-lens is (the one idea to get right)

Residual stream: token `t` at layer `ℓ` is a vector `h_{ℓ,t} ∈ R^d`. The model ends with
`logits = W_U · norm(h_L)` (final RMSNorm then unembed).

- **Logit lens** reads an intermediate state as if it were final: `softmax(W_U · norm(h_ℓ))`.
- **J-lens** first maps `h_ℓ` forward through the *corpus-averaged linearization* of the rest of the
  network, then reads it:

  ```
  J_ℓ  = E_corpus, t'≥t [ ∂h_{L,t'} / ∂h_{ℓ,t} ]        # one d×d matrix per layer
  lens_ℓ(h) = softmax( W_U · norm( J_ℓ · h ) )
  v_y  = row y of ( W_U · J_ℓ )                          # "J-vector" for token y
  ```

- **Read** = decompose `h` over J-vectors; **steer** `h ← h + α·v_y`; **patch** = swap coordinates
  in the J-vector basis (the France↔China experiment).

### The efficient computation (verified correct)
Add a **zero-initialized leaf** `δ_ℓ ∈ R^d` (broadcast over all positions) to each layer `ℓ`'s output.
Define `G_i = Σ_{t'} h_{L,t',i}` on the **raw** (pre-final-norm) last-layer residual. Then for output
coordinate `i`:

```
G_i.backward()   ⇒   δ_ℓ.grad  ==  Σ_t Σ_{t'≥t} ∂h_{L,t',i}/∂h_{ℓ,t}   ==  row i of the pair-summed J_ℓ
```

- One backward per output coord `i` fills row `i` **for every layer at once** → `d` backward passes give
  the full `J_ℓ` for all layers. Divide by the number of causal pairs to average.
- **Causal mask auto-zeros** `t'<t`; no manual masking needed.
- **Orientation trap (verified):** `J_ℓ[i, :] = δ_ℓ.grad` (row index = output/seed coord `i`). Building
  `J[:, i]` transposes it and silently corrupts `J·h`, `W_U·J`, steering, and patching.
- **Zero `δ` grads between the `d` passes** (PyTorch accumulates by default).
- Normalization by pair count is **washed out** of the lens by the final RMSNorm (scale-invariant), but it
  **does** set the magnitude of `v_y`, so it matters for the steering coefficient `α` — fix and document one convention.

### The cheap targeted path (for steering/patching & big models)
For a specific token `y`, `v_y` is a single **vector-Jacobian product**: seed `s = Σ_{t'} (W_U[y] · h_{L,t'})`
(the raw pre-norm logit of `y`, summed over positions), backprop once → `δ_ℓ.grad == v_y` for every layer.
So **k target tokens = k backward passes**, independent of `d`. This is how France→China patching and the
"desires" probes run cheaply even on 7B/9B.

---

## 1. Environment & hardware (detected)

- MacBook Pro (Mac15,9), **Apple M3 Max, 64 GB unified memory**, 12P+4E cores, arm64.
- Python 3.11.9 (pyenv), `torch 2.5.1` (MPS available), `transformers 5.13.0`, `uv` present.

### Setup
```bash
cd ~/PycharmProjects/JLence
uv init --python 3.11
uv add "torch==2.5.1" "transformers==5.13.0" accelerate datasets huggingface_hub \
       numpy matplotlib rich
# Gemma is gated — accept the license on HF, then:
huggingface-cli login          # token with Gemma license accepted
```
Runtime env (safety nets, not perf tools):
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1        # avoid hard errors on any op gap
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # only if OOM on the largest bf16 model
```

---

## 2. Model lineup (decision: "small-first, then scale")

| Phase | Qwen | Gemma | Precision | Full-J cost (rough, soft ±2–3×) |
|-------|------|-------|-----------|-------------------------|
| **A** (validate) | Qwen2.5-1.5B-Instruct (28L, d=1536, tied, vocab 151936) | Gemma-2-2b-it (26L, d=2304, tied, vocab 256000) | **fp32** | ~overnight each |
| **B** (scale) | Qwen2.5-7B-Instruct (28L, d=3584, untied, vocab 152064) | Gemma-2-9b-it (42L, d=3584) | bf16 fwd + **fp32 J accum** | multi-day full-J → prefer targeted VJP path |

Gemma-3 (QK-norm, sliding-window, dual RoPE, multimodal wrapper) is deferred — meaningfully harder and
not needed for a faithful Gemma-2-era reproduction.

---

## 3. Core implementation notes (all verified in-env)

**Loading**
```python
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,   # NOTE: dtype=, not torch_dtype=
                                         attn_implementation="eager") # eager: clean grads, dodges MPS SDPA + Gemma-2 sdpa-NaN
m.config.use_cache = False; m.to("mps"); m.eval()                     # eval() ≠ no_grad(); do NOT wrap in no_grad
m.requires_grad_(False)                                              # freeze weights: only δ leaves need grad
```

**Delta hooks** (transformers 5.x layers return a **bare Tensor**, not a tuple):
```python
d = m.config.hidden_size
deltas = [torch.zeros(d, device="mps", requires_grad=True)           # create ON device → stays a leaf
          for _ in range(m.config.num_hidden_layers)]
def make(i):
    def hook(mod, args, out):
        if isinstance(out, tuple): return (out[0] + deltas[i],) + tuple(out[1:])  # defensive only
        return out + deltas[i]
    return hook
handles = [lyr.register_forward_hook(make(i)) for i, lyr in enumerate(m.model.layers)]
```

**Getting the raw final residual `h_L`**: `hidden_states[-1]` is **post-final-norm** (= `norm(h_L)`),
so capture the *input* to `m.model.norm` (or hook the last decoder layer) for the raw `h_L` used in `G`.
Conveniently, `hidden_states[-1]` **is** exactly `norm(h_L)`, so it's fine as the final-norm term when reading.

**Read-out `norm`** must replicate the model's *own* final RMSNorm (learned gain, eps=1e-6), family-specific:
- **Qwen**: gain = `weight` (no +1); no logit softcap; no embed scaling.
- **Gemma-2**: gain = `(1 + weight)`, float32 upcast then cast; `W_U = get_input_embeddings().weight`
  (tied, **unscaled** — do NOT apply the sqrt(d) input scale on the read side); `final_logit_softcapping=30`
  is **monotonic** → it changes probabilities but **not** top-k ordering, so omit it for rank/top-k readouts,
  apply `30*tanh(logits/30)` only when faithful probabilities are wanted. `attn_softcap=50` is internal-only.

**W_U** = `m.get_output_embeddings().weight` (works for tied 1.5B/2B and untied 7B uniformly). Never hardcode vocab size.

**MPS discipline**: fp32 for 1.5B/2B; keep tensors `.contiguous()`; **no `torch.func.jacrev/vmap` on MPS**
(documented silent all-zero Jacobian risk) — use sequential `torch.autograd.grad`; cross-check one tiny case
on CPU. `torch.mps.empty_cache()` between corpus batches.

---

## 4. Cost & precision strategy

- **Full `d×d` J_ℓ** = `d` backward passes × (corpus / batch). The `d` factor dominates for a small corpus,
  not the corpus size. → Compute full-J only for the **small** models (Phase A), all layers at once, corpus
  ~128×64 tokens (reduce to ~32–64 fragments if you want a faster first pass; J is an average, fewer = noisier).
- **bf16 forward + fp32 J accumulation** for Phase B (never fp32 *weights* for ≥9B — G2-9B fp32 ≈ 37 GB).
- Batch `B=4–8` sequences × `S=64`, one forward graph per batch, `retain_graph=True` across the `d` inner
  backwards, free + `empty_cache()` per batch.
- **Benchmark first:** run the `d`-pass loop on ~10 sequences and extrapolate before committing to an overnight/
  multi-day run. (Published A100 baselines ~30 min/1.5B, ~5 h/7B; M3 Max is ~10–20× slower and estimates are soft.)
- For Phase B experiments that only need specific concepts (France→China, desires), use the **1-VJP-per-token**
  `v_y` path — minutes, not days.

---

## 5. Experiments to reproduce (mapped to the post)

Run each on Qwen2.5-1.5B **and** Gemma-2-2b in Phase A:

1. **Layer profile / three zones.** On ~32 held-out wikitext fragments, measure how often `lens_ℓ` top-1 matches
   the model's final token, per layer. Expect: unreadable first third → slow rise → sharp late convergence.
   (Nikolenko notes logit-lens can beat J-lens in the *middle* on 1.5B; J-lens should pull ahead by 7B.)
2. **Eiffel two-hop.** Prompt: *"The capital of the country where the Eiffel Tower is located is the city of"* → "Paris".
   Show J-lens surfacing **Paris/France** at layers ~7–15 while the **logit lens sees nothing** until ~layer 22;
   note the mid-layer blank/"fill-in-the-blank" token habit and (for the RU prompt) the "thinks in English/Chinese first"
   effect (Wendler et al. 2024).
3. **France→China patching (the broadcast test).** Normalized J-vectors `v̂_F`, `v̂_C` at layers ~9–19; per position:
   `h ← h + α·⟨h, v̂_F⟩₊·(v̂_C − v̂_F)`. Check one swap redirects capital/language/continent/currency together;
   verify a control ("capital of Germany") and sweep `α` (drift at α≈2 on 1.5B; tighter/cleaner at 7B).
4. **"Desires" introspection.** Ask what it wants / fears / who it is; read `lens_ℓ` where the answer forms; compare the
   **spoken** answer vs the **internal** J-space content. (Anecdotal — report as distributions, no strong claims.)
5. *(Stretch)* **Ablation / selectivity**: zero projections onto the top J-vectors and show multi-step reasoning
   degrades while grammar/recall survive.

---

## 6. Validation (before trusting any result)

- **Logit-lens identity:** substituting `J_ℓ = I` must reproduce the standard logit lens exactly.
- **Final-layer sanity:** `J_L` should be ≈ per-position identity; `lens_L` must reproduce the model's own
  next-token logits to fp32 tolerance.
- **Family read-out check:** assert `len(hidden_states) == num_layers+1` and `lm_head(hidden_states[-1]) == out.logits`
  at load (catches transformers-version indexing surprises).
- **MPS vs CPU:** diff `J_ℓ` (or one `v_y`) between MPS and CPU on a tiny case to rule out silent MPS autograd bugs.

---

## 7. Proposed repo layout

```
JLence/
  pyproject.toml
  PLAN.md                    # this file
  jlens/
    models.py                # family-parametrized load (eager, dtype, freeze, W_U, final-norm module)
    hooks.py                 # δ-injection hooks + raw h_L capture
    jacobian.py              # full averaged J_ℓ (d VJPs) + per-token v_y (1 VJP)
    lens.py                  # softmax(W_U · norm(J·h)); Qwen vs Gemma-2 norm/softcap branches
    corpus.py                # 128×64-token wikitext fragments
    validate.py              # logit-lens==(J=I), J_L≈I, MPS-vs-CPU
    experiments/{layer_profile,eiffel_twohop,france_china_patch,desires}.py
  scripts/{compute_jacobian,run_experiment}.py
```

## 8. Milestones

- **[DONE] M0** env + `models.py` loads Qwen2.5-1.5B on MPS (fp32, eager). Gemma-2-2b already in HF cache.
- **[DONE] M1** J-lens core + `validate.py` all-green on 1.5B: structural maxdiff 0.0, jlens-identity 0.0,
  `J_{L-1}` diag = 2/(S+1) exactly, finite-difference rel-err 1.8% (also rules out the MPS silent-zero bug).
- **[DONE] Eiffel two-hop** (`scripts/eiffel_twohop.py`) reproduced on 1.5B via a finite-difference JVP
  (cheap, faithful, full-vocab): J-lens ranks France #1 / Paris #2–4 in layers 6–13; logit lens is junk
  (Paris rank 173–7962) until ~layer 22. Bonus: Venice/Rome/Italy cluster, Chinese 巴黎, `____` exam tokens.
  KEY TECHNIQUE: `J_ℓ · h_probe` = corpus-averaged JVP via central finite differences → no d×d matrix needed
  for probe-specific readouts. (The `v_y`-over-candidates shortcut was tried first and is NOT faithful —
  restricted-candidate softmax hides the full-vocab rank story.)
- **M2** full averaged `J_ℓ` on Qwen2.5-1.5B (benchmark → run) — needed only for unrestricted top-k / layer
  profile over the whole vocab; probe-specific experiments use the FD-JVP path instead.
- **M3** France→China patching + layer profile on 1.5B; then cross-check on Gemma-2-2b.
- **[DONE] M4a** scaled to **Qwen2.5-7B-Instruct** (fp32, ~30 GB, fits 64 GB; untied path validated,
  structural maxdiff 0.0). Eiffel two-hop reproduced and SHARPER than 1.5B: at 7B the logit lens is pure
  noise (Paris rank 50k–120k) until layer ~22, while the J-lens reaches Paris rank 11–61 by layers 8–10
  (+`city/City/cities` cluster at L10). Confirms the post's "J-lens cleanly beats logit lens at 7B" claim.
  7B download ~5 min; fp32 load ~cached-fast; Eiffel FD-JVP run a few min.
- **[DONE] France→China patching** (`jlens/steering.py`, `scripts/france_china.py`) on 1.5B AND 7B.
  Single J-vector swap on layers 9-19 broadcasts across capital/language/continent/currency:
  1.5B Europe 0.70→Asia 0.70, French→Chinese(0.39), Euro→Renminbi, Berlin control 0.48→0.008 (BREAKS,
  matches post). 7B Europe 0.88→Asia 0.90, and the Germany control HOLDS (Beijing≈0.005, matches the
  post's "p(Beijing)≈0.002") — reproduces the scale-dependent selectivity claim. Caveat: bare-completion
  prompts let the `__` exam token grab top-1 more than the post's tables; sharpen with a bigger J-vector
  corpus (post used 128x64) / chat template if prettier top-1 wanted.
- **[DONE] Gemma-2-2b cross-check** — validation ALL-GREEN on the gemma2 family (structural maxdiff 0.0,
  i.e. readout bit-exact INCLUDING the 30·tanh soft-cap + (1+weight) RMSNorm + tied √d-scaled embeds;
  FD rel-err 0.85%). Eiffel two-hop reproduces cross-family: J-lens finds Paris rank ~89 / France ~43 by
  layer 7-8 while the logit lens is noise (rank 5k-192k) until ~layer 18. Gemma texture: a city/metropolis
  abstraction cluster mid-network, base-model code/web token directions (gameserver/UserScript) in L10-17,
  no Chinese 巴黎. Proves the code is genuinely architecture-general, not Qwen-specific.
- **[DONE] Layer profile / 3 zones** (`scripts/layer_profile.py`, needs the full d×d Jacobian = M2 step,
  ~89s on 1.5B for a small averaging batch). On Qwen2.5-1.5B the three zones are textbook: sensory
  (depth 0–0.37, match ~0), workspace (slow rise 0.04→0.40), motor (sharp 0.56→1.00 at the last layers;
  layer 27 = 1.000 for both lenses, an end-to-end sanity check). On 1.5B the logit lens ties/slightly beats
  the J-lens mid-network (post: "logit lens beats J-lens in middle on small models; J-lens predicts averaged
  future influence, not next token"). On Qwen2.5-7B (full J = 761s) the crossover FLIPS: J-lens beats logit
  lens throughout mid-late layers (L19: J 0.107 vs logit 0.027 — matches the post's "0.10 vs 0.03 at layer 20"
  almost exactly), with logit lens briefly retaking the last motor layer (L26). Reproduces the scale-dependent
  claim precisely. Three zones intact on both.
- **[DONE] Desires / introspection** (`scripts/desires.py`, Qwen2.5-*-Instruct, chat template, forced
  one-word answer; internal read via FD-JVP at the answer position). 1.5B matches the post item-for-item:
  says AI/Good/deflects while J-space holds robot(0.60), Busy(0.53), Happiness(0.39), Unknown(0.68)/Fear.
  7B: two EXACT hits — "Who are you" AI aloud but assistant=0.95 internal (post: 0.95); "afraid" Darkness
  aloud but 未知=0.36 (post: 0.38); spoken Knowledge/Darkness/AI all match. MISS: "Peace" not reproduced,
  and 7B mid-workspace layers are junk-dominated (code/punct tokens) because the averaging corpus is tiny
  (24x10=240 tokens vs paper's 128x64≈8200); clean concepts only at the deep near-motor layer. A bigger
  averaging corpus would likely clean the mid-layers. NOTE: the shallow-workspace 'Placeholder'/EOS/junk
  domination is corpus-size limited, not a code bug (validation is bit-exact).
- **[DONE] Corpus sharpening** — added `jlens/corpus.py: wiki_texts()/stream_batches()` reading the
  locally-cached English Wikipedia (Arrow via pyarrow, no `datasets` install / no network) into paper-scale
  N×64-token batches; desires now micro-batches the FD-JVP over it (96×64 = 199,680 pairs vs old 952).
  Result: 1.5B got cleaner+richer (Placeholder artifact gone; Identity/Self surfaced; Busy≈0.68, robot 0.63);
  7B mid-layer code-junk (`)));`/`.appspot`) collapsed from confident 0.05-0.10 to flat ~0.01, deep concepts
  held/improved (fear 未知 0.36→0.72; assistant 0.95→0.82; knowledge 0.70). "Peace" still NOT reproduced and
  7B mid-workspace stays uninformative on these probes — consistent with the post's own "anecdote / phrasing-
  sensitive" caveat (Nikolenko's exact prompt wording isn't published). Reproduced in-kind, not chased further.
- **[ATTEMPTED — did NOT reproduce] Ablation / selectivity** (`scripts/ablation.py`, `jlens/steering.py:ProjectOut`).
  Tried two J-space definitions (top-k SVD of J_l; top-k PCA of concept J-vectors v_y=W_U[y]@J_l) projected out
  of the residual at workspace layers, vs a magnitude-matched random-subspace control. Neither reproduced the
  post's "ablation collapses multi-step reasoning, spares automatic abilities." Instead, both proxies REMOVED the
  fill-in-the-blank/persona distractor directions → concrete answers got STRONGER (two-hop Paris 0.26→0.83),
  while the RANDOM control degraded everything. Diagnosis: (1) static global subspace ≠ the paper's per-input
  ACTIVE J-space (gradient-pursuit over an overcomplete J-vector dictionary); (2) answer-position projection ≠
  disrupting intermediate write→read ROUTING; (3) single-token-prob metric ≠ multi-token generation quality
  (their tasks: analogies/cipher/translation/sonnets); (4) 1.5B too small (Nikolenko's own caveat). This is the
  one headline that needs the paper's fuller machinery to reproduce. Cleaner alternative selectivity demo =
  Spanish→French PATCHING (language-ID breaks, continuation survives) via the validated Patch tool.
- **[FAITHFUL BUILD — 1.5B negative, 7B pending] Active J-space ablation** (`scripts/ablation_active.py`).
  Per-input ACTIVE J-space: at each position+workspace-layer, top-k J-vectors by correlation with the
  dictionary V_l=W_U@J_l, then EXACT orthogonal-projection removal; applied throughout multi-token GENERATION;
  graded on whether the answer appears in the greedy output; matched RANDOM-J-vector control.
  1.5B result (swept k=4..25, various bands, both sum-of-proj and ortho-proj): NO selective window.
  - sum-of-projections over-subtracts → non-selective repetition collapse (0/5 both groups).
  - ortho-projection → gentle + non-selective: reasoning≈automatic preserved, and J-space ablation is
    LESS destructive than the equal-dim random control (opposite of the paper). On this small model the
    top-k active J-subspace is not a reasoning bottleneck (later layers reconstruct removed content).
  7B result (k=12, 25, ortho-proj): ALSO does NOT reproduce. Outputs stay fully coherent; J-space ablation
  does NOT collapse reasoning (reasoning 4/5→5/5, because removing J-space strips the MCQ '____' format habit
  → direct answers), automatic drops only mildly (5/5→3-4/5) and comparably to the random control.
  FINAL VERDICT on property-5 ablation-selectivity: NOT reproduced on Qwen2.5-1.5B or 7B, even with a faithful
  per-input active J-space + exact orthogonal removal + generation eval + matched control (swept k/band).
  Diagnosis: needs (a) the paper's per-input gradient-pursuit reconstruction over the learned overcomplete
  dictionary, and (b) far larger scale (Sonnet/Haiku/Opus) where the workspace is a removable bottleneck;
  at 1.5B-7B the models reconstruct removed content in later layers, so reasoning survives. This is the one
  headline of the five that did NOT reproduce — reported straight, not tuned to look positive.
- **[DONE — reframed] Spanish→French language patching** (`scripts/language_selectivity.py`, reuses Patch +
  cached J). Swept α on 1.5B and 7B. The "what language is this?" answer DOES flip Spanish→French (top-1:
  Spanish→French at α≥2-3). BUT the intended SELECTIVITY (continuation unaffected) does NOT reproduce at
  either scale: the same patch also switches the CONTINUATION to fluent French (1.5B: "El ciel estival était
  doux"; 7B: "et la beauté naturelle. Elle portait une robe légère") — no α where A flips while B stays
  Spanish (B top-1 agreement ≤50% wherever A flips). Reframed positive result: a single language J-vector
  patch is a COHERENT GLOBAL language steer (fluently rewrites the output language), reinforcing J-vector
  causal potency — but it is GLOBAL, not selective. Consistent with the ablation negative: property-5
  selectivity is a larger-model phenomenon; at <=7B, J-space interventions act globally (concepts used
  pervasively). Net on property 5: NOT reproduced at 1.5B/7B via either ablation or language-patching.
- **[DONE] M5 README.md** — full write-up (idea, env, layout, six-finding scorecard with numbers, run
  commands, key techniques, honest caveats). Adversarially verified by a 4-agent workflow (paths / run
  commands / numbers-vs-logs / clarity); all flagged issues fixed: corrected the Jacobian-cache reuse list
  (layer_profile recomputes in memory; ablation.py added), softened the "bit-exact validation" claim (3 exact
  checks + ~1-2% FD), attributed the two-corpus desires numbers, scoped the 1.5B ablation degradation, and
  toned "Beijing ≈ post's ≈0.002" to "same order". Regenerated fresh passing validation logs for 1.5B + Gemma
  (/tmp/jlens_validate_15b.log, /tmp/jlens_validate_gemma.log) so "all three families green" is artifact-backed
  (FD rel-err confirmed 0.85% Gemma / 1.76% 1.5B / 2.09% 7B); removed the stale crashed /tmp/jlens_validate.log.

## STATUS: reproduction complete. 5 of 6 headline findings reproduced (lens/two-hop, France→China broadcast,
## layer-profile/zones, desires) across Qwen2.5-1.5B/7B + Gemma-2-2b; selectivity (property 5) is a documented,
## diagnosed negative at <=7B (needs the paper's gradient-pursuit J-space + larger scale). README.md is the entry point.

## 9. Open questions / risks

- Wall-clock is soft (±2–3×, likely slower than the A100-anchored estimates); benchmark before long runs.
- Exact paper conventions not pinned: `h_ℓ` = layer *output* vs *input* (off-by-one layer index); pair-count
  normalization constant; whether `v_y`/lens include the final-norm Jacobian. Cross-check vs the paper when finalizing.
- Digit-splitting tokenizers (both Qwen & Gemma) mean number/arithmetic concepts have no single token — expect the
  arithmetic experiments to fail as in the post; that's a faithful reproduction, not a bug.
- Gemma-2 sliding-window (even layers) gives layer-type-dependent receptive fields; keep in mind when averaging.
```
