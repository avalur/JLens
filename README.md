# JLence — reproducing the "Global Workspace / J-space" interpretability results on a Mac

A from-scratch reproduction of the **J-lens** (averaged-Jacobian lens) method and its experiments,
run locally on an **Apple M3 Max (64 GB)** across **Qwen2.5-1.5B/7B** and **Gemma-2-2b**.

Source material:
- Anthropic (2026), *"Verbalizable Representations Form a Global Workspace in Language Models"* — transformer-circuits.pub/2026/workspace
- Sergey Nikolenko, *"Global workspace in the J-space"* (in Russian) — https://www.sergeynikolenko.ru/blog/global-workspace-in-the-j-space

There is no public code for the post, so everything here is reimplemented from the method description
(~a few hundred lines of PyTorch, as the post estimated). The core readout and Jacobian are backed by an
independent validation suite — three bit-exact checks plus a ~1–2% finite-difference cross-check.

---

## The idea in one screen

Each token at layer $\ell$ is a residual vector $h_\ell \in \mathbb{R}^d$; the model ends with
$\mathrm{logits} = W_U \cdot \operatorname{norm}(h_L)$.

- **Logit lens** reads an intermediate state as if final: $\operatorname{softmax}\!\big(W_U \cdot \operatorname{norm}(h_\ell)\big)$.
- **J-lens** first maps $h_\ell$ through the *corpus-averaged linearization* of the rest of the network:

$$
\begin{aligned}
J_\ell &= \mathbb{E}_{\text{corpus},\; t' \ge t}\!\left[\frac{\partial h_{L,t'}}{\partial h_{\ell,t}}\right]
  \quad\text{(one } d \times d \text{ matrix per layer)} \\[4pt]
\operatorname{lens}_\ell(h) &= \operatorname{softmax}\!\big(W_U \cdot \operatorname{norm}(J_\ell\, h)\big)
  \quad\text{(generalizes the logit lens: } J = I) \\[4pt]
v_y &= (W_U J_\ell)_y
  \quad\text{(J-vector: how strongly } h \text{ pushes toward token } y)
\end{aligned}
$$

- **Steer** $h \leftarrow h + \alpha\, v_y$; **patch** = swap coordinates in the J-vector basis;
  **ablate** = remove the projection onto the active J-vectors.

$J_\ell$ is computed with a broadcast-perturbation trick: add a zero leaf $\delta_\ell \in \mathbb{R}^d$
(broadcast over positions) at each layer and backprop $G_i = \sum_{t'} h_{L,t',i}$; then the gradient
$\partial G_i / \partial \delta_\ell$ is row $i$ of the pair-summed Jacobian **for every layer at once**.
So $d$ backward passes give the full matrix.

---

## Environment

Detected/target stack (the reproduction is pinned to it):

- Apple M3 Max, 64 GB unified memory, macOS, `arm64`
- Python 3.11, **PyTorch 2.5.1** (MPS backend), **transformers 5.13.0**, `pyarrow` (for the corpus)
- All models run in **fp32** with **`attn_implementation="eager"`** (the finite differences and averaged
  Jacobian need fp32; eager avoids MPS SDPA/Gemma-2-padding pitfalls). No `datasets` install needed.

```bash
# use the existing interpreter (matches the verified stack); or:
uv venv && uv pip install "torch==2.5.1" "transformers==5.13.0" numpy safetensors huggingface_hub pyarrow
# Gemma is gated on HF (google/gemma-2-2b); accept the license + `huggingface-cli login` if not cached.
export PYTORCH_ENABLE_MPS_FALLBACK=1     # safety net for any op gap
```

Every script honors `JLENS_MODEL` (HF id) and `JLENS_DEVICE` (default `mps`).

---

## Layout

```
jlens/
  __init__.py    # package init (re-exports the submodules)
  models.py      # family-parametrized loader (eager, fp32, frozen, W_U, own final-norm) — Qwen2/3, Gemma2/3
  hooks.py       # JHooks: per-layer zero-init leaf deltas + raw h_L capture (transformers 5.x: bare Tensor)
  jacobian.py    # compute_jacobian (full averaged d×d J_ℓ) + compute_jvectors (per-token v_y via 1 VJP)
  lens.py        # logit_lens / j_lens readouts (family-aware norm + optional Gemma-2 soft-cap) + top_tokens
  steering.py    # Patch (J-vector swap) + ProjectOut (subspace ablation) + unit_jvectors
  corpus.py      # DEFAULT_TEXTS + wiki_texts()/stream_batches() (cached English Wikipedia via pyarrow)
  validate.py    # 4 correctness checks
scripts/
  validate_qwen.py        # run the 4 checks on any model
  eiffel_twohop.py        # two-hop: J-lens sees Paris/France mid-network (FD-JVP, full vocab)
  france_china.py         # France→China broadcast patch (+ Germany control)
  layer_profile.py        # three zones: lens top-1 vs model top-1 by depth (needs full J_ℓ)
  desires.py              # spoken answer vs internal J-space (instruct + chat template)
  ablation_active.py      # faithful per-input active-J-space ablation during generation (+ random control)
  ablation.py             # builds/caches the full Jacobian; initial static-subspace ablation exploration
  language_selectivity.py # Spanish→French language patch on two tasks
PLAN.md          # full technical build log + per-experiment findings (chronological)
```

---

## Results — what reproduced

Five of the six headline findings reproduce, several with near-exact numbers, across three model families.
Full details and raw tables in `PLAN.md`.

### ✅ 1. J-lens core — validated bit-exact on all three families
`validate_qwen.py` on Qwen2.5-1.5B, Qwen2.5-7B, and Gemma-2-2b, all green:
- **structural**: `logit_lens(raw h_L)` vs the model's own logits, max-diff **0.0** (incl. Gemma-2's
  `30·tanh` soft-cap, `(1+weight)` RMSNorm, tied √d-scaled unembedding, and Qwen's untied 7B head).
- **jlens-identity**: `j_lens(h, I) == logit_lens(h)`, max-diff **0.0**.
- **j-last-identity**: normalized `J_{L-1}` diagonal = `2/(S+1)` exactly, off-diagonal 0.
- **finite-difference**: `J_ℓ·v` vs a central-difference directional derivative, rel-err **0.85–2.1%**
  (also rules out the documented MPS silent-zero-Jacobian bug).

### ✅ 2. Eiffel two-hop — J-lens sees the answer mid-network, the logit lens can't
Prompt: *"…the capital of the country where the Eiffel Tower is located is the city of"* → Paris. Rank of
Paris over the full vocabulary, by layer:

| | J-lens finds Paris | Logit lens is still noise until |
|---|---|---|
| Qwen2.5-1.5B | rank 2–4 by layers 7–13 (France #1 at L6) | ~layer 22 (rank 173–7962 in the middle) |
| Qwen2.5-7B | rank 11–61 by layers 8–10 | ~layer 22 (rank 50k–120k in the middle) |
| Gemma-2-2b | rank ~89 by layers 7–8 | ~layer 18 (rank 5k–192k in the middle) |

Bonus details reproduced: a mid-network `city/Rome/Venice/Italy` geography cluster, Chinese `巴黎` at late
layers ("thinks in English/Chinese"), and the `____` "exam-habit" fill-in tokens.

### ✅ 3. France→China broadcast — one swap redirects many facts; control tightens with scale
Patching the `France`→`China` J-vector on layers 9–19 flips capital, language, continent, *and* currency at once:
- **1.5B**: Europe 0.70 → **Asia 0.70**, French → **Chinese 0.39**, Euro → **Renminbi**, capital → Beijing.
  Germany control **breaks** (Berlin 0.48 → 0.008) — matches the post ("European directions correlate").
- **7B**: Europe 0.88 → **Asia 0.90**; Germany control **holds** (Beijing ≈ 0.005 — same order as the post's
  ≈0.002). → reproduces the post's *scale-dependent selectivity of the control*.

### ✅ 4. Layer profile / three zones — and the scale crossover
How often each lens's top-1 matches the model's own top-1, by depth: **sensory** (~0), **workspace**
(slow rise), **motor** (sharp → 1.00 at the last layer).
- **1.5B**: the logit lens ties/beats the J-lens in the middle (post: "logit lens beats J-lens on small models").
- **7B**: the J-lens **beats** the logit lens through the mid-late layers — e.g. layer 19 **J-lens 0.107 vs
  logit-lens 0.027**, matching the post's *"0.10 vs 0.03 at layer 20"* almost exactly.

### ✅ 5. Desires / introspection — spoken answer vs internal J-space
Instruct models, one-word forced answer, internal read via FD-JVP at the answer position (Wikipedia-averaged J):
- **1.5B** says *AI / Good / deflects*, but J-space holds **robot 0.60, Busy ≈0.68, Happiness, Identity/Self,
  Unknown/Fear** — matching the post's 1.5B list item-for-item.
- **7B**: *"Who are you?"* → says **AI** while internally **assistant ≈ 0.8–0.95** (post: 0.95); *"afraid?"* →
  says **Darkness** while internally **未知 (unknown) ≈ 0.4–0.7** — the higher end from the Wikipedia-sharpened
  averaging corpus, the lower from the small corpus. Anecdotal by nature (persona/training artifacts).

### ❌ 6. Selectivity (property 5) — NOT reproduced at ≤7B (a large-model effect)
Tested two independent ways, at both scales, exhaustively:
- **Faithful active-J-space ablation** (per-input top-k active J-vectors, exact orthogonal removal, applied
  through multi-token generation, matched random control, swept k/band): does **not** collapse reasoning —
  reasoning is preserved/improved (ablation strips the `____` format habit). On 7B automatic drops only mildly
  and comparably to the random control; on 1.5B automatic degrades more (into multilingual gibberish). Across
  both, the equal-size random control is as- or *more*-destructive than the J-space ablation.
- **Spanish→French language patch**: the *"what language is this?"* answer **does** flip Spanish→French, but
  the same patch **also** fluently rewrites the continuation into French — a **coherent global language steer**,
  not the selective effect ("continuation unaffected") the post reports.

Both point to the same reason: at 1.5B–7B the J-space concept directions are used **pervasively**, so
interventions act **globally**; the paper's selectivity is on far larger models (Sonnet/Haiku/Opus) and uses a
per-input **gradient-pursuit** reconstruction over a learned overcomplete dictionary. Reported straight — not
tuned to look positive. (Genuine positive by-products: the language patch is a powerful global language steer,
and the France→China *control* selectivity did emerge with scale in #3.)

---

## How to run

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd JLence

# 1. Validate the core on any model
JLENS_MODEL=Qwen/Qwen2.5-1.5B-Instruct python3 scripts/validate_qwen.py
JLENS_MODEL=google/gemma-2-2b          python3 scripts/validate_qwen.py

# 2. Eiffel two-hop (full-vocab J-lens vs logit lens, by layer)
JLENS_MODEL=Qwen/Qwen2.5-7B-Instruct python3 scripts/eiffel_twohop.py

# 3. France→China broadcast patch  (JLENS_ALPHAS sweeps the strength)
JLENS_MODEL=Qwen/Qwen2.5-1.5B-Instruct JLENS_ALPHAS=1,2,4 python3 scripts/france_china.py

# 4. Layer profile / three zones   (computes the full d×d Jacobian in memory each run: ~90s @1.5B, ~13min @7B)
JLENS_MODEL=Qwen/Qwen2.5-7B-Instruct python3 scripts/layer_profile.py

# 5. Desires / introspection       (needs an -Instruct model; JLENS_NFRAG/JLENS_SEQ size the corpus)
JLENS_MODEL=Qwen/Qwen2.5-1.5B-Instruct python3 scripts/desires.py

# 6. Selectivity attempts (documented negative)
JLENS_MODEL=Qwen/Qwen2.5-7B-Instruct JLENS_K=12 python3 scripts/ablation_active.py
JLENS_MODEL=Qwen/Qwen2.5-7B-Instruct JLENS_ALPHAS=0,2,4 python3 scripts/language_selectivity.py
```

Notes:
- First run of a model downloads it; `google/gemma-2-2b` is gated (needs an accepted license / HF token).
- The full Jacobian is cached to `/tmp/jlens_J_<model>.pt` by `ablation.py` / `ablation_active.py` /
  `language_selectivity.py`, which reuse it for fast re-tuning of `k` / `α` / layer band. (`layer_profile.py`
  computes its own Jacobian in memory and does not read/write that cache.)

---

## Key techniques

- **Family-agnostic, bit-exact readout.** The lens reuses the model's *own* final RMSNorm module and
  `get_output_embeddings().weight`, so tied/untied heads, Gemma's $(1+\text{weight})$ norm + $\sqrt{d}$ embed
  scaling + $30\tanh(x/30)$ logit soft-cap all come out exactly right (`validate.structural` max-diff 0.0 everywhere).
- **FD-JVP probe readout.** For a specific probe, $J_\ell\, h$ **is** a corpus-averaged Jacobian-vector product,
  which we approximate cheaply by *central finite differences* over the averaging corpus (≈1–2% error, per the
  validation) — a **full-vocab J-lens readout without ever forming the $d \times d$ matrix**. Used by
  `eiffel_twohop` and `desires`. (The lens/readout helpers return pre-softmax logits; `top_tokens` softmaxes.)
- **Per-token J-vectors.** $v_y = W_U[y]\, J_\ell$ is one VJP (or a matvec against a cached `J`) — cheap steering
  directions that scale to big models (`france_china`, `language_selectivity`).
- **Per-input active-J-space ablation.** `ablation_active.py` selects, per position, the top-k J-vectors the
  state pushes toward and removes exactly their subspace (orthogonal projection) during generation.
- **Cache-backed Wikipedia corpus.** `wiki_texts()/stream_batches()` read the locally-cached HF Wikipedia
  Arrow shards via `pyarrow` (no `datasets`, no network) into paper-scale N×64-token averaging batches.

## Honest caveats

- **Selectivity (#6) does not reproduce at ≤7B** — it needs the paper's gradient-pursuit J-space + larger scale.
- **Desires (#5) are anecdotes** — token distributions shaped by training data and persona, not "real desires".
- **MPS specifics**: fp32 + eager throughout; `torch.func.jacrev/vmap` is avoided on MPS (silent-zero risk) in
  favor of sequential autograd / finite differences, cross-checked against the finite-difference validation.
- **Small averaging corpora** (used for speed) leave the very shallow workspace layers noisy; the paper-scale
  Wikipedia corpus in `desires.py` visibly cleans this up.
- Wall-clock (M3 Max, fp32): validation seconds; Eiffel/France-China/desires a few minutes; full-Jacobian
  layer-profile ~90s (1.5B) / ~13min (7B).
