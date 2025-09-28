# Typewise Embeddings

Typewise Embeddings represent tokens as (type, value) and avoid row‑per‑value tables and V‑way softmax. Values live in large spaces (e.g., RGB with 16,777,216 values; int64 with ~9.22×10¹⁸), so parameters and compute scale with model dimension instead of number of values.

---

## Overview

- Input: map each value to a minimal representation (often a single quaternion) and up‑project to model space via a per‑type weight bank using Hamilton products.
- Transformer: standard attention + MLP stack.
- Output: a quaternion voting decoder inverts blocks of the final hidden state to produce a continuous estimate and confidence, then ranks a small set of discrete candidates (no giant softmax).

---

## Architecture

**Token representation**

- Token = (TYPE, VALUE).
- Each TYPE (e.g., `RGB`, `int64`) provides:
  - A mapping from VALUE to a Minimal Representation (MR), typically one quaternion q ∈ R⁴.
  - A per‑type Weight Bank W^{(t)} = {W^{(t)}₁…W^{(t)}_{N_q}}, quaternions.

**Up‑projection (embedding)**

For model dimension d_model divisible by 4, set N_q = d_model / 4 and compute blockwise Hamilton products:

[
Y = vec(q ⊗ W^{(t)}₁, …, q ⊗ W^{(t)}_{N_q}).
]

The same pattern can be used for a separate TYPE segment if needed; all segments are multiples of 4 dims.

**Transformer body**

- Unchanged; standard attention and MLP blocks operate on Y.

**Decoder (quaternion voting)**

- Split the final hidden state h_T into 4‑D blocks: ȳ₀, ȳ₁, …, ȳ_{N_q−1} ∈ R⁴.
- For each block and type weight W^{(t)}_i, form an inverse vote for the value quaternion:

[
q̂_i = ȳ_i ⊗ (W^{(t)}_i)^{-1} = ȳ_i ⊗ conj(W^{(t)}_i) / ∥W^{(t)}_i∥².
]

- Fuse votes with weights β_i = ∥W^{(t)}_i∥² to obtain a mean and a scalar spread (confidence):

[
μ_q = (∑_i ȳ_i ⊗ conj(W^{(t)}_i)) / (∑_i ∥W^{(t)}_i∥²),
\quad s² = (∑_i β_i · ∥q̂_i − μ_q∥²) / (∑_i β_i).
]

- Convert μ_q to the value domain and rank a small, local set of discrete candidates by how well they re‑explain the blocks (no V‑way enumeration).

The decoder reads directly from h_T. End‑to‑end training encourages the model to place recoverable signal in the designated blocks.

---

## RGB mapping example

Encode 8‑bit RGB into a pure‑imaginary quaternion Q = [0, r̃, g̃, b̃] on [−1, 1] with step 1/128:

[
r̃ = (2r − 255)/256, \quad g̃ = (2g − 255)/256, \quad b̃ = (2b − 255)/256, \quad Q = 0 + r̃·i + g̃·j + b̃·k.
]

After decoding μ_q = [μ₀, μ_r, μ_g, μ_b], map back to 8‑bit (clamp to [0,255]):

[
\hat r₈ = round((256·μ_r + 255)/2), \quad \hat g₈ = round((256·μ_g + 255)/2), \quad \hat b₈ = round((256·μ_b + 255)/2).
]

This powers‑of‑two grid aligns well with bf16 spacing; adjacent 8‑bit codes remain distinct.

---

## Training

- Type loss: cross‑entropy (or nearest‑code) on the decoded TYPE.
- Value loss: L2 or Gaussian NLL between decoded μ_q (imaginary parts) and ground‑truth value quaternion.
- Optional vote tightening: add ∑_i β_i ∥q̂_i − μ_q∥² to encourage confident votes when possible.
- Weight hygiene: parameterize W_i = s_i·Û_i with ∥Û_i∥=1 and s_i>0 (store log s_i) to keep inverses stable.

All components are differentiable with standard quaternion ops over real tensors.

---

## Minimal pseudocode

```python
# --- encode (one value, one type) ---
# q: minimal representation quaternion, shape (4,)
# W_t: quaternion weight bank for type t, shape (Nq, 4), Nq = d_model//4
# Up-project to model space (value segment shown; type segment can use its own bank)
Y_blocks = left_broadcast(q, W_t)   # (Nq, 4), row i is q ⊗ W_t[i]
Y = Y_blocks.flatten()              # (d_model,)

# --- decode (use h_T directly) ---
# inputs: h_T (d_model,), W_t (Nq,4)

def blockify(h_T):
    return h_T.view(-1, 4)  # (Nq, 4)

def decode_mu_and_spread(h_T, W_t, eps=1e-12):
    y_hat = blockify(h_T)                         # (Nq,4)
    votes_num = hamilton(y_hat, conj(W_t))        # y_i ⊗ conj(W_i)
    Wnorm2    = (W_t * W_t).sum(dim=-1, keepdim=True).clamp_min(eps)
    q_votes   = votes_num / Wnorm2                # per-block inverse vote
    # Fuse votes (equal-noise ML): weighted average with weights ||W_i||^2
    beta = Wnorm2.squeeze(-1)                     # (Nq,)
    mu_q = (q_votes * beta[:, None]).sum(0) / beta.sum()
    # Confidence proxy: weighted spread of votes
    spread2 = (beta * ((q_votes - mu_q) ** 2).sum(dim=-1)).sum() / beta.sum()
    return mu_q, spread2
```

Use μ_q as a continuous estimate; perform a small local search in 8‑bit space and rank candidates by reconstruction quality:

[
−∑_i ∥ȳ_i − q(c) ⊗ W^{(t)}_i∥².
]

---

## Defaults and notes

- Dimensions: choose d_type and d_value divisible by 4 (e.g., d_type=128, d_value=3968 for d_model=4096).
- Type mapping: give each type a small quaternion codebook; decode the TYPE segment with the same voting mechanism and pick the nearest code.
- RGB top‑k neighborhood: m=7 bins per channel (≤343 candidates) is a practical default.
- Dtypes: activations/weights in bf16, reductions in fp32.

---

## Glossary

- Typewise Token — a token represented as (TYPE, VALUE).
- Minimal Representation (MR) — smallest structured form of a value (e.g., pure‑imaginary quaternion [0,r,g,b] for RGB).
- Weight Bank — per‑type list of learned quaternions W^{(t)}_{1..N_q} used for up‑projection.
- Quaternion Lift / Up‑Projection — Y = vec(q ⊗ W^{(t)}_i)_{i=1}^{N_q} to reach d_model.
- (Type|Value) Segment — slice of the hidden state reserved for decoding TYPE or VALUE (multiple of 4 dims).
- Block — a 4‑D slice of a segment; corresponds to one quaternion.
- Vote — per‑block inverse estimate of the value: q̂_i = ȳ_i ⊗ (W^{(t)}_i)^{-1}.
- Vote Fusion — weighted average of votes giving μ_q and a spread (confidence).
- Candidate Neighborhood — nearby 8‑bit bins per channel around μ_q used to produce top‑k predictions.
- Quaternion Voting Decoder (QVD) — output head performing blocks → votes → fuse → top‑k.

---

This repository provides a table‑free embedding/decoding scheme for large value spaces using the same structured mechanism for TYPE and VALUE, with a decoder that operates directly on the final hidden state without V‑way enumeration.

