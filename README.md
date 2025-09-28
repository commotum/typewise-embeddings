# Type & Value Embeddings (Goodbye, Giant Tables)

## The problem

Traditional LLMs predict the next token by comparing their final hidden vector against **every row of a giant embedding table** and softmaxing. That’s feasible for vocabularies ~100K. It’s **not** feasible when tokens carry **huge value spaces**:

* **RGB** has **16,777,216** distinct values.
* **int64** has ≈ **9.22×10¹⁸** values.

A row‑per‑value embedding table (and a V‑way softmax) explodes in size and compute.

---

## The idea (type + value)

Represent each token as **(type, value)** and **never** build a row‑per‑value table.

* Each **type** (e.g., `RGB`, `int64`, …) has a small **weight bank** of quaternions.
* Each **value** is mapped to a tiny **minimal representation** (often a single quaternion).
* A fast **up‑projection** (Hamilton product) lifts that 4‑D minimal representation into the model dimension.

This turns “table lookup” into **structured math**—the parameters scale with **model dimension**, not with **#values**.

---

## What we replace (and with what)

**Input side (embedding):**

* ❌ **Replace:** “learned embedding table with one row per token.”
* ✅ **With:** three pieces per **type**:

  1. **Type set** (what kinds of tokens exist).
  2. **Mapping function** from a concrete value to its **Minimal Representation** (MR), usually one quaternion (4 numbers).
  3. **Weight Bank** $\left(W^{(t)}_{1,\dots,N_q}\right)$ (quaternions) for the type, used to **up‑project** the MR to the model dimension by blockwise Hamilton products:

$$
\underbrace{Y}_{d_{\text{model}}}
= \mathrm{vec}\big(q \otimes W^{(t)}_1,\dots,q \otimes W^{(t)}_{N_q}\big),
\quad N_q = d_{\text{model}}/4.
$$

**Transformer body:** unchanged (standard attention + MLP blocks).

**Output side (prediction):**

* ❌ **Replace:** “V‑way softmax against every value’s embedding.”
* ✅ **With:** a **quaternion voting decoder** that:

  1. **Splits** the final hidden state ($h_T$) into 4‑D blocks.
  2. **Inverts** each block with the type’s Weight Bank to get a **vote** for the value.
  3. **Fuses** the votes into a **best guess + confidence** (a Gaussian “bell curve” in value space).
  4. **Converts** that fuzzy guess into a small set of **top‑k discrete values** (no giant softmax).

> **No anchors. No extra readout.** We directly use the final hidden state ($h_T$) as the source of block clues; training makes the model put the right information there.

**Why this still works despite nonlinear attention/MLPs.**

We’re not trying to *invert the whole transformer*; we’re training it **end‑to‑end** so that, after all the nonlinear mixing, the final hidden state ($h_T$) *writes the right clues into the right 4‑D slots* for our decoder to read. This is the same reason a vanilla GPT can use a simple **linear output head** after many nonlinear layers: the network learns a representation that is linearly decodable because the loss demands it. Here, our head is a slightly more structured linear reader: each 4‑D block in ($h_T$) is treated as a noisy “measurement” of the value under that block’s quaternion code; we **invert per block** to get votes, then **fuse** them (least‑squares/“Gaussian” mean) to produce a best guess and a confidence. Upstream nonlinearities don’t break this—they just change how the model *encodes* information internally; the loss on the decoded value pushes the model to **place recoverable signal** in those blocks so that the vote fusion is accurate (skinny bell curve when votes agree, wide when they don’t). In short: we don’t assume ($h_T$) equals the pre‑mix embedding; we **define a decodable head**, train against it, and the transformer learns to make the head’s job easy—exactly how large language models already learn to make a simple output layer work.


---

## How decoding works (the “voting” Gaussian)

**1) Split into clues**
Take the final hidden state and chop it into 4‑number blocks:

$$
\widehat{y}_0,\;\widehat{y}_1,\;\dots,\;\widehat{y}_{N_q-1}\in\mathbb{R}^4.
$$

**2) Per‑block vote**
Each type ($t$) has its learned quaternions ($W^{(t)}_i$). “Undo” each code to get a vote for the value quaternion:

$$
\widehat{q}_i = \widehat{y}_i \otimes \big(W^{(t)}_i\big)^{-1}
= \widehat{y}_i \otimes \frac{\mathrm{conj}(W^{(t)}_i)}{\lvert W^{(t)}_i\rvert^2}.
$$

**3) Fuse votes → best guess + confidence**

* **Best guess (mean):**

$$
\boxed{\mu_q =
\frac{\sum_i \widehat{y}_i \otimes \mathrm{conj}(W^{(t)}_i)}
{\sum_i \lvert W^{(t)}_i\rvert^2}}
$$

  (equivalently: a weighted average of the per‑block inverse votes with weights $\beta_i=\lvert W^{(t)}_i\rvert^2$).

* **Confidence (variance):** how tightly the votes cluster. A simple, effective scalar:

$$
s^2 = \frac{\sum_i \beta_i \,\lvert\widehat{q}_i - \mu_q\rvert^2}{\sum_i \beta_i}.
$$

  Skinny bell curve (small $s^2$) → high confidence. Wide bell curve → low confidence.

**4) From fuzzy guess to discrete values (no giant softmax)**

* Convert $\mu_q$ to the float domain for RGB, then pick the **nearest few** bins per channel (e.g., 7 closest red, 7 green, 7 blue → ≤ 343 candidates).
* **Score** each candidate color ($c$) by how well it explains the clues:

$$
\mathrm{Score}(c) = -\sum_i \left\lvert\widehat{y}_i - q(c)\otimes W^{(t)}_i\right\rvert^2.
$$

* Take **top‑k** or **sample** by a softmax over these scores.

That’s it—**no comparison against millions** of rows.

---

## Unifying type & value (one mechanism for both)

You can allocate fixed slices of the hidden state for **type** and **value** and run the **same** voting decoder on each:

* Choose **$d_{\text{type}}$** and **$d_{\text{value}}$** (both divisible by 4).
* Concatenate the two segments in the token embedding and keep them separate through the model (they’re just parts of the same $h_T$).
* Maintain **two Weight Banks**:

  * $W^{(\text{TYPE})}$ for the **type** segment,
  * $W^{(t)}$ for the **value** segment (conditioned on which type you decoded).
* **Decode TYPE** first from its segment using the same per‑block voting. Then use that predicted type to select the value Weight Bank and **decode VALUE** from its segment.

Because the number of types is small, picking the discrete type from the type decoder’s $\mu_q$ is trivial (nearest code or small softmax over the type set).

---

## bf16‑friendly RGB ↔ quaternion mapping (Minimal Representation)

Use a centered, power‑of‑two‑aligned grid so bf16 never collapses adjacent bins.

**Map 8‑bit to $[-1,1]$ bin centers (purely imaginary quaternion):**

$$
\tilde{r}=\frac{2r-255}{256},\quad
\tilde{g}=\frac{2g-255}{256},\quad
\tilde{b}=\frac{2b-255}{256},\qquad
Q = 0 + \tilde{r}\,\mathbf{i} + \tilde{g}\,\mathbf{j} + \tilde{b}\,\mathbf{k}.
$$

* Step size is exactly $2/256 = 1/128$ on $[-1,1]$.
* Powers‑of‑two grid aligns with **bf16** spacing; adjacent 8‑bit codes remain distinct.

**Map back to 8‑bit after decoding $\widehat{Q}=[0,\hat{r},\hat{g},\hat{b}]$:**

$$
\widehat{r}_{\text{8-bit}} = \mathrm{round}\!\left(\frac{256\,\hat{r} + 255}{2}\right)
\quad\text{(and same for }g,b\text{; clamp to }[0,255]\text{).}
$$

---

## Training (what the model learns)

* **Type loss:** cross‑entropy (or nearest‑code loss) on the decoded type.
* **Value loss:** L2 (or Gaussian NLL) between decoded $\mu_q$ (imaginary parts) and ground‑truth RGB quaternion.
* **Optional vote‑tightening:** small penalty $\sum_i \beta_i \,\lvert\widehat{q}_i-\mu_q\rvert^2$ to encourage confident, skinny bell curves when the model can be certain.
* **Weight hygiene:** parametrize each $W_i$ as $(W_i=s_i,\, \hat W_i)$ with $\lvert\hat W_i\rvert=1$ and $s_i>0$ (store $\log s_i$); this keeps inverses stable.

Everything autograds cleanly (just quaternion ops over real tensors).

---

## Why this works

* Complexity is **$O(d_{\text{model}})$**, not $O(\#\text{values})$.
* The decoder gives a **best guess and a confidence**, not just a number.
* No giant tables; **no enumeration** over 16.7M colors—ever.

---

## Minimal pseudocode (encode & decode)

```python
# --- encode (one value, one type) ---
# q: minimal representation quaternion, shape (4,)
# W_t: quaternion weight bank for type t, shape (Nq, 4), Nq = d_model//4
# up-project to model space (value segment shown; type segment uses its own bank)
Y_blocks = left_broadcast(q, W_t)   # (Nq, 4) where each row is q ⊗ W_t[i]
Y = Y_blocks.flatten()              # (d_model,)

# --- decode (no anchors/readout; use h_T directly) ---
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

Use `mu_q` as the continuous estimate; do small **local search** in 8‑bit space and **rank** candidates by how well they re‑explain the blocks:
$$
-\sum_i \left\lvert\widehat{y}_i - q(c)\otimes W^{(t)}_i\right\rvert^2.
$$

---

## Design choices (defaults we recommend)

* **`d_type`, `d_value`:** choose both divisible by 4 (e.g., $d_{\text{type}}=128$, $d_{\text{value}}=3968$ for $d_{\text{model}}=4096$).
* **Type mapping:** give each type a small **codebook** in quaternion space (often just one pure‑imaginary unit quaternion per type works); decode the type segment with the same voting mechanism, then pick the **nearest code** (tiny set).
* **Neighborhood size for top‑k RGB:** $m=7$ bins per channel (≤ 343 candidates) is a good start.
* **Dtypes:** activations/weights in **bf16**, reductions in **fp32**.

---

## Glossary

* **Typewise Token** — a token represented as **(TYPE, VALUE)**.
* **Minimal Representation (MR)** — the smallest structured form of a value (e.g., one **pure‑imaginary quaternion** ($[0,r,g,b]$) for RGB).
* **Weight Bank** — per‑type list of learned **quaternions** $\left(W^{(t)}_{1,\dots,N_q}\right)$ used for up‑projection.
* **Quaternion Lift / Up‑Projection** — compute $Y = \mathrm{vec}\big(q \otimes W^{(t)}_i\big)_{i=1}^{N_q}$ to reach $d_{\text{model}}$.
* **(Type|Value) Segment** — the slice of the hidden state reserved for decoding TYPE or VALUE (each a multiple of 4 dims).
* **Block** — a 4‑D slice of a segment; corresponds to one quaternion.
* **Vote** — per‑block inverse estimate of the value: $\widehat{q}_i = \widehat{y}_i \otimes (W^{(t)}_i)^{-1}$.
* **Vote Fusion (Gaussian Fuse)** — weighted average of votes to get **$\mu_q$** and a **spread** (confidence).
* **Candidate Neighborhood** — the small set of nearby 8‑bit bins per channel around $\mu_q$ used to produce **top‑k** discrete predictions.
* **Quaternion Voting Decoder (QVD)** — the whole output head that performs blocks→votes→fuse→top‑k.
* **Type Codebook** — the small set of canonical quaternions used to identify the discrete **TYPE** from its segment.

---

This is a complete, table‑free recipe: **same structured mechanism for TYPE and VALUE**, no anchors, no readout, and a principled “bell‑curve” decoder that turns $d_{\text{model}}$ math into high‑confidence discrete predictions **without** ever touching a 16.7M‑way softmax.
