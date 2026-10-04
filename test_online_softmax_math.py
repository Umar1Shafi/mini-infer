# test_online_softmax_math.py
"""
Proves the online-softmax algorithm (used in FlashAttention / multi-page
paged-attention) is mathematically identical to a normal, one-shot softmax.

Setup: one query vector attends over a KV sequence that is split into
multiple pages (chunks). We compute attention two ways:

  1. "Reference" - concatenate everything, do one normal softmax (what we
     already trust).
  2. "Online" - process the KV sequence one page at a time, maintaining a
     running max (m), running sum of exponentials (l), and a running
     weighted output accumulator (acc), updating all three correctly as
     each new page arrives, without ever seeing the full sequence at once.

If the online method is correct, the two outputs must match exactly
(up to float rounding).
"""

import torch

torch.manual_seed(0)

HEAD_DIM = 64
PAGE_SIZE = 16
NUM_PAGES = 3
TOTAL_LEN = 37  # deliberately NOT a multiple of PAGE_SIZE -> last page is partial

scale = 1.0 / (HEAD_DIM ** 0.5)

# A single query vector, and a KV sequence long enough to span 3 pages
# (16 + 16 + 5 = 37), with the last page partially filled.
q = torch.randn(HEAD_DIM)
k_full = torch.randn(TOTAL_LEN, HEAD_DIM)
v_full = torch.randn(TOTAL_LEN, HEAD_DIM)

# Split into pages, exactly like the paged cache would store them.
# The last page only has 5 real tokens; we pad it up to PAGE_SIZE with
# zeros to simulate unused page slots (the kernel will mask these out).
pages_k = []
pages_v = []
page_lens = []
for start in range(0, TOTAL_LEN, PAGE_SIZE):
    end = min(start + PAGE_SIZE, TOTAL_LEN)
    length = end - start

    k_chunk = torch.zeros(PAGE_SIZE, HEAD_DIM)
    v_chunk = torch.zeros(PAGE_SIZE, HEAD_DIM)
    k_chunk[:length] = k_full[start:end]
    v_chunk[:length] = v_full[start:end]

    pages_k.append(k_chunk)
    pages_v.append(v_chunk)
    page_lens.append(length)

print(f"Split {TOTAL_LEN} tokens into {len(pages_k)} pages of size {PAGE_SIZE}")
print(f"Page lengths (last one partial): {page_lens}")

# ---------------------------------------------------------------------
# 1. REFERENCE: normal one-shot softmax over the whole sequence at once
# ---------------------------------------------------------------------
scores_ref = (k_full @ q) * scale          # (TOTAL_LEN,)
probs_ref = torch.softmax(scores_ref, dim=0)
out_ref = probs_ref @ v_full               # (HEAD_DIM,)

# ---------------------------------------------------------------------
# 2. ONLINE: process one page at a time, no global view of the sequence
# ---------------------------------------------------------------------
m_i = torch.tensor(float("-inf"))          # running max
l_i = torch.tensor(0.0)                    # running sum of exponentials
acc = torch.zeros(HEAD_DIM)                # running weighted V accumulator

for page_idx in range(NUM_PAGES):
    k_block = pages_k[page_idx]            # (PAGE_SIZE, HEAD_DIM)
    v_block = pages_v[page_idx]
    length = page_lens[page_idx]

    # Scores for this page only. Mask out padding slots with -inf so they
    # contribute zero probability (matches what the real kernel will do
    # with a validity mask).
    scores_block = (k_block @ q) * scale   # (PAGE_SIZE,)
    valid = torch.arange(PAGE_SIZE) < length
    scores_block = torch.where(valid, scores_block, torch.tensor(float("-inf")))

    # --- online softmax update ---
    m_new = torch.maximum(m_i, scores_block.max())

    # Rescale factor for everything accumulated so far, now that the
    # running max may have changed.
    correction = torch.exp(m_i - m_new)

    p = torch.exp(scores_block - m_new)    # unnormalized probs for this page
    # padded slots: exp(-inf - m_new) = 0, so they vanish automatically

    l_new = l_i * correction + p.sum()
    acc_new = acc * correction + (p.unsqueeze(1) * v_block).sum(dim=0)

    m_i, l_i, acc = m_new, l_new, acc_new

out_online = acc / l_i

# ---------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------
diff = (out_ref - out_online).abs().max().item()
print(f"\nMax difference (online softmax vs. one-shot softmax): {diff:.2e}")
if diff < 1e-5:
    print("Match! Online softmax math is correct.")
else:
    print("MISMATCH - do not proceed to the Triton kernel yet.")
