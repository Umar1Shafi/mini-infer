import torch
import triton
import triton.language as tl

@triton.jit
def paged_attention_decode_single_page_kernel(
    q_ptr,              # (num_heads, head_dim) - the new token's query, all heads
    k_pool_ptr,         # (num_pages, page_size, num_kv_heads, head_dim) - THIS LAYER's K pool
    v_pool_ptr,         # same shape - THIS LAYER's V pool
    out_ptr,            # (num_heads, head_dim) - output
    physical_page,      # int: which physical page this sequence's (only) page lives in
    seq_len,            # int: how many of the page's slots are actually valid tokens
    num_groups,         # int: num_heads // num_kv_heads (GQA sharing factor)
    scale,              # float: 1/sqrt(head_dim)
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    head_idx = tl.program_id(0)
    kv_head_idx = head_idx // num_groups  # which shared KV head this query head reads from

    # ---- Load this head's query vector ----
    q_offsets = head_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + q_offsets).to(tl.float32)  # (HEAD_DIM,)

    # ---- Compute the base address of this page, for this kv_head, in the pool ----
    # Pool layout: (num_pages, page_size, num_kv_heads, head_dim)
    page_stride = PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM
    slot_stride = NUM_KV_HEADS * HEAD_DIM
    base = physical_page * page_stride + kv_head_idx * HEAD_DIM

    slot_offsets = tl.arange(0, PAGE_SIZE)           # which of the 16 slots in this page
    dim_offsets = tl.arange(0, HEAD_DIM)
    valid = slot_offsets < seq_len                    # mask out slots beyond this sequence's real length

    # ---- Load the whole page's K and V for this kv_head: (PAGE_SIZE, HEAD_DIM) ----
    kv_ptrs_offset = base + slot_offsets[:, None] * slot_stride + dim_offsets[None, :]
    k_block = tl.load(k_pool_ptr + kv_ptrs_offset, mask=valid[:, None], other=0.0).to(tl.float32)
    v_block = tl.load(v_pool_ptr + kv_ptrs_offset, mask=valid[:, None], other=0.0).to(tl.float32)

    # ---- Attention scores: dot product of q with every cached key ----
    scores = tl.sum(q[None, :] * k_block, axis=1) * scale        # (PAGE_SIZE,)
    scores = tl.where(valid, scores, float("-inf"))

    # ---- Softmax ----
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    l = tl.sum(p, axis=0)

    # ---- Weighted sum of values ----
    out = tl.sum(p[:, None] * v_block, axis=0) / l                 # (HEAD_DIM,)

    tl.store(out_ptr + head_idx * HEAD_DIM + dim_offsets, out)


def paged_attention_decode_single_page(q, k_pool_layer, v_pool_layer, physical_page, seq_len, num_heads, num_kv_heads, head_dim, page_size):
    """
    q: (num_heads, head_dim) - this step's new token's query, ALL heads, ROPE ALREADY APPLIED
    k_pool_layer, v_pool_layer: (num_pages, page_size, num_kv_heads, head_dim) for ONE layer
    Returns: (num_heads, head_dim)
    Precondition: seq_len <= page_size (single-page case only)
    """
    assert seq_len <= page_size, "This kernel only supports sequences fitting in one page"
    out = torch.empty(num_heads, head_dim, dtype=torch.float32, device=q.device)
    scale = 1.0 / (head_dim ** 0.5)
    num_groups = num_heads // num_kv_heads

    grid = (num_heads,)
    paged_attention_decode_single_page_kernel[grid](
        q, k_pool_layer, v_pool_layer, out,
        physical_page, seq_len, num_groups, scale,
        PAGE_SIZE=page_size, HEAD_DIM=head_dim, NUM_KV_HEADS=num_kv_heads,
    )
    return out

@triton.jit
def paged_attention_decode_multi_page_kernel(
    q_ptr, k_pool_ptr, v_pool_ptr, out_ptr, page_table_ptr,
    seq_len, num_groups, scale,
    PAGE_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr, MAX_PAGES: tl.constexpr,
):
    head_idx = tl.program_id(0)
    kv_head_idx = head_idx // num_groups

    q_offsets = head_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + q_offsets).to(tl.float32)

    page_stride = PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM
    slot_stride = NUM_KV_HEADS * HEAD_DIM
    slot_offsets = tl.arange(0, PAGE_SIZE)
    dim_offsets = tl.arange(0, HEAD_DIM)

    # Online-softmax running state (proven correct in test_online_softmax_math.py)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    num_pages = tl.cdiv(seq_len, PAGE_SIZE)

    for page_idx in range(MAX_PAGES):
        page_active = page_idx < num_pages

        # Masked load: if page_active is False, this never actually
        # dereferences memory, so reading past the real page table is safe.
        physical_page = tl.load(page_table_ptr + page_idx, mask=page_active, other=0)

        token_start = page_idx * PAGE_SIZE
        valid = page_active & ((slot_offsets + token_start) < seq_len)

        base = physical_page * page_stride + kv_head_idx * HEAD_DIM
        kv_ptrs_offset = base + slot_offsets[:, None] * slot_stride + dim_offsets[None, :]

        k_block = tl.load(k_pool_ptr + kv_ptrs_offset, mask=valid[:, None], other=0.0).to(tl.float32)
        v_block = tl.load(v_pool_ptr + kv_ptrs_offset, mask=valid[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))

        # --- online softmax update (identical math to the PyTorch proof) ---
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        correction = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * correction + tl.sum(p, axis=0)
        acc = acc * correction + tl.sum(p[:, None] * v_block, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + head_idx * HEAD_DIM + dim_offsets, out)


def paged_attention_decode_multi_page(
    q, k_pool, v_pool, page_table, seq_len,
    num_heads, num_kv_heads, head_dim, page_size, max_pages,
):
    """
    q: (num_heads * head_dim,) flat query vector for this one decode step
    k_pool, v_pool: (num_pages, page_size, num_kv_heads, head_dim) physical pool
    page_table: (max_pages,) int32 tensor mapping logical page -> physical page
    seq_len: number of real tokens (including the ones being written this step)
    """
    assert q.is_cuda and k_pool.is_cuda and v_pool.is_cuda and page_table.is_cuda

    out = torch.empty(num_heads * head_dim, device=q.device, dtype=torch.float32)
    num_groups = num_heads // num_kv_heads
    scale = 1.0 / (head_dim ** 0.5)

    grid = (num_heads,)
    paged_attention_decode_multi_page_kernel[grid](
        q, k_pool, v_pool, out, page_table,
        seq_len, num_groups, scale,
        PAGE_SIZE=page_size, HEAD_DIM=head_dim,
        NUM_KV_HEADS=num_kv_heads, MAX_PAGES=max_pages,
    )
    return out
