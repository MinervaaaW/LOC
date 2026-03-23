# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

from torch.nn.attention.flex_attention import (
    flex_attention,
)


flex_attention = torch.compile(flex_attention, dynamic=False,  mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 4
torch._dynamo.config.accumulated_cache_size_limit = 192 * 4


__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def radial_attention(
        neighbor_query_states,
        neighbor_key_states,
        shifted_query_states_1,
        shifted_key_states_1,
        shifted_query_states_2,
        shifted_key_states_2,
        value_states,
        diag_size=1560*(8-1),
        block_mask_diag=None,
        block_mask_shift_1=None,
        block_mask_shift_2=None,
        dtype=torch.bfloat16,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert neighbor_query_states.device.type == 'cuda' and neighbor_query_states.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = neighbor_query_states.size(0), neighbor_query_states.size(1), neighbor_query_states.size(1), neighbor_query_states.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)
    
    neighbor_query_states = half(neighbor_query_states)
    shifted_query_states_1 = half(shifted_query_states_1)
    shifted_query_states_2 = half(shifted_query_states_2)
    neighbor_key_states = half(neighbor_key_states)
    shifted_key_states_1 = half(shifted_key_states_1)
    shifted_key_states_2 = half(shifted_key_states_2)
    value_states = half(value_states)

    shifted_query_states_1 = shifted_query_states_1.transpose(1, 2)
    shifted_key_states_1 = shifted_key_states_1.transpose(1, 2)
    shifted_query_states_2 = shifted_query_states_2.transpose(1, 2)
    shifted_key_states_2 = shifted_key_states_2.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    # import pdb;pdb.set_trace()
    shifted_out_1, shifted_lse_1 = flex_attention(
        shifted_query_states_1,
        shifted_key_states_1, 
        value_states,
        block_mask=block_mask_shift_1,
        return_lse=True)
    
    shifted_out_2, shifted_lse_2 = flex_attention(
        shifted_query_states_2,
        shifted_key_states_2, 
        value_states,
        block_mask=block_mask_shift_2,
        return_lse=True)
    
    bsz, kv_seq_len, _, head_dim = neighbor_query_states.size()
    neighbor_query_states = neighbor_query_states.transpose(1, 2)
    neighbor_key_states = neighbor_key_states.transpose(1, 2)
    diag_out, diag_lse = flex_attention(
        neighbor_query_states,
        neighbor_key_states,
        value_states,
        block_mask=block_mask_diag,
        return_lse=True,
    )  # [bsz, L, h, d]

    
    # L = diag_out.size(1)
    # N = shifted_out.size(1)
    shifted_out_1 = shifted_out_1.transpose(1, 2)
    shifted_out_2 = shifted_out_2.transpose(1, 2)
    diag_out = diag_out.transpose(1, 2)
    # import pdb;pdb.set_trace()
    assert diag_out.size(1) == shifted_out_1.size(1)
    assert diag_out.size(1) == shifted_out_2.size(1)
    # diag_lse = diag_lse.to(torch.float32)
    # shifted_lse = shifted_lse.to(torch.float32)
    attn_outputs = torch.stack([diag_out, shifted_out_1, shifted_out_2])
    logits = torch.stack([diag_lse, shifted_lse_1, shifted_lse_2]).to(torch.float32)
    max_logits = torch.max(logits, dim=0).values
    stable_logits = logits - max_logits.unsqueeze(0)

    lse_s = torch.exp(stable_logits).detach()
    lse_sum = torch.sum(lse_s, dim=0)
    lse_s /= lse_sum
    lse_s = lse_s.to(torch.bfloat16).transpose(2, 3)
    # import pdb;pdb.set_trace()
    attn_outputs *= lse_s.unsqueeze(-1)
    output = attn_outputs.sum(dim=0)

    return output.type(out_dtype)



def vrpr_attention(
        neighbor_query_states,
        neighbor_key_states,
        shifted_query_states_1,
        shifted_key_states_1,
        shifted_query_states_2,
        shifted_key_states_2,
        shifted_query_states_3,
        shifted_key_states_3,
        shifted_query_states_4,
        shifted_key_states_4,
        value_states,
        diag_size=1560*(16-1),
        block_mask_1=None,
        block_mask_2=None,
        block_mask_3=None,
        block_mask_4=None,
        dtype=torch.bfloat16,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert neighbor_query_states.device.type == 'cuda' and neighbor_query_states.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = neighbor_query_states.size(0), neighbor_query_states.size(1), neighbor_query_states.size(1), neighbor_query_states.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)
    
    neighbor_query_states = half(neighbor_query_states)
    shifted_query_states_1 = half(shifted_query_states_1)
    shifted_query_states_2 = half(shifted_query_states_2)
    shifted_query_states_3 = half(shifted_query_states_3)
    shifted_query_states_4 = half(shifted_query_states_4)
    neighbor_key_states = half(neighbor_key_states)
    shifted_key_states_1 = half(shifted_key_states_1)
    shifted_key_states_2 = half(shifted_key_states_2)
    shifted_key_states_3 = half(shifted_key_states_3)
    shifted_key_states_4 = half(shifted_key_states_4)
    value_states = half(value_states)

    # params
    # b, lq, lk, out_dtype = q.size(0), q.size(1), k.size1), q.dtype

    bsz, kv_seq_len, _, head_dim = neighbor_query_states.size()
    diag_out, diag_lse, _ = flash_attn.flash_attn_func(
        neighbor_query_states,
        neighbor_key_states,
        value_states,
        window_size=[diag_size, diag_size],
        return_attn_probs=True,
    )  # [bsz, L, h, d]

    # return diag_out.type(out_dtype)
    triangle_len = (
            kv_seq_len - diag_size
    )  # here we should use kv_seq_len rather than max_kv_len since we have paddings in qkv and attention_mask
    if triangle_len < 0:
        return diag_out
    shifted_query_states_1 = shifted_query_states_1.transpose(1, 2)
    shifted_key_states_1 = shifted_key_states_1.transpose(1, 2)
    shifted_query_states_2 = shifted_query_states_2.transpose(1, 2)
    shifted_key_states_2 = shifted_key_states_2.transpose(1, 2)
    shifted_query_states_3 = shifted_query_states_3.transpose(1, 2)
    shifted_key_states_3 = shifted_key_states_3.transpose(1, 2)
    shifted_query_states_4 = shifted_query_states_4.transpose(1, 2)
    shifted_key_states_4 = shifted_key_states_4.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    # import pdb;pdb.set_trace()
    shifted_out_1, shifted_lse_1 = flex_attention(
        shifted_query_states_1,
        shifted_key_states_1, 
        value_states,
        block_mask=block_mask_1,
        return_lse=True)
    shifted_out_2, shifted_lse_2 = flex_attention(
        shifted_query_states_2,
        shifted_key_states_2, 
        value_states,
        block_mask=block_mask_2,
        return_lse=True)
    shifted_out_3, shifted_lse_3 = flex_attention(
        shifted_query_states_3,
        shifted_key_states_3, 
        value_states,
        block_mask=block_mask_3,
        return_lse=True)
    shifted_out_4, shifted_lse_4 = flex_attention(
        shifted_query_states_4,
        shifted_key_states_4, 
        value_states,
        block_mask=block_mask_4,
        return_lse=True)
    # L = diag_out.size(1)
    # N = shifted_out.size(1)
    shifted_out_1 = shifted_out_1.transpose(1, 2)
    shifted_out_2 = shifted_out_2.transpose(1, 2)
    shifted_out_3 = shifted_out_3.transpose(1, 2)
    shifted_out_4 = shifted_out_4.transpose(1, 2)
    # import pdb;pdb.set_trace()
    assert diag_out.size(1) == shifted_out_1.size(1)
    assert diag_out.size(1) == shifted_out_2.size(1)
    assert diag_out.size(1) == shifted_out_3.size(1)
    assert diag_out.size(1) == shifted_out_4.size(1)


    attn_outputs = torch.stack([diag_out, shifted_out_1, shifted_out_2, shifted_out_3, shifted_out_4])
    logits = torch.stack([diag_lse, shifted_lse_1, shifted_lse_2, shifted_lse_3, shifted_lse_4]).to(torch.float32)
    max_logits = torch.max(logits, dim=0).values
    stable_logits = logits - max_logits.unsqueeze(0)

    lse_s = torch.exp(stable_logits).detach()
    lse_sum = torch.sum(lse_s, dim=0)
    lse_s /= lse_sum
    lse_s = lse_s.to(torch.bfloat16).transpose(2, 3)
    # import pdb;pdb.set_trace()
    attn_outputs *= lse_s.unsqueeze(-1)
    output = attn_outputs.sum(dim=0)

    return output.type(out_dtype)
