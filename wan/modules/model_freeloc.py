# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from enum import Flag
from gc import enable
import math
from functools import lru_cache

from numpy import True_
from regex import D
from scipy import sparse
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention, vrpr_attention, radial_attention

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)

import torch._inductor.runtime.triton_heuristics as hints

# 增加 Triton 最大块大小
hints.TRITON_MAX_BLOCK["X"] = 4096

flex_attention = torch.compile(flex_attention, dynamic=False,  mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 4
torch._dynamo.config.accumulated_cache_size_limit = 192 * 4

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def prepare_flexattention(device, frame_size, context_length=0, num_frame=20, \
multiplier=2
):  
    # context_length=0 for Wan2.1
    seq_len = context_length + num_frame * frame_size
    
    mask_mod = generate_mask_mod(token_per_frame=frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)

    return block_mask


def prepare_flexattention_VRPR(device, frame_size, context_length=0, num_frame=20, \
window_size=2, window_size_2=4,
):  
    # context_length=0 for Wan2.1
    seq_len = context_length + num_frame * frame_size
    
    mask_mods = generate_VRPR_mask_mod(token_per_frame=frame_size, window_size=window_size, window_size_2=window_size_2)

    block_masks = []
    for mask_mod in mask_mods:
        block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)
        block_masks.append(block_mask)
       # hidden_states = flex_attention(query, key, value, block_mask=block_mask)

    return block_masks

def prepare_flexattention_radial_attention(device, frame_size, context_length=0, num_frame=20, \
window_size=2, window_size_2=4, enbale_relative_position_map=True, dtype=torch.bfloat16
):  
    # context_length=0 for Wan2.1
    seq_len = context_length + num_frame * frame_size

    if not enbale_relative_position_map:
        mask_mod= generate_radial_mask_mod(token_per_frame=frame_size, window_size=window_size, window_size_2=window_size_2,use_first_frame_mask=True)
        block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)
    else:
        mask_mods= generate_radial_mask_mod_plus(token_per_frame=frame_size, window_size=window_size, window_size_2=window_size_2,use_first_frame_mask=True)
        block_mask = [create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True) for mask_mod in mask_mods]

    return block_mask


@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask

def generate_mask_mod(token_per_frame=1350, mul=2,use_first_frame_mask=True):
    
    def round_to_multiple(idx):
        return math.ceil(idx / 128) * 128

    def mask_mod(b, h, q_idx, kv_idx):
        two_frames = round_to_multiple(token_per_frame * mul)
        local_attn_mask = (torch.abs(q_idx - kv_idx) <= two_frames)
        
        if use_first_frame_mask:
            first_frame_mask = (kv_idx < token_per_frame)
            video_mask = first_frame_mask | local_attn_mask
        else:
            video_mask = local_attn_mask
        return video_mask
        
    return mask_mod


def generate_VRPR_mask_mod(token_per_frame=1350, window_size=2, window_size_2=4,):
    
    def round_to_multiple(idx):
        return math.ceil(idx / 128) * 128
    
    window_token_num = round_to_multiple(token_per_frame * window_size)
    window_token_num_2 = round_to_multiple(token_per_frame * window_size_2)

    def mask_mod_1(b, h,q_idx, kv_idx):
        shift_attn_mask = (q_idx-kv_idx>=window_token_num_2)
        return shift_attn_mask
    
    def mask_mod_2(b, h,q_idx, kv_idx):
        shift_attn_mask_1 = (q_idx - kv_idx >= window_token_num)
        shift_attn_mask_2 = (q_idx - kv_idx < window_token_num_2)
        shift_attn_mask = shift_attn_mask_1 & shift_attn_mask_2
        return shift_attn_mask
    
    def mask_mod_3(b, h,q_idx, kv_idx):
        shift_attn_mask_1 = (q_idx - kv_idx > -window_token_num_2)
        shift_attn_mask_2 =  (q_idx - kv_idx <= -window_token_num)
        shift_attn_mask = shift_attn_mask_1 & shift_attn_mask_2
        return shift_attn_mask
    
    def mask_mod_4(b, h,q_idx, kv_idx):
        shift_attn_mask = (q_idx - kv_idx <= -window_token_num_2)
        return shift_attn_mask
        
    return mask_mod_1, mask_mod_2, mask_mod_3, mask_mod_4

def generate_radial_mask_mod(token_per_frame=1350, window_size=2, window_size_2=4,use_first_frame_mask=True, frame_idx=78):
    
    def round_to_multiple(idx):
        return math.ceil(idx / 128) * 128
    
    def get_token_id_in_frame(idx, prompt_length):
        return (idx - prompt_length) % token_per_frame

    band = (window_size_2 - window_size) // (window_size/2)
    width = band / (window_size_2-window_size)
    mask_width = (width * token_per_frame) // 2
    # band_2 = band // 2
    # mask_width_2 =  math.ceil((band_2 / (81-window_size_2)) * token_per_frame / 2)
    window_token_num = round_to_multiple(token_per_frame * window_size)
    window_token_num_2 = round_to_multiple(token_per_frame * window_size_2)
    # import pdb;pdb.set_trace()
    def mask_mod(b, h, q_idx, kv_idx):
        shift_attn_mask =  (torch.abs(q_idx - kv_idx) > window_token_num) & (torch.abs(q_idx - kv_idx) <= window_token_num_2)
        q_token_id = get_token_id_in_frame(q_idx, 0)
        kv_token_id = get_token_id_in_frame(kv_idx, 0)
        shift_attn_mask_tmp = torch.abs(q_token_id - kv_token_id) < mask_width
        attn_mask = shift_attn_mask & shift_attn_mask_tmp
        if use_first_frame_mask:
            first_column_mask = kv_idx < round_to_multiple(token_per_frame * 1)

            first_column_mask = first_column_mask 
        attn_window = torch.abs(q_idx - kv_idx) <= window_token_num

        return first_column_mask | attn_mask | attn_window 
    
    return mask_mod

def generate_radial_mask_mod_plus(token_per_frame=1350, window_size=2, window_size_2=4,use_first_frame_mask=True):
    
    def round_to_multiple(idx):
        return math.ceil(idx / 128) * 128
    
    def get_token_id_in_frame(idx, prompt_length):
        return (idx - prompt_length) % token_per_frame

    band = (window_size_2 - window_size) // (window_size/2)
    width = band / (window_size_2-window_size)
    mask_width = (width * token_per_frame) // 2
    window_token_num = round_to_multiple(token_per_frame * window_size)
    window_token_num_2 = round_to_multiple(token_per_frame * window_size_2)

    def mask_mod_1(b, h,q_idx, kv_idx):
        if use_first_frame_mask:
            first_column_mask = (kv_idx < round_to_multiple(token_per_frame * 1)) & (q_idx - kv_idx > window_token_num_2)

        return first_column_mask
    
    def mask_mod_2(b, h,q_idx, kv_idx):
        if use_first_frame_mask:
            first_column_mask = (kv_idx > round_to_multiple(token_per_frame * 78)) & (q_idx - kv_idx < -window_token_num_2)

        return first_column_mask
    
    def mask_mod_3(b, h,q_idx, kv_idx):
        shift_attn_mask =  (torch.abs(q_idx - kv_idx) > window_token_num) & (torch.abs(q_idx - kv_idx) <= window_token_num_2)

        q_token_id = get_token_id_in_frame(q_idx, 0)
        kv_token_id = get_token_id_in_frame(kv_idx, 0)
        shift_attn_mask_tmp = torch.abs(q_token_id - kv_token_id) < mask_width
        attn_mask = shift_attn_mask & shift_attn_mask_tmp

        if use_first_frame_mask:
            first_column_mask = (kv_idx < round_to_multiple(token_per_frame * 1))  & (torch.abs(q_idx - kv_idx) <= window_token_num_2)

        attn_window = torch.abs(q_idx - kv_idx) <= window_token_num

        return attn_window | first_column_mask | attn_mask

    return mask_mod_1, mask_mod_2, mask_mod_3

        
def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

@amp.autocast(enabled=False)
def rope_params_NTK(max_seq_len, dim, theta=10000, L_prime=None, L=None):
    assert dim % 2 == 0
    assert L_prime is not None and L is not None
    new_base = theta * (L_prime / L)**(dim / (dim - 2))
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(new_base,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim))) #(L. C)
    freqs = torch.polar(torch.ones_like(freqs), freqs) # (L, C)
    return freqs

@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
            torch.arange(max_seq_len),
            1.0 / torch.pow(theta,
                            torch.arange(0, dim, 2).to(torch.float64).div(dim))) #(L. C)
    freqs = torch.polar(torch.ones_like(freqs), freqs) # (L, C)
    return freqs

@amp.autocast(enabled=False)
def rope_params_PI(max_seq_len, dim, theta=10000, scale_factor=2.0):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    pos = torch.arange(max_seq_len)
    pos = pos / scale_factor
    freqs = torch.outer(pos, inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs) # (L, C)
    return freqs

@amp.autocast(enabled=False) 
def rope_params_riflex(max_seq_len, dim, theta=10000, k=5, L_test=30):
    print(k)
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    inv_theta_pow[k-1] =  2 * torch.pi / 80
        
    freqs = torch.outer(torch.arange(max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@amp.autocast(enabled=False)
def rope_params_YARN(max_seq_len, dim, theta=10000, L_prime=41, L=21, beta_fast=1, beta_slow=0.6, extrapolation_factor=1):
    assert dim % 2 == 0
    
    scale = torch.tensor(L_prime/ L)
    freqs_extrapolation = 1.0 / torch.pow(theta ,(torch.arange(0, dim, 2, dtype=torch.float64).div(dim)))
    freqs_interpolation = 1.0 / torch.einsum('..., f -> ... f', scale, (theta ** (torch.arange(0, dim, 2, dtype=torch.float64).div(dim))))
    
    def find_correction_factor(num_rotations, dim, base=10000, max_position_embeddings=2048):
        return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
        low = math.floor(find_correction_factor(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_factor(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim-1) #Clamp values just in case
    
    def linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001 #Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
        
    low, high = find_correction_range(beta_fast, beta_slow, dim, theta, 21)
    freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(scale).float()) * extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
    freqs = freqs_interpolation * (1 - freqs_mask) + freqs_extrapolation * freqs_mask
    
    freqs = torch.outer(torch.arange(max_seq_len), freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs) # (L, C)
    return freqs

@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, shift=0): 
    n, c = x.size(2), x.size(3) // 2 # (B, H, L, C)

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][shift:f+shift].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

@amp.autocast(enabled=False)
def rope_apply_custom_fp(x, grid_sizes, freqs, position_idx): 
    # enable_rp_map: wheter to enable relative position mapping. 
    b, n, c = x.size(0), x.size(2), x.size(3) // 2 # (B, H, L, C)

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    grid_size_list = grid_sizes.tolist()

    output = []
    for i in range(b):
        f, h, w = grid_size_list[i]
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        f_pos = position_idx[i]
        freqs_i = torch.cat([
            freqs[0][f_pos].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 save_attn_map=False,
                 ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.save_attn_map = save_attn_map

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        
    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)


    def forward(self, 
                x, 
                seq_lens, 
                grid_sizes, 
                freqs, 
                block_mask=None, 
                block_mask_kwargs=None,
                rp_map_kwargs=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        if rp_map_kwargs is not None or isinstance(block_mask, list):
            q_shift = q
            k_shift = k
        q=rope_apply(q, grid_sizes, freqs)
        k=rope_apply(k, grid_sizes, freqs)

        
        if block_mask is not None and rp_map_kwargs is None:

            print('implet sparse attention')
            #import pdb;pdb.set_trace()
            if isinstance(block_mask, list):
                block_mask_kwargs = block_mask_kwargs or {}
                group_size = block_mask_kwargs.get("group_size", 8)
                window_size = block_mask_kwargs.get("window_size", 12)
                diag_size_token_per_frame = block_mask_kwargs.get(
                    "diag_size_token_per_frame", 1560)

                grid_size_list = grid_sizes.tolist()
                frame_num_list = [grid_size[0] for grid_size in grid_size_list]
                # 
                frame_pos_list = torch.stack([torch.arange(frame_num, dtype=torch.long) for frame_num in frame_num_list])
                q_frame_pos_list_1 = (frame_pos_list // group_size + window_size - window_size / group_size).to(dtype=torch.long)
                k_frame_pos_list_1 = frame_pos_list // group_size 

                q_frame_pos_list_2 = frame_pos_list // group_size
                k_frame_pos_list_2 = (frame_pos_list // group_size + window_size - window_size / group_size).to(dtype=torch.long)

                q_shift_1 = rope_apply_custom_fp(q_shift, grid_sizes, freqs=freqs, position_idx=q_frame_pos_list_1)
                k_shift_1 = rope_apply_custom_fp(k_shift, grid_sizes, freqs=freqs,position_idx=k_frame_pos_list_1)
                q_shift_2 = rope_apply_custom_fp(q_shift, grid_sizes, freqs=freqs, position_idx=q_frame_pos_list_2)
                k_shift_2 = rope_apply_custom_fp(k_shift, grid_sizes, freqs=freqs,position_idx=k_frame_pos_list_2)

                x_local = radial_attention(
                    neighbor_query_states=q,
                    neighbor_key_states=k,
                    shifted_query_states_1=q_shift_1,
                    shifted_key_states_1=k_shift_1,
                    shifted_query_states_2=q_shift_2,
                    shifted_key_states_2=k_shift_2,
                    value_states=v,
                    diag_size=window_size * diag_size_token_per_frame,
                    block_mask_diag=block_mask[2],
                    block_mask_shift_1=block_mask[0],
                    block_mask_shift_2=block_mask[1]
                )
            else:
                f, h, w = map(int, grid_sizes[0])
                q= q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                q = q.to(v.dtype)
                k = k.to(v.dtype)
                x_local = self.sparse_flex_attention(q, k, v, block_mask=block_mask).permute(0, 2, 1, 3)
                print('no relative position map')
            x_local = x_local.flatten(2)
            x_local = self.o(x_local)

            x = x_local
        
        elif rp_map_kwargs is not None:
            if rp_map_kwargs['relative_map_mod'] == 'vrpr':
                block_mask_3, block_mask_1, block_mask_2, block_mask_4 = rp_map_kwargs['block_masks']
                group_size_1 = rp_map_kwargs['group_size_1']
                group_size_2 = rp_map_kwargs['group_size_2']
                window_size_1 = rp_map_kwargs['window_size_1']
                window_size_2 = rp_map_kwargs['window_size_2']
                grid_size_list = grid_sizes.tolist()
                frame_num_list = [grid_size[0] for grid_size in grid_size_list]
                # 
                frame_pos_list = torch.stack([torch.arange(frame_num, dtype=torch.long) for frame_num in frame_num_list])
                q_frame_pos_list_1 = (frame_pos_list // group_size_1 + window_size_1 - window_size_1 / group_size_1).to(dtype=torch.long)
                k_frame_pos_list_1 = frame_pos_list // group_size_1
                q_frame_pos_list_2 = frame_pos_list // group_size_1
                k_frame_pos_list_2 = (frame_pos_list // group_size_1 + window_size_1 - window_size_1 / group_size_1).to(dtype=torch.long)
                 
                q_frame_pos_list_3 = torch.clip(
                    (frame_pos_list // group_size_2 + window_size_2 -
                     window_size_2 // group_size_2 -
                     (window_size_2 - window_size_1) // group_size_1),
                    max=rp_map_kwargs.get("vrpr_clip_max", 20)).to(
                        dtype=torch.long)
                k_frame_pos_list_3 = frame_pos_list // group_size_2
                q_frame_pos_list_4 = frame_pos_list // group_size_2
                k_frame_pos_list_4 = torch.clip(
                    (frame_pos_list // group_size_2 + window_size_2 -
                     window_size_2 // group_size_2 -
                     (window_size_2 - window_size_1) // group_size_1),
                    max=rp_map_kwargs.get("vrpr_clip_max", 20)).to(
                        dtype=torch.long)

                q_shift_1 = rope_apply_custom_fp(q_shift, grid_sizes, freqs=freqs, position_idx=q_frame_pos_list_1)
                k_shift_1 = rope_apply_custom_fp(k_shift, grid_sizes, freqs=freqs,position_idx=k_frame_pos_list_1)
                q_shift_2 = rope_apply_custom_fp(q_shift, grid_sizes, freqs=freqs, position_idx=q_frame_pos_list_2)
                k_shift_2 = rope_apply_custom_fp(k_shift, grid_sizes, freqs=freqs,position_idx=k_frame_pos_list_2)

                q_shift_3 = rope_apply_custom_fp(q_shift, grid_sizes, freqs=freqs, position_idx=q_frame_pos_list_3)
                k_shift_3 = rope_apply_custom_fp(k_shift, grid_sizes, freqs=freqs,position_idx=k_frame_pos_list_3)
                q_shift_4 = rope_apply_custom_fp(q_shift, grid_sizes, freqs=freqs, position_idx=q_frame_pos_list_4)
                k_shift_4 = rope_apply_custom_fp(k_shift, grid_sizes, freqs=freqs, position_idx=k_frame_pos_list_4)

                x = vrpr_attention(
                    neighbor_query_states=q,
                    neighbor_key_states=k,
                    shifted_query_states_1=q_shift_1,
                    shifted_key_states_1=k_shift_1,
                    shifted_query_states_2=q_shift_2,
                    shifted_key_states_2=k_shift_2,
                    shifted_query_states_3=q_shift_3,
                    shifted_key_states_3=k_shift_3,
                    shifted_query_states_4=q_shift_4,
                    shifted_key_states_4=k_shift_4,
                    value_states=v,
                    diag_size=(window_size_1 - 1) * rp_map_kwargs.get(
                        "diag_size_token_per_frame", 1560),
                    block_mask_1=block_mask_1,
                    block_mask_2=block_mask_2,
                    block_mask_3=block_mask_3,
                    block_mask_4=block_mask_4,
                )
                x = x.flatten(2)
                x = self.o(x)
            else:
                raise ValueError(
                    f"Only relative_map_mod='vrpr' is supported in rp_map_kwargs, got '{rp_map_kwargs['relative_map_mod']}'."
                )
        else:

            x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
            
            x = x.flatten(2)
            x = self.o(x)
        

        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)
    
            
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask=None,
        block_mask_kwargs=None,
        rp_map_kwargs=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        return self.normal_forward(
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            block_mask=block_mask,
            block_mask_kwargs=block_mask_kwargs,
            rp_map_kwargs=rp_map_kwargs,
        )

    def normal_forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask=None,
        block_mask_kwargs=None,
        rp_map_kwargs=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # import pdb;pdb.set_trace()
        # self-attention
        
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens,
            grid_sizes,
            freqs,
            block_mask=block_mask,
            block_mask_kwargs=block_mask_kwargs,
            rp_map_kwargs=rp_map_kwargs,
        )
        # attn_output['self_attn_output'] = self_attn_output
        with amp.autocast(dtype=torch.float32):
            # import pdb;pdb.set_trace()
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            cross_out = self.cross_attn(self.norm3(x), context, context_lens)
            x += cross_out
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel_Freeloc(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
    
        self.freqs_origin = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
                                dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()
    
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        enable_rp_map=False,
        enable_layer_modify=False,
        rope_type=None,
        rope_relative_layers=None,
        relative_map_mod='vrpr',
        use_radial_attention=True,
        runtime_config=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        runtime_config = runtime_config or {}
        model_runtime_config = runtime_config.get("model", runtime_config)
        default_rope_relative_layers = model_runtime_config.get(
            "default_rope_relative_layers",
            [1, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 18, 22, 23, 24, 25])
        radial_attention_cfg = model_runtime_config.get("radial_attention", {})
        rp_map_cfg = model_runtime_config.get("rp_map", {})

        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None

        if enable_layer_modify:
            if rope_relative_layers is None:
                rope_relative_layers = default_rope_relative_layers
        else:
            rope_relative_layers = []
        B = len(x)
        C_in, F, H, W = x[0].shape # VAE latens [C, F, H / 8, W / 8]
            
        # params
        device = self.patch_embedding.weight.device
        # Riflex
        if rope_type=='riflex':
            d = self.dim // self.num_heads
            self.freqs = torch.cat([
                    rope_params_riflex(1024, d - 4 * (d // 6), L_test=F),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6))
                ],
                                dim=1)
        elif rope_type=='ntk':
            d = self.dim // self.num_heads
            self.freqs = torch.cat([
                    rope_params_NTK(1024, d - 4 * (d // 6), L_prime=F, L=21),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6))
                ], 
                                dim=1)
        elif rope_type=='pi':
            d = self.dim // self.num_heads
            self.freqs = torch.cat([
                    rope_params_PI(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6))
                ], 
                                dim=1)
        elif rope_type=='yarn':
            d = self.dim // self.num_heads
            self.freqs = torch.cat([
                    rope_params_YARN(1024, d - 4 * (d // 6), L_prime=F, L=21),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6))
                ], 
                                dim=1)
        else:
            self.freqs = self.freqs_origin
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
            self.freqs_origin =  self.freqs_origin.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)
        
        if use_radial_attention:
            radial_window_size_1 = radial_attention_cfg.get(
                "window_size_1", 8)
            radial_window_size_2 = radial_attention_cfg.get(
                "window_size_2", 32)
            block_mask = prepare_flexattention_radial_attention(
                # B,
                # self.num_heads,
                # self.dim // self.num_heads,
                # dtype=torch.bfloat16,
                device='cuda',
                frame_size=((H // 2) * (W // 2)),
                context_length=0,
                num_frame=F,
                window_size=radial_window_size_1,
                window_size_2=radial_window_size_2,
                enbale_relative_position_map=True)
            rope_shift_cfg = radial_attention_cfg.get("rope_shift", {})
            block_mask_kwargs = {
                "group_size": rope_shift_cfg.get("group_size", 8),
                "window_size": rope_shift_cfg.get("window_size", 12),
                "diag_size_token_per_frame":
                radial_attention_cfg.get("diag_size_token_per_frame", 1560)
            }
        else:
            block_mask = prepare_flexattention(
                # B,
                # self.num_heads,
                # self.dim // self.num_heads,
                # dtype=torch.bfloat16,
                device='cuda',
                frame_size=((H // 2) * (W // 2)),
                context_length=0,
                num_frame=F,
                multiplier=radial_attention_cfg.get("fallback_multiplier",
                                                    12))
            block_mask_kwargs = None

        if enable_rp_map:
            rp_map_diag_factor = rp_map_cfg.get("diag_size_token_per_frame",
                                                1560)
            if relative_map_mod == 'vrpr':
                vrpr_cfg = rp_map_cfg.get("vrpr", {})
                frame_settings = vrpr_cfg.get("frame_settings", {})
                frame_cfg = frame_settings.get(str(F))
                if frame_cfg is None:
                    available_frames = sorted(frame_settings.keys())
                    raise ValueError(
                        f"Missing VRPR frame settings for F={F}. Available: {available_frames}"
                    )
                window_size_1 = frame_cfg["window_size_1"]
                window_size_2 = frame_cfg["window_size_2"]
                group_size_1 = frame_cfg["group_size_1"]
                group_size_2 = frame_cfg["group_size_2"]
                block_masks = prepare_flexattention_VRPR(
                    # B,
                    # self.num_heads,
                    # self.dim // self.num_heads,
                    # dtype=torch.bfloat16,
                    device='cuda',
                    frame_size=((H // 2) * (W // 2)),
                    context_length=0,
                    num_frame=F,
                    window_size=window_size_1,
                    window_size_2=window_size_2)
                rp_map_kwargs = {
                    'block_masks': block_masks,
                    'window_size_1': window_size_1,
                    'window_size_2': window_size_2,
                    'group_size_1': group_size_1,
                    "group_size_2": group_size_2,
                    'relative_map_mod': relative_map_mod,
                    'diag_size_token_per_frame': rp_map_diag_factor,
                    'vrpr_clip_max': rp_map_cfg.get("vrpr_clip_max", 20)
                }
            else:
                raise ValueError(
                    f"Only relative_map_mod='vrpr' is supported, got '{relative_map_mod}'."
                )
        else:
            rp_map_kwargs = None

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=block_mask,
            block_mask_kwargs=block_mask_kwargs,
            rp_map_kwargs=rp_map_kwargs,
        )

        for i, block in enumerate(self.blocks):
            if enable_layer_modify:
                if i in rope_relative_layers:
                    kwargs['block_mask'] = block_mask
                    kwargs['block_mask_kwargs'] = block_mask_kwargs
                    kwargs['freqs'] = self.freqs
                    kwargs['rp_map_kwargs'] = None
                else:
                    kwargs['block_mask'] = None
                    kwargs['block_mask_kwargs'] = None
                    kwargs['freqs'] = self.freqs
                    kwargs['rp_map_kwargs'] = rp_map_kwargs
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
            

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
