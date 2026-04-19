# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import copy
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model_freeloc import WanModel_Freeloc
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .text2video_freeloc import WanT2V_Freeloc
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanI2V_Freeloc(WanT2V_Freeloc):

    def __init__(
        self,
        config,
        checkpoint_dir,
        dit_checkpoint,
        clip_checkpoint_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        runtime_config=None,
        init_on_cpu=True,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.runtime_config = runtime_config or {}

        clip_checkpoint_dir = clip_checkpoint_dir or checkpoint_dir
        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(clip_checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(clip_checkpoint_dir,
                                        config.clip_tokenizer))

        logging.info("Creating WanI2V_Freeloc model.")
        self.model = WanModel_Freeloc(
            model_type='i2v',
            patch_size=config.patch_size,
            text_len=config.text_len,
            in_dim=getattr(config, "in_dim", 36),
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=getattr(config, "text_dim", 4096),
            out_dim=getattr(config, "out_dim", self.vae.model.z_dim),
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=config.window_size,
            qk_norm=config.qk_norm,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
        )

        if dit_checkpoint is None:
            raise ValueError(
                "WanI2V_Freeloc requires --dit_checkpoint because the custom i2v backbone "
                "cannot be constructed from the base t2v checkpoint directory alone."
            )
        self._load_custom_dit_checkpoint(dit_checkpoint)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 rope_relative_layers=None):
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        i2v_runtime_cfg = self.runtime_config.get(
            "image2video", self.runtime_config.get("text2video", {}))
        base_model_forward_kwargs = copy.deepcopy(
            i2v_runtime_cfg.get("model_forward_kwargs", {}))
        step_overrides = i2v_runtime_cfg.get("step_overrides", [])
        self._validate_relative_map_mod_settings(base_model_forward_kwargs,
                                                 step_overrides)

        if rope_relative_layers is not None:
            base_model_forward_kwargs["rope_relative_layers"] = rope_relative_layers
        if base_model_forward_kwargs.get("rope_relative_layers") is None:
            base_model_forward_kwargs.pop("rope_relative_layers", None)
        base_model_forward_kwargs["runtime_config"] = self.runtime_config

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        latent = torch.randn(
            self.vae.model.z_dim, (F - 1) // 4 + 1, lat_h, lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [tensor.to(self.device) for tensor in context]
            context_null = [tensor.to(self.device) for tensor in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        condition = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        condition = torch.concat([msk, condition])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [condition],
            }
            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [condition],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for idx, t in enumerate(tqdm(timesteps)):
                step_model_forward_kwargs = self._apply_step_overrides(
                    base_model_forward_kwargs, step_overrides, idx)
                arg_c.update(step_model_forward_kwargs)
                arg_null.update(step_model_forward_kwargs)

                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

            x0 = [latent.to(self.device)]
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        return videos[0]
