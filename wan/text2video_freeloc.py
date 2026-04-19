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

from numpy import False_
from sympy import N
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model_freeloc import WanModel_Freeloc
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.checkpoint_utils import (
    compare_state_dicts,
    infer_model_family,
    load_state_dict_from_checkpoint,
    summarize_state_dict,
)
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

class WanT2V_Freeloc:

    def __init__(
        self,
        config,
        checkpoint_dir,
        dit_checkpoint=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        runtime_config=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            dit_checkpoint (`str`, *optional*, defaults to None):
                Optional checkpoint file for overriding the DiT weights while
                still using T5/VAE/tokenizer assets from `checkpoint_dir`.
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.runtime_config = runtime_config or {}

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel_Freeloc.from_pretrained(checkpoint_dir)
        if dit_checkpoint is not None:
            self._load_custom_dit_checkpoint(dit_checkpoint)
        self.model.eval().requires_grad_(False)

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
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def _load_custom_dit_checkpoint(self, checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
        logging.info(f"Loading custom DiT checkpoint from {checkpoint_path}")

        candidate_state_dict = load_state_dict_from_checkpoint(checkpoint_path)
        target_state_dict = self.model.state_dict()
        comparison = compare_state_dicts(target_state_dict, candidate_state_dict)

        if comparison["shared_key_count"] == 0:
            raise ValueError(
                f"No matching DiT weights were found in custom checkpoint: {checkpoint_path}"
            )

        if comparison["shape_mismatch_count"] > 0:
            candidate_summary = summarize_state_dict(candidate_state_dict)
            target_summary = summarize_state_dict(target_state_dict)
            sample_mismatches = comparison["shape_mismatches"][:5]
            mismatch_lines = [
                f"{item['key']}: checkpoint{item['candidate_shape']} vs model{item['reference_shape']}"
                for item in sample_mismatches
            ]
            hint = ""
            if (
                candidate_summary["patch_in_channels"] == 36
                and target_summary["patch_in_channels"] == 16
            ):
                hint = (
                    " The checkpoint looks like an i2v/flf2v-style conditional backbone "
                    "with 36 input channels, while the current FreeLOC t2v backbone expects 16."
                )
            raise ValueError(
                "Custom DiT checkpoint is not shape-compatible with the current FreeLOC backbone. "
                f"Shared keys: {comparison['shared_key_count']}, "
                f"shape mismatches: {comparison['shape_mismatch_count']}. "
                f"Checkpoint family: {infer_model_family(candidate_summary)}; "
                f"target family: {infer_model_family(target_summary)}. "
                f"Examples: {'; '.join(mismatch_lines)}.{hint}"
            )

        filtered_state_dict = {
            key: candidate_state_dict[key]
            for key in target_state_dict.keys()
            if key in candidate_state_dict
        }

        missing_keys, unexpected_keys = self.model.load_state_dict(
            filtered_state_dict, strict=False)
        loaded_count = len(target_state_dict) - len(missing_keys)
        logging.info(
            f"Loaded {loaded_count}/{len(target_state_dict)} DiT parameters from custom checkpoint."
        )

        if missing_keys:
            logging.warning(
                f"Custom DiT checkpoint is missing {len(missing_keys)} parameters. "
                f"First few: {missing_keys[:10]}"
            )
        if unexpected_keys:
            logging.warning(
                f"Custom DiT checkpoint has {len(unexpected_keys)} unexpected parameters. "
                f"First few: {unexpected_keys[:10]}"
            )

    @staticmethod
    def _apply_step_overrides(base_kwargs, step_overrides, step_idx):
        step_kwargs = copy.deepcopy(base_kwargs)
        for override in step_overrides:
            start_step = override.get("start_step", 0)
            end_step = override.get("end_step", start_step)
            if start_step <= step_idx <= end_step:
                step_kwargs.update(override.get("kwargs", {}))
        return step_kwargs

    @staticmethod
    def _validate_relative_map_mod_settings(base_kwargs, step_overrides):
        supported_mod = "vrpr"
        base_mod = base_kwargs.get("relative_map_mod")
        if base_mod is not None and base_mod != supported_mod:
            raise ValueError(
                f"Only relative_map_mod='{supported_mod}' is supported, got '{base_mod}'."
            )

        for override in step_overrides:
            override_kwargs = override.get("kwargs", {})
            override_mod = override_kwargs.get("relative_map_mod")
            if override_mod is not None and override_mod != supported_mod:
                raise ValueError(
                    f"Only relative_map_mod='{supported_mod}' is supported in step_overrides, got '{override_mod}'."
                )

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 rope_relative_layers=None,
                 ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        t2v_runtime_cfg = self.runtime_config.get("text2video", {})
        base_model_forward_kwargs = copy.deepcopy(
            t2v_runtime_cfg.get("model_forward_kwargs", {}))
        step_overrides = t2v_runtime_cfg.get("step_overrides", [])
        self._validate_relative_map_mod_settings(base_model_forward_kwargs,
                                                 step_overrides)

        if rope_relative_layers is not None:
            base_model_forward_kwargs["rope_relative_layers"] = rope_relative_layers
        if base_model_forward_kwargs.get("rope_relative_layers") is None:
            base_model_forward_kwargs.pop("rope_relative_layers", None)
        base_model_forward_kwargs["runtime_config"] = self.runtime_config

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
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

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            
            for idx, t in enumerate(tqdm(timesteps)):
                step_model_forward_kwargs = self._apply_step_overrides(
                    base_model_forward_kwargs, step_overrides, idx)
                arg_c.update(step_model_forward_kwargs)
                arg_null.update(step_model_forward_kwargs)


                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        return videos[0]
