# FreeLOC I2V Implementation Plan

## Goal

Add a new FreeLOC image-to-video inference path for the custom 36-channel `i2v` backbone while preserving the existing `t2v` inference path unchanged.

The current repository already supports:

- FreeLOC long-video `t2v` inference on top of `WanT2V_Freeloc`
- standard Wan `i2v` inference on top of `WanI2V`

The target is to combine these two capabilities:

- keep the existing FreeLOC long-video runtime behavior
- replace the pure `t2v` backbone with the custom `i2v` backbone that requires:
  - `patch_embedding.in_channels = 36`
  - `img_emb`
  - image-conditioned cross-attention weights such as `k_img`, `v_img`, and `norm_k_img`

## Decision Summary

The custom checkpoint is not a drop-in replacement for the current FreeLOC `t2v` backbone.

Checkpoint comparison result:

- reference model family: `t2v-like`
- candidate model family: `i2v/flf2v-like`
- shared keys: `825`
- shape matches: `824`
- shape mismatches: `1`
- only mismatch: `patch_embedding.weight`
- reference `patch_embedding.weight`: `(1536, 16, 1, 2, 2)`
- candidate `patch_embedding.weight`: `(1536, 36, 1, 2, 2)`
- only-in-candidate parameters: `158`, mainly image-conditioning modules such as:
  - `img_emb`
  - `cross_attn.k_img`
  - `cross_attn.v_img`
  - `cross_attn.norm_k_img`

Conclusion:

- the custom model should be integrated as a dedicated `i2v` FreeLOC pipeline
- the current `t2v` FreeLOC path should remain intact
- direct loading into the current `t2v` FreeLOC model is structurally incorrect

## Scope

### In scope

- add a new `i2v-1.3B-freeloc` inference task
- keep current `t2v-1.3B` FreeLOC behavior unchanged
- support custom `--dit_checkpoint` for the new `i2v` FreeLOC path
- support runtime config for `i2v` FreeLOC
- support single-image-conditioned long-video generation as the first implementation target

### Out of scope for first pass

- auto-regressive multi-segment conditioning between chunks
- first-last-frame `flf2v` FreeLOC adaptation
- training code changes
- converting the 36-channel checkpoint into a 16-channel `t2v` checkpoint

## Implementation Strategy

### Phase 1: Add an `i2v-1.3B` config

Files:

- `wan/configs/wan_i2v_1_3B.py`
- `wan/configs/__init__.py`

Tasks:

- create a new `i2v_1_3B` config based on the `t2v_1_3B` transformer size
- set:
  - `dim = 1536`
  - `num_heads = 12`
  - `num_layers = 30`
  - `patch_size = (1, 2, 2)`
  - `vae_checkpoint = 'Wan2.1_VAE.pth'`
  - `t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'`
- add CLIP-related fields required by the image-conditioned pipeline
- register a new task in `WAN_CONFIGS`
- add supported sizes for the new task

Expected outcome:

- repository can resolve a 1.3B `i2v` task configuration without affecting existing `t2v` tasks

### Phase 2: Add a dedicated `WanI2V_Freeloc` pipeline

Files:

- `wan/image2video_freeloc.py`
- `wan/__init__.py`

Tasks:

- implement a new class `WanI2V_Freeloc`
- use `wan/image2video.py` as the base pipeline skeleton
- use `wan/modules/model_freeloc.py` as the backbone model implementation
- support:
  - `runtime_config`
  - `dit_checkpoint`
  - `t5_cpu`
  - `offload_model`
  - distributed path parity with the current FreeLOC implementation where practical

Important design choice:

- do not call `WanModel_Freeloc.from_pretrained(checkpoint_dir)` for the new `i2v` path if the source directory is a `t2v` checkpoint directory
- instead, instantiate the model with the correct `i2v` structure and then load the custom checkpoint

Reason:

- the base `t2v` directory defines a 16-channel model
- the target `i2v` checkpoint requires a 36-channel model definition

Expected outcome:

- the repository has a dedicated FreeLOC-compatible `i2v` pipeline class

### Phase 3: Split backbone assets from auxiliary assets

Files:

- `generate_freeloc.py`
- potentially `wan/image2video_freeloc.py`

Tasks:

- keep `--ckpt_dir` for T5 and VAE assets
- keep `--dit_checkpoint` for the custom `i2v` backbone
- add a new argument such as `--clip_ckpt_dir` to specify where CLIP assets come from

Reason:

- the current `Wan2.1-T2V-1.3B` directory does not necessarily contain the CLIP weights required by the `i2v` pipeline
- the custom `i2v` backbone is only valid if the associated CLIP encoder setup is also available

Expected outcome:

- the `i2v` FreeLOC path can load all required assets explicitly instead of relying on an implicit checkpoint layout

### Phase 4: Extend `generate_freeloc.py` with a new task path

Files:

- `generate_freeloc.py`

Tasks:

- add a new task name, recommended: `i2v-1.3B-freeloc`
- route that task to `WanI2V_Freeloc`
- preserve current behavior for:
  - `t2v-1.3B`
  - `t2v-14B`
  - `i2v-14B`
  - `flf2v-14B`
  - `vace-*`
- reuse the existing prompt extension path for image-conditioned tasks

Expected outcome:

- users can explicitly select the new FreeLOC `i2v` pipeline without changing any current `t2v` command

### Phase 5: Add `i2v` runtime config support

Files:

- `wan/runtime_config_freeloc.py`
- `wan/configs/freeloc_config.json`

Tasks:

- add a new `image2video` section parallel to `text2video`
- allow:
  - `model_forward_kwargs`
  - `step_overrides`
- define fallback behavior:
  - if `image2video` is missing, optionally reuse `text2video`
  - otherwise use dedicated `image2video` settings

Reason:

- the best FreeLOC/TSA/VRPR settings for `t2v` may not be optimal for `i2v`
- splitting config now avoids later compatibility problems

Expected outcome:

- FreeLOC runtime control becomes task-specific instead of hardcoded to text-to-video only

### Phase 6: Implement first-pass `i2v` conditioning logic

Files:

- `wan/image2video_freeloc.py`

Tasks:

- copy the conditioning flow from `wan/image2video.py`
- keep the following core steps:
  - encode prompt and negative prompt with T5
  - encode conditioning image with CLIP to obtain `clip_fea`
  - encode the conditioning image into VAE latent space
  - build `msk`
  - concatenate `msk` and latent condition into `y`
  - call the FreeLOC backbone with:
    - `context`
    - `context_null`
    - `clip_fea`
    - `y`
    - `seq_len`
    - FreeLOC runtime kwargs

Expected outcome:

- first-pass `i2v` FreeLOC generation path matches the structural requirements of the 36-channel checkpoint

### Phase 7: Validation and regression checks

Tasks:

- smoke test model construction and checkpoint loading
- smoke test inference on:
  - `832*480`
  - `frame_num = 81`
  - minimal sampling steps
- verify that the generated pipeline saves video output
- verify that current `t2v` FreeLOC commands still run unchanged

Expected outcome:

- new path is usable
- old path is preserved

## Verification Checklist

### Loading checks

- `i2v-1.3B-freeloc` constructs without shape mismatch
- `patch_embedding.weight` loads as 36-channel
- `img_emb` loads successfully
- `cross_attn.k_img`, `cross_attn.v_img`, and related parameters load successfully

### Inference checks

- image-conditioned generation produces a valid tensor and output video
- runtime config settings are actually applied
- CPU offload path still works

### Regression checks

- `t2v-1.3B` FreeLOC remains unchanged
- current custom-checkpoint diagnostics for `t2v` still fail fast on incompatible weights
- `compare_checkpoints.py` remains usable

## Risks

### CLIP dependency risk

The custom `i2v` model depends on a compatible CLIP encoder path. If the wrong CLIP weights are loaded, the pipeline may run but quality may be significantly degraded.

### Runtime-config mismatch risk

Current FreeLOC runtime settings were tuned for `text2video`. `i2v` may require different `rope_relative_layers`, TSA settings, or VRPR windows.

### Long-video quality risk

Even after the pipeline loads successfully, quality and temporal consistency at `161` or `321` frames are not guaranteed. First-pass integration should focus on correctness and runnability before optimization.

## Open Dependencies

- confirm which CLIP checkpoint and tokenizer were used during training of the custom `stage3/model.pt`
- confirm whether the first implementation should support only a single conditioning image or also chunk-to-chunk conditioning for very long generation
- confirm whether the target task naming should be `i2v-1.3B-freeloc` or another repository-specific convention

## Deliverables

- new `i2v` config for 1.3B
- new `WanI2V_Freeloc` pipeline
- updated `generate_freeloc.py` entry path
- runtime config support for `image2video`
- inference example script for the new task
- updated documentation
- experiment log updates for every implementation step
