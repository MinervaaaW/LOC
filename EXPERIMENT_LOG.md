# Experiment Log

## 2026-04-20

### Direction decision: preserve `t2v`, add a dedicated `i2v` FreeLOC path

- Confirmed the repository should not replace the current FreeLOC `t2v` inference path.
- Chosen direction:
  keep the existing `t2v` FreeLOC path intact and implement a separate `i2v` FreeLOC path for the custom 36-channel checkpoint.
- Reason for the decision:
  the custom checkpoint is structurally `i2v/flf2v`-like rather than a drop-in `t2v` backbone.

### Checkpoint comparison result recorded for implementation planning

- Ran the comparison between the original Wan `t2v-1.3B` diffusion weights and the custom `stage3/model.pt`.
- Result summary:
  - reference `num_keys`: `825`
  - candidate `num_keys`: `983`
  - shared keys: `825`
  - shape matches: `824`
  - shape mismatches: `1`
  - only mismatch: `patch_embedding.weight`
  - reference `patch_embedding.weight`: `(1536, 16, 1, 2, 2)`
  - candidate `patch_embedding.weight`: `(1536, 36, 1, 2, 2)`
  - reference inferred family: `t2v-like`
  - candidate inferred family: `i2v/flf2v-like`
  - candidate contains `img_emb`
  - candidate has `158` extra image-conditioning parameters, including:
    - `cross_attn.k_img.*`
    - `cross_attn.v_img.*`
    - `cross_attn.norm_k_img.*`
- Interpretation:
  the checkpoint is largely compatible at the transformer block level, but requires an `i2v` model definition because the input channel contract and image-conditioning branches are different.

### Detailed implementation plan documentation

- Added a dedicated implementation plan file:
  `I2V_FREELOC_PLAN.md`
- The plan records:
  - goal and scope
  - decision summary
  - phased implementation strategy
  - validation checklist
  - risks
  - open dependencies
  - expected deliverables

### Planned implementation phases

- Phase 1:
  add a new `i2v-1.3B` config derived from the `t2v-1.3B` transformer size.
- Phase 2:
  add a dedicated `WanI2V_Freeloc` pipeline based on the existing `WanI2V` flow and the `WanModel_Freeloc` backbone.
- Phase 3:
  split checkpoint responsibilities so T5/VAE, CLIP, and custom DiT assets can be loaded explicitly.
- Phase 4:
  extend `generate_freeloc.py` with a new explicit task entry, keeping all current `t2v` behavior unchanged.
- Phase 5:
  add `image2video` runtime-config support parallel to the existing `text2video` config.
- Phase 6:
  implement first-pass single-image-conditioned long-video inference.
- Phase 7:
  run loading, smoke-test, and regression validation.

### Constraints and risks recorded before implementation

- The largest dependency risk is CLIP compatibility.
- The FreeLOC runtime config currently targets `text2video`; `i2v` likely needs a separate config section.
- Long-video quality at `161` or `321` frames is not guaranteed by structural compatibility alone and will need dedicated validation.

### Phase 1-5 implementation: first-pass `i2v` FreeLOC integration

- Added a new 1.3B image-to-video config:
  `wan/configs/wan_i2v_1_3B.py`
- Registered the new task in `wan/configs/__init__.py`:
  `i2v-1.3B`
- Set the 1.3B `i2v` config to use:
  - 1.3B transformer dimensions (`dim=1536`, `num_heads=12`, `num_layers=30`)
  - `in_dim=36`
  - CLIP assets required by image-conditioned inference

- Added a new dedicated FreeLOC pipeline:
  `wan/image2video_freeloc.py`
- Implementation choice:
  reuse `WanModel_Freeloc` with `model_type='i2v'` instead of trying to coerce the current `t2v` pipeline to accept the 36-channel checkpoint.
- The new pipeline:
  - loads T5 from `--ckpt_dir`
  - loads VAE from `--ckpt_dir`
  - loads CLIP from `--clip_ckpt_dir`
  - loads the custom 36-channel backbone from `--dit_checkpoint`
  - reuses FreeLOC runtime config overrides and step overrides during sampling

- Updated `wan/__init__.py` to export `WanI2V_Freeloc`.

- Updated `generate_freeloc.py`:
  - added `i2v-1.3B` example prompt and image entry
  - added `--clip_ckpt_dir`
  - required `--dit_checkpoint` for `i2v-1.3B`
  - validated the CLIP checkpoint path for the new task
  - routed `i2v-1.3B` to `WanI2V_Freeloc`
  - kept existing `t2v`, `i2v-14B`, `flf2v`, and `vace` routing unchanged

- Updated `generate.py` to reject `i2v-1.3B` explicitly and direct users to `generate_freeloc.py`, preventing accidental use through the standard non-FreeLOC entry point.

- Extended runtime config support:
  - added `image2video` defaults in `wan/runtime_config_freeloc.py`
  - added `image2video` section in `wan/configs/freeloc_config.json`

- Added `FreeLOC_i2v_inference.sh` as an example command for the new `i2v-1.3B` FreeLOC path.

- Updated `README.md` with a first-pass `i2v-1.3B` FreeLOC inference example and asset-role explanation for:
  - `--ckpt_dir`
  - `--dit_checkpoint`
  - `--clip_ckpt_dir`

### First-pass verification

- `python -m py_compile generate_freeloc.py generate.py wan/image2video_freeloc.py wan/configs/wan_i2v_1_3B.py wan/runtime_config_freeloc.py wan/__init__.py`

### Current status after first-pass implementation

- The codebase now has a dedicated structural path for the custom 36-channel `i2v` checkpoint.
- The existing `t2v` FreeLOC path remains present and separately routed.
- Runtime correctness against the real model assets is not yet verified in a full inference run; the next step is a smoke test on the target environment with real T5/VAE/CLIP/custom-backbone files.

## 2026-04-19

### FreeLOC inference weight loading

- Added `--dit_checkpoint` to `generate_freeloc.py` so FreeLOC can keep loading T5, VAE, and tokenizer assets from the original Wan checkpoint directory while swapping only the DiT backbone weights.
- Updated `wan/text2video_freeloc.py` to support loading an external DiT checkpoint and to reject incompatible checkpoints with clearer shape-mismatch diagnostics instead of failing deep inside `load_state_dict`.
- Updated `FreeLOC_inference.sh` to use `/commondocument/group2/Self-Forcing/wan_models/Wan2.1-T2V-1.3B` as `--ckpt_dir` and `/commondocument/group2/CFmodel/stage3/model.pt` as `--dit_checkpoint`.
- Added `wan/configs/freeloc_config.json` and allowed it in `.gitignore` so the runtime config path used by the FreeLOC examples exists in the repository.

### Checkpoint diagnosis for custom 36-channel model

- Confirmed the custom `stage3/model.pt` is not a drop-in replacement for the current FreeLOC `t2v-1.3B` backbone because its `patch_embedding.weight` expects 36 input channels while the FreeLOC text-to-video backbone expects 16.
- Added `wan/utils/checkpoint_utils.py` to centralize checkpoint loading, state-dict normalization, summary, and compatibility comparison.
- Added `compare_checkpoints.py` to compare the original Wan diffusion checkpoint against a custom checkpoint and report:
  `patch_embedding.weight` shape, inferred model family, shared keys, shape mismatches, and keys unique to each checkpoint.
- Tightened custom checkpoint loading so incompatible `i2v/flf2v`-style backbones fail early with an explicit diagnostic message.

### Verification

- `python -m py_compile compare_checkpoints.py wan/text2video_freeloc.py wan/utils/checkpoint_utils.py`
- `python compare_checkpoints.py --help`
