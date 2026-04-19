# Experiment Log

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
