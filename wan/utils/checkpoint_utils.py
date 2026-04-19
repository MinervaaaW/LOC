import os

import torch
from safetensors.torch import load_file as safe_load_file


def looks_like_state_dict(obj):
    return isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values())


def extract_state_dict(checkpoint):
    if looks_like_state_dict(checkpoint):
        return checkpoint

    preferred_keys = (
        "state_dict",
        "model",
        "module",
        "model_state_dict",
        "net",
        "generator",
        "ema",
        "ema_state_dict",
    )
    for key in preferred_keys:
        value = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if looks_like_state_dict(value):
            return value

    if isinstance(checkpoint, dict):
        nested_candidates = [
            value for value in checkpoint.values() if looks_like_state_dict(value)
        ]
        if nested_candidates:
            nested_candidates.sort(key=lambda item: len(item), reverse=True)
            return nested_candidates[0]

    raise ValueError("Unable to find a usable state_dict in the checkpoint.")


def normalize_state_dict_key(key):
    prefixes = (
        "module.",
        "_orig_mod.",
        "model.",
        "net.",
        "generator.",
        "ema_model.",
        "ema.",
        "diffusion_model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def normalize_state_dict(state_dict):
    return {normalize_state_dict_key(key): value for key, value in state_dict.items()}


def load_checkpoint_file(checkpoint_path):
    if checkpoint_path.endswith(".safetensors"):
        return safe_load_file(checkpoint_path)
    return torch.load(checkpoint_path, map_location="cpu")


def load_state_dict_from_checkpoint(checkpoint_path):
    raw_checkpoint = load_checkpoint_file(checkpoint_path)
    return normalize_state_dict(extract_state_dict(raw_checkpoint))


def resolve_diffusion_checkpoint(path_or_dir):
    if os.path.isfile(path_or_dir):
        return os.path.abspath(path_or_dir)

    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"Checkpoint path not found: {path_or_dir}")

    candidates = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
        "model.pt",
    )
    for filename in candidates:
        candidate_path = os.path.join(path_or_dir, filename)
        if os.path.isfile(candidate_path):
            return os.path.abspath(candidate_path)

    raise FileNotFoundError(
        f"Unable to find a diffusion checkpoint under directory: {path_or_dir}"
    )


def summarize_state_dict(state_dict):
    patch_weight = state_dict.get("patch_embedding.weight")
    patch_shape = tuple(patch_weight.shape) if patch_weight is not None else None
    patch_in_channels = patch_shape[1] if patch_shape is not None else None
    has_img_emb = any(key.startswith("img_emb.") for key in state_dict)
    return {
        "num_keys": len(state_dict),
        "patch_embedding_shape": patch_shape,
        "patch_in_channels": patch_in_channels,
        "has_img_emb": has_img_emb,
    }


def infer_model_family(summary):
    patch_in_channels = summary.get("patch_in_channels")
    has_img_emb = summary.get("has_img_emb")

    if patch_in_channels == 16 and not has_img_emb:
        return "t2v-like"
    if patch_in_channels == 36 and has_img_emb:
        return "i2v/flf2v-like"
    if patch_in_channels == 36:
        return "conditional-video-like"
    if patch_in_channels == 16:
        return "t2v-like (with extra modules)"
    return "unknown"


def compare_state_dicts(reference_state_dict, candidate_state_dict):
    reference_keys = set(reference_state_dict.keys())
    candidate_keys = set(candidate_state_dict.keys())
    shared_keys = sorted(reference_keys & candidate_keys)

    shape_matches = []
    shape_mismatches = []
    for key in shared_keys:
        reference_shape = tuple(reference_state_dict[key].shape)
        candidate_shape = tuple(candidate_state_dict[key].shape)
        if reference_shape == candidate_shape:
            shape_matches.append(key)
        else:
            shape_mismatches.append({
                "key": key,
                "reference_shape": reference_shape,
                "candidate_shape": candidate_shape,
            })

    return {
        "reference_key_count": len(reference_keys),
        "candidate_key_count": len(candidate_keys),
        "shared_key_count": len(shared_keys),
        "shape_match_count": len(shape_matches),
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatches": shape_mismatches,
        "only_in_reference": sorted(reference_keys - candidate_keys),
        "only_in_candidate": sorted(candidate_keys - reference_keys),
    }
