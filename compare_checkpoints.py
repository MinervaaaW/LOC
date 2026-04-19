import argparse
import importlib.util
import os
import sys


def _load_checkpoint_utils():
    module_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "wan",
        "utils",
        "checkpoint_utils.py",
    )
    spec = importlib.util.spec_from_file_location("checkpoint_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


checkpoint_utils = _load_checkpoint_utils()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare a reference Wan diffusion checkpoint with a custom checkpoint."
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help=(
            "Reference checkpoint file or checkpoint directory. If a directory is "
            "provided, the script will try to find diffusion_pytorch_model.safetensors."
        ),
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Candidate checkpoint file to compare, e.g. stage3/model.pt.",
    )
    parser.add_argument(
        "--max_mismatches",
        type=int,
        default=20,
        help="How many shape mismatches to print.",
    )
    parser.add_argument(
        "--max_unique_keys",
        type=int,
        default=20,
        help="How many keys unique to each checkpoint to print.",
    )
    return parser.parse_args()


def _print_summary(label, summary):
    print(f"{label}:")
    print(f"  num_keys: {summary['num_keys']}")
    print(f"  patch_embedding.weight: {summary['patch_embedding_shape']}")
    print(f"  patch_in_channels: {summary['patch_in_channels']}")
    print(f"  has_img_emb: {summary['has_img_emb']}")
    print(f"  inferred_family: {checkpoint_utils.infer_model_family(summary)}")


def _print_key_list(title, keys, limit):
    if not keys:
        print(f"{title}: 0")
        return
    print(f"{title}: {len(keys)}")
    for key in keys[:limit]:
        print(f"  - {key}")
    if len(keys) > limit:
        print(f"  ... ({len(keys) - limit} more)")


def main():
    args = _parse_args()

    reference_path = checkpoint_utils.resolve_diffusion_checkpoint(args.reference)
    candidate_path = checkpoint_utils.resolve_diffusion_checkpoint(args.candidate)

    print(f"reference_path: {reference_path}")
    print(f"candidate_path: {candidate_path}")

    reference_state_dict = checkpoint_utils.load_state_dict_from_checkpoint(
        reference_path
    )
    candidate_state_dict = checkpoint_utils.load_state_dict_from_checkpoint(
        candidate_path
    )

    reference_summary = checkpoint_utils.summarize_state_dict(reference_state_dict)
    candidate_summary = checkpoint_utils.summarize_state_dict(candidate_state_dict)
    comparison = checkpoint_utils.compare_state_dicts(
        reference_state_dict, candidate_state_dict
    )

    print()
    _print_summary("reference", reference_summary)
    _print_summary("candidate", candidate_summary)

    print()
    print("comparison:")
    print(f"  shared_keys: {comparison['shared_key_count']}")
    print(f"  shape_matches: {comparison['shape_match_count']}")
    print(f"  shape_mismatches: {comparison['shape_mismatch_count']}")

    if comparison["shape_mismatches"]:
        print("  mismatch_examples:")
        for item in comparison["shape_mismatches"][: args.max_mismatches]:
            print(
                "  - "
                f"{item['key']}: "
                f"reference{item['reference_shape']} vs candidate{item['candidate_shape']}"
            )

    print()
    _print_key_list(
        "only_in_reference", comparison["only_in_reference"], args.max_unique_keys
    )
    print()
    _print_key_list(
        "only_in_candidate", comparison["only_in_candidate"], args.max_unique_keys
    )

    print()
    patch_ref = reference_summary["patch_in_channels"]
    patch_cand = candidate_summary["patch_in_channels"]
    if patch_ref == 16 and patch_cand == 36:
        print(
            "diagnosis: candidate checkpoint is not a drop-in replacement for FreeLOC t2v."
        )
        print(
            "reason: reference patch input channels are 16, but candidate uses 36, "
            "which usually means an i2v/flf2v-style conditional backbone."
        )
        return 2

    if comparison["shape_mismatch_count"] > 0:
        print(
            "diagnosis: checkpoints share parameter names but are not shape-compatible."
        )
        return 1

    print("diagnosis: checkpoints are shape-compatible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
