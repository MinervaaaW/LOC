import copy
import json
import os
from typing import Any, Dict, Optional


DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "generate": {
        "force_offload_model": None
    },
    "text2video": {
        "generate": {},
        "model_forward_kwargs": {
            "rope_type": None,
            "enable_rp_map": True,
            "enable_layer_modify": True,
            "relative_map_mod": "vrpr",
            "use_radial_attention": True,
            "rope_relative_layers": None
        },
        "step_overrides": []
    },
    "model": {
        "default_rope_relative_layers": [0, 1, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 18, 22, 23, 24, 25],
        "radial_attention": {
            "window_size_1": 8,
            "window_size_2": 24,
            "fallback_multiplier": 12,
            "rope_shift": {
                "group_size": 8,
                "window_size": 12
            },
            "diag_size_token_per_frame": 1560
        },
        "rp_map": {
            "diag_size_token_per_frame": 1560,
            "vrpr_clip_max": 20,
            "vrpr": {
                "frame_settings": {
                    "41": {
                        "window_size_1": 12,
                        "window_size_2": 20,
                        "group_size_1": 2,
                        "group_size_2": 8
                    },
                    "81": {
                        "window_size_1": 10,
                        "window_size_2": 14,
                        "group_size_1": 2,
                        "group_size_2": 8
                    }
                }
            }
        }
    }
}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_runtime_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    merged_config = copy.deepcopy(DEFAULT_RUNTIME_CONFIG)
     # ====================== 核心修复 ======================
    # 不传参数 / 传空字符串 / 传空白字符 → 直接用默认，不读任何文件
    if config_path is None or config_path.strip() == "":
        return merged_config

    resolved_path = os.path.abspath(config_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Runtime config file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    if not isinstance(user_config, dict):
        raise ValueError(f"Runtime config root must be a JSON object: {resolved_path}")

    return _deep_merge_dict(merged_config, user_config)
