# Finds the Linear-layer module names in a model, for use as
# peft.LoraConfig(target_modules=...).
#
# It builds the model from its config only (AutoModelForCausalLM.from_config),
# not from pretrained weights. This is fast and needs no checkpoint download,
# because module names come from the architecture, not from the weights.
#
# Usage:
#   python find_lora_target_modules.py Qwen/Qwen3.5-2B
#   python find_lora_target_modules.py Qwen/Qwen2.5-1.5B

import sys
import transformers
import torch.nn as nn
from collections import defaultdict
from transformers import AutoConfig

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"


def find_linear_module_names(model_id):
    """
    Returns a dict: {module_class_name: sorted set of Linear-layer name suffixes}.

    The suffix (last path component, e.g. "q_proj") is what peft matches
    against in LoraConfig.target_modules. The parent class name shows which
    layer type each suffix belongs to -- useful for hybrid models, where
    different layer types use different projection names.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Build the exact class named in config.architectures, rather than going
    # through AutoModelForCausalLM. For composite/multimodal configs (e.g.
    # Qwen3.5's vision+text config), the CausalLM auto-mapping can resolve to
    # a text-only model class that doesn't match the config's shape (it
    # expects a flat config with e.g. `vocab_size`, but gets the composite
    # one with `text_config`/`vision_config`), causing an AttributeError.
    # Going by config.architectures instantiates the model the config
    # actually describes.
    model_cls = getattr(transformers, config.architectures[0])
    model = model_cls._from_config(config)

    by_parent_class = defaultdict(set)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            suffix = name.rsplit(".", 1)[-1]
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            by_parent_class[type(parent).__name__].add(suffix)

    return by_parent_class


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
    print(f"Loading config and empty model for: {model_id}\n")

    by_parent_class = find_linear_module_names(model_id)

    all_suffixes = set()
    for parent_class, suffixes in sorted(by_parent_class.items()):
        print(f"{parent_class}:")
        for suffix in sorted(suffixes):
            print(f"  - {suffix}")
        all_suffixes |= suffixes

    print("\nAll Linear-layer suffixes found (includes MLP projections):")
    print(sorted(all_suffixes))

    print(
        "\nNote: this list includes every Linear layer, e.g. also lm_head and "
        "MLP projections (gate_proj/up_proj/down_proj). Pick the subset that "
        "matches your intended target_modules -- e.g. attention-only, or "
        "attention + MLP -- from the per-class breakdown above."
    )


if __name__ == "__main__":
    main()