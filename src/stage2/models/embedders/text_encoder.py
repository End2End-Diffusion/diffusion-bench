from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_layers(spec, num_hidden_states):
    idx = list(range(num_hidden_states))
    spec = str(spec).strip()
    if spec == "all":
        sel = idx[1:]
    elif ":" in spec:
        sel = idx[slice(*(int(p) if p.strip() else None for p in spec.split(":")))]
    else:
        sel = [idx[int(i)] for i in spec.split(",")]
    if not sel:
        raise ValueError(f"layers spec '{spec}' selected no layers.")
    return sel


class TextEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_length=256,
                 layers="-1", feature_norm="raw", system_prompt=""):
        super().__init__()
        if feature_norm not in ("raw", "ln"):
            raise ValueError(f"feature_norm must be 'raw' or 'ln', got '{feature_norm}'.")
        self.model_name = model_name
        self.max_length = max_length
        self.feature_norm = feature_norm
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(model_name).eval()

        self.feature_dim = self.text_model.config.hidden_size
        self.layer_indices = select_layers(layers, self.text_model.config.num_hidden_layers + 1)

    @torch.no_grad()
    def forward(self, texts: List[str]):
        device = next(self.text_model.parameters()).device

        if self.system_prompt:
            texts = [self.system_prompt + t for t in texts]

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.text_model(
            **tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=False,
        )
        hs = torch.stack([outputs[1][i] for i in self.layer_indices], dim=2)  # [B, T, L, D]
        if self.feature_norm == "ln":
            hs = F.layer_norm(hs, (hs.shape[-1],))
        return {
            "tokens": hs.mean(dim=2),  # average the selected layers
            "attention_mask": tokens["attention_mask"],
        }
