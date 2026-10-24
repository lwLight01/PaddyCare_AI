import hashlib
import torch
import torch.nn as nn
from torchvision import models

from config import EMBEDDING_DIM


class PaddyHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(),              
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    def forward_with_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embed(x)
        return self.head(embedding), embedding


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    if not num_classes or num_classes < 1:
        raise ValueError(f"num_classes must be a positive integer (got {num_classes!r})")

    weights = models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b5(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = PaddyHead(in_features, num_classes)

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def set_bn_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def unfreeze_all(model: nn.Module, keep_bn_frozen: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad = True
    if keep_bn_frozen:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for param in module.parameters():
                    param.requires_grad = False


def checkpoint_fingerprint(path: str, length: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def _remap_legacy_head_keys(state_dict: dict) -> dict:
    if not any(k.startswith("classifier.7.") for k in state_dict):
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("classifier."):
            _, idx, rest = key.split(".", 2)
            idx = int(idx)
            if idx < 6:
                key = f"classifier.embed.{idx}.{rest}"
            else:
                key = f"classifier.head.{idx - 6}.{rest}"
        remapped[key] = value
    return remapped


def load_checkpoint(model_path: str, map_location="cpu") -> dict:
    checkpoint = torch.load(model_path, map_location=map_location, weights_only=True)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError(
            f"{model_path} has no 'class_names'. Refusing to guess the label order - "
            "re-train with the current train.py (every checkpoint it saves includes class_names)."
        )
    checkpoint["model_state_dict"] = _remap_legacy_head_keys(checkpoint["model_state_dict"])
    return checkpoint


def load_model(model_path: str, device: torch.device) -> nn.Module:
    checkpoint = load_checkpoint(model_path, map_location=device)
    class_names = list(checkpoint["class_names"])

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    model.class_names = class_names

    epoch = checkpoint.get("epoch", "?")
    val_score = checkpoint.get("val_macro_f1", checkpoint.get("val_acc", "?"))
    print(f"[model] Loaded from {model_path} (epoch {epoch}, val score: {val_score}, classes: {class_names})")
    return model

def forward_with_embedding(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs EfficientNet + PaddyHead and returns (logits, 128-dim embedding). No hooks needed."""
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return model.classifier.forward_with_embedding(x)


def extract_embedding(model: nn.Module, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        return forward_with_embedding(model, input_tensor)


class DualOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return forward_with_embedding(self.model, x)


if __name__ == "__main__":
    m = build_model(num_classes=5)
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Backbone       : EfficientNet-B5")
    print(f"Total params   : {total:,}")
    print(f"Trainable (P1) : {trainable:,}")
    unfreeze_all(m)
    trainable_p2 = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable (P2) : {trainable_p2:,}")

    m.eval()
    dummy = torch.randn(1, 3, 456, 456)
    logits, emb = extract_embedding(m, dummy)
    print(f"Logits shape   : {logits.shape}")
    print(f"Embedding shape: {emb.shape}")

    wrapper = DualOutputWrapper(m)
    wrapper.eval()
    with torch.inference_mode():
        l2, e2 = wrapper(dummy)
    print(f"Wrapper logits : {l2.shape}")
    print(f"Wrapper embed  : {e2.shape}")
