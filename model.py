import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES


def build_model(num_classes: int = None, pretrained: bool = True) -> nn.Module:
    """Builds an EfficientNet-B5 model with custom classifier head."""
    if num_classes is None:
        num_classes = NUM_CLASSES

    weights = models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b5(weights=weights)

    # ── Phase 1: freeze the entire feature extractor ──
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),                       # SiLU (Swish) matches EfficientNet's design language
        nn.Dropout(p=0.3),
        nn.Linear(512, 128),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )

    # Classifier head is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_all(model: nn.Module) -> None:
    """Unfreezes all layers for Phase 2 fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Loads model checkpoint and dynamically sizes the classifier head
    to match the number of classes saved in the checkpoint.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    if "class_names" in checkpoint:
        num_classes = len(checkpoint["class_names"])
    elif "classifier.7.weight" in state_dict:
        num_classes = state_dict["classifier.7.weight"].shape[0]
    else:
        num_classes = NUM_CLASSES

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_score = checkpoint.get("val_macro_f1", checkpoint.get("val_acc", "?"))
    classes = checkpoint.get("class_names", f"{num_classes} classes")
    print(f"[model] Loaded from {model_path} (epoch {epoch}, val score: {val_score}, classes: {classes})")
    return model


# ── Embedding extraction (for Mahalanobis OOD detection) ─────────────────────

def extract_embedding(model: nn.Module, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts both logits and the 128-dim penultimate embedding."""
    embedding_output = {}

    def hook_fn(module, input, output):
        embedding_output["embedding"] = output

    # Register hook on the SiLU after Linear(512, 128)
    handle = model.classifier[5].register_forward_hook(hook_fn)

    with torch.no_grad():
        logits = model(input_tensor)

    handle.remove()
    return logits, embedding_output["embedding"]


class DualOutputWrapper(nn.Module):
    """Wraps model to output (logits, embedding) simultaneously for ONNX / TFLite export."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.features = model.features
        self.avgpool  = model.avgpool
        # classifier[0:6] → everything up to and including the 128-dim SiLU
        self.embed_layers = model.classifier[:6]
        # classifier[6:8] → Dropout + final Linear(128, num_classes)
        self.head_layers  = model.classifier[6:]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embedding = self.embed_layers(x)
        logits    = self.head_layers(embedding)
        return logits, embedding


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

    # Test embedding extraction
    dummy = torch.randn(1, 3, 456, 456)
    logits, emb = extract_embedding(m, dummy)
    print(f"Logits shape   : {logits.shape}")
    print(f"Embedding shape: {emb.shape}")

    # Test dual-output wrapper
    wrapper = DualOutputWrapper(m)
    wrapper.eval()
    l2, e2 = wrapper(dummy)
    print(f"Wrapper logits : {l2.shape}")
    print(f"Wrapper embed  : {e2.shape}")
