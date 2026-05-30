"""
Export a trained model.pt to ONNX for use in the Chrome extension.

Usage:
    python export.py                          # model.pt → ../extension/model.onnx
    python export.py --model model.pt --out ../extension/model.onnx

The exported model:
    Input:  float32 [1, 3, 224, 224]  (ImageNet-normalized)
    Output: float32 [1, N]            (raw logits; apply softmax in extension)
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights, EfficientNet_B0_Weights


def load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    arch = ckpt["arch"]
    classes = ckpt["classes"]
    num_classes = len(classes)

    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--out",   type=Path, default=Path("../extension/model.onnx"))
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model, classes = load_model(args.model)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Also write class map next to the model so the extension JS can import it
    class_map = {i: name for i, name in enumerate(classes)}
    map_path = args.out.parent / "class_map.json"
    map_path.write_text(json.dumps(class_map, indent=2))
    print(f"Classes: {classes}")

    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
        dynamo=False,
    )

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"Exported -> {args.out}  ({size_mb:.1f} MB)")
    print(f"Class map -> {map_path}")


if __name__ == "__main__":
    main()
