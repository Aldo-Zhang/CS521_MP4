import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  # (may be needed by resnet_torch)
import torchvision
import torchvision.transforms as transforms

from torch.ao.quantization import QuantStub, DeQuantStub, prepare, convert, QConfig
from torch.ao.quantization.observer import MinMaxObserver

# ---------------------------------------------------------------------
# Paths: point to the folder that contains resnet_torch.py + weights
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSIGN_DIR = os.path.join(PROJECT_ROOT, "CS521-Spring2024-rsmooth-assignment")
sys.path.append(ASSIGN_DIR)

from resnet_torch import ResNet18  # noqa: E402


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def print_size_of_model(model: nn.Module, label: str = "model") -> float:
    """Save state_dict to disk and report size in MB."""
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    print(f"{label} size: {size_mb:.2f} MB")
    return size_mb


def test_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total


def measure_inference_time(model: nn.Module, loader, device: torch.device,
                           num_batches: int = 50) -> float:
    """Average inference time in ms over num_batches batches."""
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            _ = model(images)
            if i >= num_batches:
                break
    end = time.time()
    return (end - start) * 1000.0 / num_batches


# ---------------------------------------------------------------------
# Quantization wrapper
# ---------------------------------------------------------------------
class QuantizedResNet18(nn.Module):
    """Wrap the trained float ResNet18 with Quant/DeQuant stubs."""

    def __init__(self, float_model: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.model = float_model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true",
                        help="Use cuda for FP32 evaluation if available.")
    parser.add_argument("--data-root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data"),
                        help="Directory for CIFAR10.")
    parser.add_argument("--calib-batches", type=int, default=20,
                        help="Batches for calibration.")
    parser.add_argument("--time-batches", type=int, default=50,
                        help="Batches for timing.")
    args = parser.parse_args()

    # 1) Device for FP32 evaluation
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device for FP32 evaluation: {device}")

    # 2) CIFAR10 data (assignment mean/std)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    # 3) Load ResNet18 model + weights
    model_fp32 = ResNet18()
    ckpt_path = os.path.join(ASSIGN_DIR, "resnet18_cifar10.pth")
    print(f"Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_fp32.load_state_dict(state_dict)
    model_fp32 = model_fp32.to(device).eval()

    # ---------------- FP32 evaluation ----------------
    print("\n=== Float (FP32) model evaluation ===")
    # size must be measured on CPU
    fp32_size = print_size_of_model(model_fp32.to("cpu"), "FP32 ResNet18")
    model_fp32 = model_fp32.to(device)

    fp32_acc = test_accuracy(model_fp32, testloader, device)
    fp32_time = measure_inference_time(
        model_fp32, testloader, device, num_batches=args.time_batches
    )
    print(f"FP32 Test Accuracy: {fp32_acc:.2f}%")
    print(f"FP32 Inference Time: {fp32_time:.2f} ms "
          f"(avg over {args.time_batches} batches)")

    # ---------------- Quantization ----------------
    print("\n=== Building quantized model (post-training static) ===")

    # quantization must run on CPU
    model_fp32_cpu = model_fp32.to("cpu").eval()
    qmodel = QuantizedResNet18(model_fp32_cpu).eval()

    torch.backends.quantized.engine = "fbgemm"
    per_tensor_qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )
    qmodel.qconfig = per_tensor_qconfig

    print("Preparing model for static quantization...")
    qmodel_prepared = prepare(qmodel, inplace=False).eval()

    # calibration
    print(f"Calibrating with {args.calib_batches} batches from training set...")
    with torch.no_grad():
        for i, (images, _) in enumerate(trainloader):
            qmodel_prepared(images)  # images stay on CPU
            if i >= args.calib_batches:
                break
    print("Calibration done.")

    print("Converting model to INT8...")
    qmodel_int8 = convert(qmodel_prepared, inplace=False).eval()
    print("Quantization complete!")

    # ---------------- INT8 evaluation ----------------
    int8_size = print_size_of_model(qmodel_int8, "INT8 ResNet18")

    cpu = torch.device("cpu")
    quant_acc = test_accuracy(qmodel_int8, testloader, device=cpu)
    quant_time = measure_inference_time(
        qmodel_int8, testloader, device=cpu, num_batches=args.time_batches
    )

    print(f"INT8 Test Accuracy (CPU): {quant_acc:.2f}%")
    print(f"INT8 Inference Time (CPU): {quant_time:.2f} ms "
          f"(avg over {args.time_batches} batches)")

    # ---------------- Summary ----------------
    print("\n=== Summary ===")
    print(f"FP32 size      : {fp32_size:.2f} MB")
    print(f"INT8 size      : {int8_size:.2f} MB")
    print(f"FP32 accuracy  : {fp32_acc:.2f}%")
    print(f"INT8 accuracy  : {quant_acc:.2f}%")
    print(f"FP32 time      : {fp32_time:.2f} ms")
    print(f"INT8 time      : {quant_time:.2f} ms")


if __name__ == "__main__":
    main()