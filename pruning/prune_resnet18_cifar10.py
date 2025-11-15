#!/usr/bin/env python
import os
import sys
import copy
import argparse
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths: folder with resnet_torch.py + weights
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSIGN_DIR = os.path.join(PROJECT_ROOT, "CS521-Spring2024-rsmooth-assignment")
sys.path.append(ASSIGN_DIR)

from resnet_torch import ResNet18  # noqa: E402


# ---------------------------------------------------------------------
# Data + evaluation helpers
# ---------------------------------------------------------------------
def get_cifar10_loaders(data_root: str, batch_size: int = 128,
                        num_workers: int = 2,
                        test_subset: int = None):
    """Return train/test loaders. Optionally restrict test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    if test_subset is not None and test_subset < len(testset):
        testset = torch.utils.data.Subset(testset,
                                          list(range(test_subset)))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return trainloader, testloader


def evaluate_accuracy(model: nn.Module, loader, device: torch.device,
                      max_batches: int = None) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            if max_batches is not None and i + 1 >= max_batches:
                break
    return 100.0 * correct / total


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_resnet18(device: torch.device) -> nn.Module:
    """Load pretrained ResNet18 for CIFAR-10."""
    model = ResNet18().to(device)
    ckpt_path = os.path.join(ASSIGN_DIR, "resnet18_cifar10.pth")
    print(f"Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Part 1: list conv / linear layers and parameter counts
# ---------------------------------------------------------------------
def get_conv_and_linear_layers(model: nn.Module
                               ) -> List[Tuple[str, nn.Module]]:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers


def report_layer_params(model: nn.Module) -> None:
    layers = get_conv_and_linear_layers(model)
    print("\n=== Convolutional and Linear layers in ResNet18 ===")
    for name, module in layers:
        w = module.weight
        n_params = w.numel()
        layer_type = type(module).__name__
        print(f"{name:40s} ({layer_type:7s}) -> "
              f"weight shape {tuple(w.shape)}, #params = {n_params}")


# ---------------------------------------------------------------------
# Part 2: prune one layer at 90% and measure accuracy
# ---------------------------------------------------------------------
def prune_single_layer_90(model: nn.Module,
                          layer_name: str,
                          module: nn.Module) -> nn.Module:
    """Return a deepcopy of model with ONLY this layer pruned 90%."""
    pruned_model = copy.deepcopy(model)
    # find the corresponding module in the copy by name
    submodule = dict(pruned_model.named_modules())[layer_name]
    prune.l1_unstructured(submodule, name="weight", amount=0.9)
    # make pruning permanent (remove mask / orig)
    prune.remove(submodule, "weight")
    return pruned_model


def single_layer_pruning_experiment(model: nn.Module,
                                    testloader,
                                    device: torch.device,
                                    max_test_batches: int = None
                                    ) -> Dict[str, float]:
    """Prune each conv/linear layer by 90% (L1) and record accuracy."""
    base_acc = evaluate_accuracy(model, testloader, device,
                                 max_batches=max_test_batches)
    print(f"\nBase (unpruned) accuracy on subset: {base_acc:.2f}%")

    layers = get_conv_and_linear_layers(model)
    acc_results = {}

    print("\n=== Pruning each layer by 90% (L1 unstructured) ===")
    for name, module in layers:
        pruned_model = prune_single_layer_90(model, name, module)
        acc = evaluate_accuracy(pruned_model, testloader, device,
                                max_batches=max_test_batches)
        acc_results[name] = acc
        print(f"Layer {name:40s} -> accuracy = {acc:.2f}%")

    # bar plot
    plt.figure(figsize=(12, 4))
    names = list(acc_results.keys())
    accs = [acc_results[n] for n in names]
    plt.bar(range(len(names)), accs)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel("Accuracy after 90% pruning (%)")
    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "pruning",
                            "single_layer_pruning_accuracy.png")
    plt.savefig(out_path)
    print(f"\nSaved bar plot to {out_path}\n")

    return acc_results


# ---------------------------------------------------------------------
# Part 3: search for maximum k for linear layers
# ---------------------------------------------------------------------
def prune_all_linear_layers(model: nn.Module, k: int) -> nn.Module:
    """Prune all Linear weights by k% using L1 unstructured."""
    amount = k / 100.0
    pruned_model = copy.deepcopy(model)
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return pruned_model


def find_max_k_for_linear_layers(model: nn.Module,
                                 testloader,
                                 device: torch.device,
                                 orig_acc: float,
                                 max_drop: float = 2.0,
                                 max_k: int = 100,
                                 max_test_batches: int = None) -> Tuple[int, float]:
    """
    Binary search for largest integer k in [0, max_k] such that
    pruning all Linear layers by k% causes accuracy drop <= max_drop.
    """
    low, high = 0, max_k
    best_k = 0
    best_acc = orig_acc

    print("\n=== Searching for maximal k for linear-layer pruning ===")
    while low <= high:
        mid = (low + high) // 2
        print(f"Testing k = {mid}% ... ", end="", flush=True)
        pruned_model = prune_all_linear_layers(model, mid)
        acc = evaluate_accuracy(pruned_model, testloader, device,
                                max_batches=max_test_batches)
        drop = orig_acc - acc
        print(f"acc = {acc:.2f}%, drop = {drop:.2f}%")

        if drop <= max_drop:
            best_k = mid
            best_acc = acc
            low = mid + 1    # try more aggressive pruning
        else:
            high = mid - 1   # too much pruning, go lower

    print(f"\nBest k = {best_k} with accuracy {best_acc:.2f}% "
          f"(drop {orig_acc - best_acc:.2f}%)")
    return best_k, best_acc


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data"),
                        help="Where to store CIFAR10 data.")
    parser.add_argument("--test-subset", type=int, default=5000,
                        help="Use only first N test examples for speed "
                             "(set None for full test set).")
    parser.add_argument("--max-test-batches", type=int, default=50,
                        help="Limit number of test batches for speed.")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # data
    _, testloader = get_cifar10_loaders(
        args.data_root, batch_size=128, num_workers=2,
        test_subset=args.test_subset,
    )

    # model
    model = load_resnet18(device)

    # 1) report conv/linear layers + params
    report_layer_params(model)

    # 2) prune each layer by 90% and collect accuracies
    layer_accs = single_layer_pruning_experiment(
        model, testloader, device, max_test_batches=args.max_test_batches
    )

    # 3) search for maximal k for all Linear layers
    orig_acc = evaluate_accuracy(model, testloader, device,
                                 max_batches=args.max_test_batches)
    max_k, acc_at_k = find_max_k_for_linear_layers(
        model, testloader, device, orig_acc,
        max_drop=2.0, max_k=95,
        max_test_batches=args.max_test_batches,
    )

    print("\n=== Summary for report ===")
    print(f"Original accuracy on subset: {orig_acc:.2f}%")
    print(f"Max k for linear-layer pruning (<=2% drop): {max_k}% "
          f"with accuracy {acc_at_k:.2f}%")
    print("\nLayer-wise 90% pruning accuracies:")
    for name, acc in layer_accs.items():
        print(f"  {name:40s} -> {acc:.2f}%")


if __name__ == "__main__":
    main()