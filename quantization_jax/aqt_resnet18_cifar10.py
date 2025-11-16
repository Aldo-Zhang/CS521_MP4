#!/usr/bin/env python
#
# aqt_resnet18_cifar10.py
#
# Part 4: JAX + AQT quantization of a ResNet-18 for CIFAR-10.
#
# - Flax ResNet-18 that matches resnet_torch.ResNet18.
# - Imports pretrained PyTorch weights from resnet18_cifar10.pth.
# - Evaluates FP32 model in JAX (no training).
# - Builds INT8 model using AQT for the final linear layer (same weights).
# - Compares size, accuracy, speed.
# - Dumps HLO before and after quantization via as_hlo_text().
#

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn

import torch
import torchvision
import torchvision.transforms as T

import aqt.jax.v2.flax.aqt_flax as aqt
import aqt.jax.v2.config as aqt_config

# ---------------------------------------------------------------------
# Make sure we can import resnet_torch.py from the project root
# ---------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # .../CS521_MP4/quantization_jax
ROOT_DIR = os.path.dirname(THIS_DIR)                        # .../CS521_MP4
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import resnet_torch  # your original PyTorch ResNet


# ---------------------------------------------------------------------
# 1. Flax ResNet-18 (CIFAR-10) matching resnet_torch.ResNet18
# ---------------------------------------------------------------------


class BasicBlock(nn.Module):
    """CIFAR-10 BasicBlock matching resnet_torch.BasicBlock.

    - two 3x3 convs with BN and ReLU
    - optional 1x1 conv+BN shortcut when stride != 1 or channels change
    """
    in_planes: int
    planes: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        identity = x

        # conv1: 3x3
        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(x)
        out = nn.BatchNorm(use_running_average=not train, name="bn1")(out)
        out = nn.relu(out)

        # conv2: 3x3
        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(out)
        out = nn.BatchNorm(use_running_average=not train, name="bn2")(out)

        # Shortcut: identity or 1x1 conv when shape changes
        if self.stride != 1 or self.in_planes != self.planes:
            identity = nn.Conv(
                features=self.planes,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                name="shortcut_conv",
            )(x)
            identity = nn.BatchNorm(
                use_running_average=not train,
                name="shortcut_bn",
            )(identity)

        out = nn.relu(out + identity)
        return out


class ResNet18(nn.Module):
    """
    Flax ResNet-18 for CIFAR-10, matching resnet_torch.ResNet18:

    - Conv1: 3x3, stride=1, padding=1, out_channels=64 (no maxpool)
    - 4 stages with BasicBlocks:
        layer1: 64 -> 64  (2 blocks, stride 1)
        layer2: 64 -> 128 (stride 2), then 128 -> 128
        layer3: 128 -> 256 (stride 2), then 256 -> 256
        layer4: 256 -> 512 (stride 2), then 512 -> 512
    - global average pool + linear(512 -> num_classes)

    If dot_general is None, final linear is a standard FP32 Dense.
    If dot_general is an AqtDotGeneral, final linear uses INT8 dot_general.
    """
    num_classes: int = 10
    dot_general: object | None = None  # aqt.AqtDotGeneral or None

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Expect NHWC input: [N, 32, 32, 3]
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)

        # layer1: 64 -> 64 (2 blocks)
        x = BasicBlock(in_planes=64, planes=64, stride=1, name="layer1_0")(x, train=train)
        x = BasicBlock(in_planes=64, planes=64, stride=1, name="layer1_1")(x, train=train)

        # layer2: 64 -> 128, then 128 -> 128
        x = BasicBlock(in_planes=64, planes=128, stride=2, name="layer2_0")(x, train=train)
        x = BasicBlock(in_planes=128, planes=128, stride=1, name="layer2_1")(x, train=train)

        # layer3: 128 -> 256, then 256 -> 256
        x = BasicBlock(in_planes=128, planes=256, stride=2, name="layer3_0")(x, train=train)
        x = BasicBlock(in_planes=256, planes=256, stride=1, name="layer3_1")(x, train=train)

        # layer4: 256 -> 512, then 512 -> 512
        x = BasicBlock(in_planes=256, planes=512, stride=2, name="layer4_0")(x, train=train)
        x = BasicBlock(in_planes=512, planes=512, stride=1, name="layer4_1")(x, train=train)

        # global avg pool over spatial dims (like F.avg_pool2d(out, 4))
        x = jnp.mean(x, axis=(1, 2))  # [N, 512]

        dense_kwargs = {}
        if self.dot_general is not None:
            dense_kwargs["dot_general"] = self.dot_general

        x = nn.Dense(features=self.num_classes, name="linear", **dense_kwargs)(x)
        return x


def init_resnet18_params(rng, input_shape=(1, 32, 32, 3), num_classes: int = 10):
    """Init ResNet18 and return params + batch_stats."""
    model = ResNet18(num_classes=num_classes, dot_general=None)
    dummy_x = jnp.zeros(input_shape, jnp.float32)

    # train=False → BatchNorm creates batch_stats but uses running_average
    variables = model.init(rng, dummy_x, train=False)

    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})
    return model, params, batch_stats


# ---------------------------------------------------------------------
# 2. PyTorch -> Flax weight import
# ---------------------------------------------------------------------


def load_resnet18_params_from_pytorch(
    rng,
    ckpt_path: str,
    input_shape=(1, 32, 32, 3),
    num_classes: int = 10,
):
    """
    Load pretrained PyTorch weights into Flax ResNet18.

    We reuse the batch_stats from a normal Flax init.
    """
    # 1) Build PyTorch model and load weights
    pt_model = resnet_torch.ResNet18()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        pt_state = checkpoint["state_dict"]
    else:
        pt_state = checkpoint
    pt_model.load_state_dict(pt_state)
    pt_model.eval()

    sd = pt_model.state_dict()

    # 2) Build Flax model to get params + batch_stats structure
    model, _, batch_stats = init_resnet18_params(
        rng,
        input_shape=input_shape,
        num_classes=num_classes,
    )

    # 3) Helpers for mapping weights

    def conv_weight(pt_key: str):
        """Map PyTorch conv weight [out, in, kh, kw] -> Flax [kh, kw, in, out]."""
        w = sd[pt_key].cpu().numpy()
        return jnp.asarray(w.transpose(2, 3, 1, 0))

    def bn_params(pt_prefix: str):
        """Map PyTorch BatchNorm gamma/beta -> Flax BN scale/bias."""
        gamma = sd[f"{pt_prefix}.weight"].cpu().numpy()
        beta = sd[f"{pt_prefix}.bias"].cpu().numpy()
        return {
            "scale": jnp.asarray(gamma),
            "bias": jnp.asarray(beta),
        }

    def basic_block(pt_prefix: str):
        """
        Map one BasicBlock:
          pt_prefix = "layer1.0" etc.
        """
        block = {}

        # conv1, bn1
        block["conv1"] = {"kernel": conv_weight(f"{pt_prefix}.conv1.weight")}
        block["bn1"] = bn_params(f"{pt_prefix}.bn1")

        # conv2, bn2
        block["conv2"] = {"kernel": conv_weight(f"{pt_prefix}.conv2.weight")}
        block["bn2"] = bn_params(f"{pt_prefix}.bn2")

        # shortcut if present
        sc_conv_key = f"{pt_prefix}.shortcut.0.weight"
        if sc_conv_key in sd:
            block["shortcut_conv"] = {"kernel": conv_weight(sc_conv_key)}
            block["shortcut_bn"] = bn_params(f"{pt_prefix}.shortcut.1")

        return block

    # 4) Build the entire Flax param tree from the PyTorch state_dict

    params = {}

    # Top conv + BN
    params["conv1"] = {"kernel": conv_weight("conv1.weight")}
    params["bn1"] = bn_params("bn1")

    # Layer1: two BasicBlocks
    params["layer1_0"] = basic_block("layer1.0")
    params["layer1_1"] = basic_block("layer1.1")

    # Layer2
    params["layer2_0"] = basic_block("layer2.0")
    params["layer2_1"] = basic_block("layer2.1")

    # Layer3
    params["layer3_0"] = basic_block("layer3.0")
    params["layer3_1"] = basic_block("layer3.1")

    # Layer4
    params["layer4_0"] = basic_block("layer4.0")
    params["layer4_1"] = basic_block("layer4.1")

    # Final linear: PyTorch [out, in] -> Flax Dense kernel [in, out]
    w = sd["linear.weight"].cpu().numpy()  # [num_classes, 512]
    b = sd["linear.bias"].cpu().numpy()    # [num_classes]
    params["linear"] = {
        "kernel": jnp.asarray(w.T),  # [512, num_classes]
        "bias": jnp.asarray(b),
    }

    return model, params, batch_stats


# ---------------------------------------------------------------------
# 3. CIFAR-10 test loader (TorchVision) and basic JAX helpers
# ---------------------------------------------------------------------


def make_cifar10_testloader(batch_size: int = 128, num_workers: int = 2):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return testloader


def torch_batch_to_jax(batch) -> Tuple[jnp.ndarray, jnp.ndarray]:
    images, labels = batch
    # PyTorch: NCHW, JAX/Flax: NHWC
    images = images.permute(0, 2, 3, 1).numpy()
    labels = labels.numpy()
    return jnp.asarray(images), jnp.asarray(labels)


def params_nbytes(params) -> float:
    """Approximate parameter size in MB."""
    def leaf_nbytes(x):
        arr = np.asarray(x)
        return arr.size * arr.itemsize

    total = jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_util.tree_map(leaf_nbytes, params),
    )
    return total / (1024 ** 2)


def forward_logits(model, params, batch_stats, x, train: bool = False):
    """
    Forward pass with explicit params + batch_stats.

    For evaluation we call with train=False so BatchNorm uses
    running_average=True and does not update batch_stats.
    """
    variables = {
        "params": params,
        "batch_stats": batch_stats,
    }
    return model.apply(variables, x, train=train, mutable=False)


def evaluate(model, params, batch_stats, testloader, max_batches: int | None = None) -> float:
    correct = 0
    total = 0
    for i, batch in enumerate(testloader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = torch_batch_to_jax(batch)
        logits = forward_logits(model, params, batch_stats, x, train=False)
        preds = jnp.argmax(logits, axis=-1)
        correct += int((preds == y).sum())
        total += y.shape[0]
    return 100.0 * correct / total if total > 0 else 0.0


def measure_inference_time(model, params, batch_stats, example_batch, n_iters: int = 50) -> float:
    x, _ = example_batch

    @jax.jit
    def run_once(x_):
        variables = {
            "params": params,
            "batch_stats": batch_stats,
        }
        return model.apply(variables, x_, train=False, mutable=False)

    # Warmup
    run_once(x).block_until_ready()

    t0 = time.time()
    for _ in range(n_iters):
        out = run_once(x)
    out.block_until_ready()
    t1 = time.time()

    return (t1 - t0) * 1000.0 / n_iters  # ms per batch


def dump_hlo(model, params, batch_stats, example_batch, out_path_prefix: str):
    """Dump HLO text for forward pass using as_hlo_text()."""
    x, _ = example_batch

    def fwd(x_):
        variables = {
            "params": params,
            "batch_stats": batch_stats,
        }
        return model.apply(variables, x_, train=False, mutable=False)

    comp = jax.xla_computation(fwd)(x)
    hlo_text = comp.as_hlo_text()

    out_dir = os.path.dirname(out_path_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    path = out_path_prefix + ".hlo.txt"
    with open(path, "w") as f:
        f.write(hlo_text)
    print(f"[HLO] saved to {path}")


# ---------------------------------------------------------------------
# 4. Main: FP32 vs INT8 (AQT) comparison
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=50,
                        help="number of test batches for timing/accuracy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pt-ckpt-path", type=str, default="resnet18_cifar10.pth",
                        help="path to PyTorch ResNet18 checkpoint")
    args = parser.parse_args()

    print("JAX devices:", jax.devices())

    # Data
    testloader = make_cifar10_testloader(batch_size=args.batch_size)
    first_batch = next(iter(testloader))
    example_x, example_y = torch_batch_to_jax(first_batch)

    rng = jax.random.PRNGKey(args.seed)

    # ---------------- FP32 model (with imported PT weights) ----------------
    print("\n=== FP32 Flax ResNet18 (imported from PyTorch) ===")
    print(f"Loading PyTorch weights from: {args.pt_ckpt_path}")
    model_fp32, params_fp32, batch_stats_fp32 = load_resnet18_params_from_pytorch(
        rng,
        ckpt_path=args.pt_ckpt_path,
        input_shape=example_x.shape,
        num_classes=10,
    )

    size_fp32_mb = params_nbytes(params_fp32)
    print(f"FP32 param size: {size_fp32_mb:.2f} MB")

    acc_fp32 = evaluate(
        model_fp32, params_fp32, batch_stats_fp32,
        testloader, max_batches=args.eval_batches
    )
    print(f"FP32 test accuracy (~{args.eval_batches} batches): {acc_fp32:.2f}%")

    t_fp32 = measure_inference_time(
        model_fp32,
        params_fp32,
        batch_stats_fp32,
        example_batch=(example_x, example_y),
        n_iters=50,
    )
    print(f"FP32 avg inference time: {t_fp32:.2f} ms per batch")

    dump_hlo(
        model_fp32,
        params_fp32,
        batch_stats_fp32,
        example_batch=(example_x, example_y),
        out_path_prefix="quantization_jax/resnet18_fp32",
    )

    # ---------------- INT8 model (AQT on final linear) --------------------
    print("\n=== INT8 (AQT) Flax ResNet18 (same weights, quantized final layer) ===")

    int8_cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
    dot_gen = aqt.AqtDotGeneral(int8_cfg)

    # Same architecture, but Dense uses AQT dot_general
    model_int8 = ResNet18(num_classes=10, dot_general=dot_gen)

    # We reuse the SAME params (weights) for the INT8 model
    params_int8 = params_fp32

    batch_stats_int8 = batch_stats_fp32

    size_int8_mb = params_nbytes(params_int8)
    print(f"INT8 param size (same shapes, AQT matmul): {size_int8_mb:.2f} MB")

    acc_int8 = evaluate(
        model_int8, params_int8, batch_stats_int8,
        testloader, max_batches=args.eval_batches
    )
    print(f"INT8 test accuracy (~{args.eval_batches} batches): {acc_int8:.2f}%")

    t_int8 = measure_inference_time(
        model_int8,
        params_int8,
        batch_stats_int8,
        example_batch=(example_x, example_y),
        n_iters=50,
    )
    print(f"INT8 avg inference time: {t_int8:.2f} ms per batch")

    dump_hlo(
        model_int8,
        params_int8,
        batch_stats_int8,
        example_batch=(example_x, example_y),
        out_path_prefix="quantization_jax/resnet18_int8",
    )
    # ---------------- Summary ----------------
    print("\n=== Summary (JAX + AQT, ResNet18 with imported PT weights) ===")
    print(f"FP32 size      : {size_fp32_mb:.2f} MB")
    print(f"INT8 size      : {size_int8_mb:.2f} MB")
    print(f"FP32 accuracy  : {acc_fp32:.2f}%")
    print(f"INT8 accuracy  : {acc_int8:.2f}%")
    print(f"FP32 time (ms) : {t_fp32:.2f}")
    print(f"INT8 time (ms) : {t_int8:.2f}")
    print("HLO files:")
    print("  jax_quantization/resnet18_fp32.hlo.txt")
    print("  jax_quantization/resnet18_int8.hlo.txt")


if __name__ == "__main__":
    main()