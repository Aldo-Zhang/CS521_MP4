import time
import jax
import jax.numpy as jnp
from jax import grad


# -----------------------------
# Original function and grads
# -----------------------------
def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)


dy_dx1 = grad(f, argnums=0)
dy_dx2 = grad(f, argnums=1)

# -----------------------------
# Version (a): separate jits
# g1(x1, x2) = (jax.jit(f)(x1,x2), jax.jit(dy_dx1)(x1,x2), jax.jit(dy_dx2)(x1,x2))
# but with explicit CPU / GPU variants
# -----------------------------
f_cpu = jax.jit(f, backend="cpu")
dx1_cpu = jax.jit(dy_dx1, backend="cpu")
dx2_cpu = jax.jit(dy_dx2, backend="cpu")

f_gpu = jax.jit(f, backend="gpu") if any(d.platform == "gpu" for d in jax.devices()) else None
dx1_gpu = jax.jit(dy_dx1, backend="gpu") if any(d.platform == "gpu" for d in jax.devices()) else None
dx2_gpu = jax.jit(dy_dx2, backend="gpu") if any(d.platform == "gpu" for d in jax.devices()) else None


def g1_cpu(x1, x2):
    return f_cpu(x1, x2), dx1_cpu(x1, x2), dx2_cpu(x1, x2)


def g1_gpu(x1, x2):
    return f_gpu(x1, x2), dx1_gpu(x1, x2), dx2_gpu(x1, x2)


# -----------------------------
# Version (b): one big jit
# g2 = jax.jit(lambda x1, x2: (f(x1,x2), dy_dx1(x1,x2), dy_dx2(x1,x2)))
# -----------------------------
g2_cpu = jax.jit(lambda x1, x2: (f(x1, x2), dy_dx1(x1, x2), dy_dx2(x1, x2)),
                 backend="cpu")

g2_gpu = (
    jax.jit(lambda x1, x2: (f(x1, x2), dy_dx1(x1, x2), dy_dx2(x1, x2)),
            backend="gpu")
    if any(d.platform == "gpu" for d in jax.devices())
    else None
)


# -----------------------------
# Helper: run and block for fair timing
# -----------------------------
def run_and_block(fun, x1, x2):
    out = fun(x1, x2)
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), out)
    return out


def benchmark(fun, x1, x2, n_iters=1000, label=""):
    # warm-up (compile)
    run_and_block(fun, x1, x2)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        run_and_block(fun, x1, x2)
    t1 = time.perf_counter()

    avg = (t1 - t0) / n_iters
    print(f"{label}: {avg * 1e6:.2f} us per call over {n_iters} iters")


# -----------------------------
# Main benchmarking for CPU and GPU
# -----------------------------
if __name__ == "__main__":
    x1 = 2.0
    x2 = 5.0

    print("=== CPU timings ===")
    benchmark(g1_cpu, x1, x2, n_iters=1000, label="g1_cpu (separate jits)")
    benchmark(g2_cpu, x1, x2, n_iters=1000, label="g2_cpu (single jit)")

    if any(d.platform == "gpu" for d in jax.devices()):
        print("\n=== GPU timings ===")
        benchmark(g1_gpu, x1, x2, n_iters=1000, label="g1_gpu (separate jits)")
        benchmark(g2_gpu, x1, x2, n_iters=1000, label="g2_gpu (single jit)")
    else:
        print("\nNo GPU detected; skipping GPU benchmarks.")

    # -------------------------
    # HLO generation
    # -------------------------
    print("\n=== HLO for g1 pieces (CPU) ===")
    print("HLO for jitted f:")
    print(jax.jit(f, backend="cpu").lower(x1, x2).compiler_ir(dialect="hlo").as_hlo_text())

    print("\nHLO for jitted dy_dx1:")
    print(jax.jit(dy_dx1, backend="cpu").lower(x1, x2).compiler_ir(dialect="hlo").as_hlo_text())

    print("\nHLO for jitted dy_dx2:")
    print(jax.jit(dy_dx2, backend="cpu").lower(x1, x2).compiler_ir(dialect="hlo").as_hlo_text())

    print("\n=== HLO for g2 (single jit, CPU) ===")
    g2_cpu_hlo = g2_cpu.lower(x1, x2).compiler_ir(dialect="hlo").as_hlo_text()
    print(g2_cpu_hlo)

    # If you also want GPU HLO, uncomment this block:
    # if g2_gpu is not None:
    #     print("\n=== HLO for g2 (single jit, GPU) ===")
    #     print(g2_gpu.lower(x1, x2).compiler_ir(dialect='hlo').as_hlo_text())