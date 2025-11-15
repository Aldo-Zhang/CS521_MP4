import jax
import jax.numpy as jnp
from jax import grad


# --------------------------------------------------
# Original scalar function and its gradients
# --------------------------------------------------
def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)


dy_dx1 = grad(f, argnums=0)
dy_dx2 = grad(f, argnums=1)

# g2 from Task 5: single JIT that returns (f, df/dx1, df/dx2)
g2 = jax.jit(lambda x1, x2: (f(x1, x2), dy_dx1(x1, x2), dy_dx2(x1, x2)))


# --------------------------------------------------
# Task 6: vectorized versions of g2 using vmap
# --------------------------------------------------

# (a) batching across BOTH arguments x1 and x2
#     x1s and x2s must have the same shape; in_axes=(0, 0)
g2_vmap_both = jax.vmap(g2, in_axes=(0, 0))

# (b) batching across ONLY the first argument x1; x2 remains scalar
#     x1s is a vector, x2 is a scalar; in_axes=(0, None)
g2_vmap_x1_only = jax.vmap(g2, in_axes=(0, None))


# --------------------------------------------------
# Example usage and jaxpr printing
# --------------------------------------------------
if __name__ == "__main__":
    # Example vectors
    x1s = jnp.linspace(1.0, 10.0, 1000)
    x2s = x1s + 1.0

    # (a) vectorization across both x1s and x2s
    y_vals_a, dy_dx1_a, dy_dx2_a = g2_vmap_both(x1s, x2s)
    print("Result shapes for case (a):")
    print("  y_vals_a   :", y_vals_a.shape)
    print("  dy_dx1_a   :", dy_dx1_a.shape)
    print("  dy_dx2_a   :", dy_dx2_a.shape)

    # (b) vectorization across x1s only, with scalar x2 = 0.5
    y_vals_b, dy_dx1_b, dy_dx2_b = g2_vmap_x1_only(x1s, 0.5)
    print("\nResult shapes for case (b):")
    print("  y_vals_b   :", y_vals_b.shape)
    print("  dy_dx1_b   :", dy_dx1_b.shape)
    print("  dy_dx2_b   :", dy_dx2_b.shape)

    # --------------------------------------------------
    # Jaxpr for your report
    # --------------------------------------------------
    print("\n=== Jaxpr for case (a): vmap over both x1 and x2 ===")
    print(jax.make_jaxpr(g2_vmap_both)(x1s, x2s))

    print("\n=== Jaxpr for case (b): vmap over x1 only, x2 scalar ===")
    print(jax.make_jaxpr(g2_vmap_x1_only)(x1s, 0.5))