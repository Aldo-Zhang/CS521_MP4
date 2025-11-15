import jax
import jax.numpy as jnp

# original function
def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

# gradients
dy_dx1 = jax.grad(f, argnums=0)
dy_dx2 = jax.grad(f, argnums=1)

x1, x2 = 2.0, 5.0

# ---- Option 1: new API (if available in your JAX) ----
print("=== HLO for dy_dx1 ===")
print(jax.jit(dy_dx1).lower(x1, x2).compiler_ir(dialect="hlo"))

print("\n=== HLO for dy_dx2 ===")
print(jax.jit(dy_dx2).lower(x1, x2).compiler_ir(dialect="hlo"))