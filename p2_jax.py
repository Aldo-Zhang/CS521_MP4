import jax
import jax.numpy as jnp


# f(x1, x2) = ln(x1) + x1 * x2 - sin(x2)
def f(x1, x2):
    # use jax.numpy, NOT regular numpy
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)


# Gradients of f w.r.t. x1 and x2, using reverse-mode AD (jax.grad)
dy_dx1 = jax.grad(f, argnums=0)   # ∂f/∂x1
dy_dx2 = jax.grad(f, argnums=1)   # ∂f/∂x2


if __name__ == "__main__":
    x1 = 2.0
    x2 = 5.0

    y = f(x1, x2)
    print("f(x1, x2) =", y)

    g1 = dy_dx1(x1, x2)
    g2 = dy_dx2(x1, x2)
    print("dy_dx1 =", g1)
    print("dy_dx2 =", g2)

    # (this is already useful for Task 2 later)
    print("\nJaxpr for dy_dx1:")
    print(jax.make_jaxpr(dy_dx1)(x1, x2))
    print("\nJaxpr for dy_dx2:")
    print(jax.make_jaxpr(dy_dx2)(x1, x2))