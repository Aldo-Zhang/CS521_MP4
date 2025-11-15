import math
import jax
import jax.numpy as jnp

# --------------------------------------------------
# JAX version of f and its gradients
# --------------------------------------------------
def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

dy_dx1 = jax.grad(f, argnums=0)
dy_dx2 = jax.grad(f, argnums=1)

# --------------------------------------------------
# Classical reverse-mode trace from the lecture
# --------------------------------------------------
def classical_reverse_trace(x1, x2):
    print("=== Classical forward (primal) trace ===")
    v_m1 = x1
    v0 = x2
    v1 = math.log(v_m1)
    v2 = v_m1 * v0
    v3 = math.sin(v0)
    v4 = v1 + v2
    v5 = v4 - v3
    y = v5

    print(f"v_-1 = x1   = {v_m1}")
    print(f"v_0  = x2   = {v0}")
    print(f"v_1  = ln v_-1   = {v1}")
    print(f"v_2  = v_-1 * v_0 = {v2}")
    print(f"v_3  = sin v_0   = {v3}")
    print(f"v_4  = v_1 + v_2 = {v4}")
    print(f"v_5  = v_4 - v_3 = {v5}")
    print(f"y = v_5 = {y}\n")

    print("=== Classical reverse (adjoint) trace ===")
    vbar5 = 1.0
    vbar4 = vbar3 = vbar2 = vbar1 = vbar0 = vbar_m1 = 0.0

    # v5 = v4 - v3
    vbar4 += vbar5 * 1.0
    vbar3 += vbar5 * (-1.0)

    # v4 = v1 + v2
    vbar1 += vbar4 * 1.0
    vbar2 += vbar4 * 1.0

    # v3 = sin v0
    vbar0 += vbar3 * math.cos(v0)

    # v2 = v_-1 * v0
    vbar_m1 += vbar2 * v0
    vbar0  += vbar2 * v_m1

    # v1 = ln v_-1
    vbar_m1 += vbar1 * (1.0 / v_m1)

    print(f"vbar_5 = {vbar5}")
    print(f"vbar_4 = {vbar4}")
    print(f"vbar_3 = {vbar3}")
    print(f"vbar_2 = {vbar2}")
    print(f"vbar_1 = {vbar1}")
    print(f"vbar_0 = {vbar0}   (∂y/∂x2)")
    print(f"vbar_-1 = {vbar_m1} (∂y/∂x1)")
    print()

    return y, vbar_m1, vbar0


# --------------------------------------------------
# JAX jaxpr-style computations for the gradients
# --------------------------------------------------
def jaxpr_grad_traces(x1, x2):
    a = x1
    b = x2

    print("=== JAX jaxpr-style for dy_dx1 ===")
    c = math.log(a)
    d = a * b
    e = c + d
    f_ = math.sin(b)
    _tmp = e - f_          # value of f(x1,x2)
    g = 1.0 * b            # x2
    h = 1.0 / a            # 1/x1
    i = g + h              # dy/dx1

    print(f"c = log(a)      = {c}")
    print(f"d = a * b       = {d}")
    print(f"e = c + d       = {e}")
    print(f"f = sin(b)      = {f_}")
    print(f"_ = e - f       = {_tmp}")
    print(f"g = 1.0 * b     = {g}")
    print(f"h = 1.0 / a     = {h}")
    print(f"i = g + h       = {i}   (dy/dx1)\n")

    print("=== JAX jaxpr-style for dy_dx2 ===")
    c = math.log(a)
    d = a * b
    e = c + d
    f_ = math.sin(b)
    g = math.cos(b)
    _tmp = e - f_
    h = -1.0
    i2 = h * g             # -cos(b)
    j = a * 1.0            # x1
    k = i2 + j             # dy/dx2

    print(f"c = log(a)      = {c}")
    print(f"d = a * b       = {d}")
    print(f"e = c + d       = {e}")
    print(f"f = sin(b)      = {f_}")
    print(f"g = cos(b)      = {g}")
    print(f"_ = e - f       = {_tmp}")
    print(f"h = -1.0        = {h}")
    print(f"i = h * g       = {i2}")
    print(f"j = a * 1.0     = {j}")
    print(f"k = i + j       = {k}   (dy/dx2)\n")

    return i, k


# --------------------------------------------------
# Run everything at (2.0, 5.0)
# --------------------------------------------------
if __name__ == "__main__":
    x1, x2 = 2.0, 5.0

    print("=== JAX grad results ===")
    print("dy_dx1(x1,x2) =", float(dy_dx1(x1, x2)))
    print("dy_dx2(x1,x2) =", float(dy_dx2(x1, x2)))
    print()

    classical_reverse_trace(x1, x2)
    jaxpr_grad_traces(x1, x2)