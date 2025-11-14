#ifndef DUAL_NUMBER_H
#define DUAL_NUMBER_H

#include <cmath>
#include <vector>

struct dual_number {
    float val;  // primal value
    float der;  // derivative (dual part)

    // Constructors

    // default: value = 0, derivative = 0
    dual_number() : val(0.0f), der(0.0f) {}

    // value v, derivative = 0
    dual_number(float v) : val(v), der(0.0f) {}

    // full constructor: value v, derivative d
    dual_number(float v, float d) : val(v), der(d) {}

    // Accessors to match the spec
    float value() const { return val; }
    float dual()  const { return der; }

    // In-place arithmetic operators

    dual_number& operator+=(const dual_number& rhs) {
        val += rhs.val;
        der += rhs.der;
        return *this;
    }

    dual_number& operator-=(const dual_number& rhs) {
        val -= rhs.val;
        der -= rhs.der;
        return *this;
    }

    dual_number& operator*=(const dual_number& rhs) {
        // (u, u') * (v, v') = (uv, u'v + uv')
        float new_val = val * rhs.val;
        float new_der = der * rhs.val + val * rhs.der;
        val = new_val;
        der = new_der;
        return *this;
    }

    dual_number& operator/=(const dual_number& rhs) {
        // (u, u') / (v, v') = (u/v, (u'v - uv') / v^2)
        float inv_v  = 1.0f / rhs.val;
        float new_val = val * inv_v;
        float new_der = (der * rhs.val - val * rhs.der) * (inv_v * inv_v);
        val = new_val;
        der = new_der;
        return *this;
    }
};

// Binary arithmetic operators (non-member, use in-place versions)
// inline used for possible compiler optimization and avoid linker error

inline dual_number operator+(dual_number lhs, const dual_number& rhs) {
    lhs += rhs;
    return lhs;
}

inline dual_number operator-(dual_number lhs, const dual_number& rhs) {
    lhs -= rhs;
    return lhs;
}

inline dual_number operator*(dual_number lhs, const dual_number& rhs) {
    lhs *= rhs;
    return lhs;
}

inline dual_number operator/(dual_number lhs, const dual_number& rhs) {
    lhs /= rhs;
    return lhs;
}

// Convenience overloads with float (converted to dual with zero derivative)
inline dual_number operator+(dual_number lhs, float rhs) {
    return lhs + dual_number(rhs);
}
inline dual_number operator+(float lhs, dual_number rhs) {
    return dual_number(lhs) + rhs;
}

inline dual_number operator-(dual_number lhs, float rhs) {
    return lhs - dual_number(rhs);
}
inline dual_number operator-(float lhs, dual_number rhs) {
    return dual_number(lhs) - rhs;
}

inline dual_number operator*(dual_number lhs, float rhs) {
    return lhs * dual_number(rhs);
}
inline dual_number operator*(float lhs, dual_number rhs) {
    return dual_number(lhs) * rhs;
}

inline dual_number operator/(dual_number lhs, float rhs) {
    return lhs / dual_number(rhs);
}
inline dual_number operator/(float lhs, dual_number rhs) {
    return dual_number(lhs) / rhs;
}

// Elementary functions on dual numbers

// sin(x): (sin v, cos v * v')
inline dual_number sin(const dual_number& x) {
    float v = std::sin(x.val);
    float d = std::cos(x.val) * x.der;
    return dual_number(v, d);
}

// cos(x): (cos v, -sin v * v')
inline dual_number cos(const dual_number& x) {
    float v = std::cos(x.val);
    float d = -std::sin(x.val) * x.der;
    return dual_number(v, d);
}

// exp(x): (exp v, exp v * v')
inline dual_number exp(const dual_number& x) {
    float v = std::exp(x.val);
    float d = v * x.der;
    return dual_number(v, d);
}

// ln(x): (ln v, v'/v)
inline dual_number ln(const dual_number& x) {
    float v = std::log(x.val);
    float d = x.der / x.val;
    return dual_number(v, d);
}

// relu(x): (max(0,v), (v > 0 ? v' : 0))
inline dual_number relu(const dual_number& x) {
    if (x.val > 0.0f) {
        return dual_number(x.val, x.der);
    } else {
        return dual_number(0.0f, 0.0f);
    }
}

// sigmoid(x) = 1 / (1 + e^{-x})
inline dual_number sigmoid(const dual_number& x) {
    float s = 1.0f / (1.0f + std::exp(-x.val));
    float d = s * (1.0f - s) * x.der;
    return dual_number(s, d);
}

// tanh(x)
inline dual_number tanh(const dual_number& x) {
    float t = std::tanh(x.val);
    float d = (1.0f - t * t) * x.der;
    return dual_number(t, d);
}

// A simple dual_vector type and element-wise operations

using dual_vector = std::vector<dual_number>;

inline dual_vector sin(const dual_vector& xs) {
    dual_vector ys;
    ys.reserve(xs.size());
    for (const auto& x : xs) ys.push_back(sin(x));
    return ys;
}

inline dual_vector cos(const dual_vector& xs) {
    dual_vector ys;
    ys.reserve(xs.size());
    for (const auto& x : xs) ys.push_back(cos(x));
    return ys;
}

inline dual_vector exp(const dual_vector& xs) {
    dual_vector ys;
    ys.reserve(xs.size());
    for (const auto& x: xs) ys.push_back(exp(x));
    return ys;
}

inline dual_vector ln(const dual_vector& xs) {
    dual_vector ys;
    ys.reserve(xs.size());
    for (const auto& x: xs) ys.push_back(ln(x));
    return ys;
}

inline dual_vector relu(const dual_vector& xs) {
    dual_vector ys;
    ys.reserve(xs.size());
    for (const auto& x: xs) ys.push_back(relu(x));
    return ys;
}

inline dual_vector sigmoid(const dual_vector& xs) {
    dual_vector ys;
    ys.reserve(xs.size());
    for (const auto& x: xs) ys.push_back(sigmoid(x));
    return ys;
}

inline dual_vector tanh(const dual_vector& xs) {
    dual_vector ys;
    ys.reserve(xs.size());
    for (const auto& x: xs) ys.push_back(tanh(x));
    return ys;
}

#endif // DUAL_NUMBER_H