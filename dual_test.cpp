// dual_tests.cpp
#include "dual_number.h"

#include <gtest/gtest.h>
#include <cmath>

// Small helper for comparing floats
static constexpr float kEps = 1e-4f;

inline void ExpectNearFloat(float a, float b, float eps = kEps) {
    EXPECT_NEAR(a, b, eps);
}


// Constructor tests

TEST(DualNumberBasics, DefaultConstructor) {
    dual_number x;
    ExpectNearFloat(x.value(), 0.0f);
    ExpectNearFloat(x.dual(),  0.0f);
}

TEST(DualNumberBasics, ValueConstructor) {
    dual_number x(1.5f);
    ExpectNearFloat(x.value(), 1.5f);
    ExpectNearFloat(x.dual(),  0.0f);
}

TEST(DualNumberBasics, ValueDualConstructor) {
    dual_number x(1.0f, 2.0f);
    ExpectNearFloat(x.value(), 1.0f);
    ExpectNearFloat(x.dual(),  2.0f);
}

// Arithmetic operators

TEST(DualNumberOps, Addition) {
    dual_number x(1.0f, 1.0f);
    dual_number y(2.0f, 3.0f);

    auto z = x + y;
    ExpectNearFloat(z.value(), 3.0f);  // 1 + 2
    ExpectNearFloat(z.dual(),  4.0f);  // 1 + 3
}

TEST(DualNumberOps, Subtraction) {
    dual_number x(5.0f, 4.0f);
    dual_number y(2.0f, 1.0f);

    auto z = x - y;
    ExpectNearFloat(z.value(), 3.0f);  // 5 - 2
    ExpectNearFloat(z.dual(),  3.0f);  // 4 - 1
}

TEST(DualNumberOps, MultiplicationProductRule) {
    dual_number x(2.0f, 3.0f);  // (v, v')
    dual_number y(4.0f, 5.0f);

    auto z = x * y;
    // Value: 2 * 4 = 8
    ExpectNearFloat(z.value(), 8.0f);

    // Derivative: (x' * y) + (x * y') = 3*4 + 2*5 = 12 + 10 = 22
    ExpectNearFloat(z.dual(), 22.0f);
}

TEST(DualNumberOps, DivisionQuotientRule) {
    dual_number x(6.0f, 2.0f);  // u, u'
    dual_number y(3.0f, 1.0f);  // v, v'

    auto z = x / y;
    // Value: 6 / 3 = 2
    ExpectNearFloat(z.value(), 2.0f);

    // Derivative: (u'v - u v') / v^2 = (2*3 - 6*1) / 9 = (6-6)/9 = 0
    ExpectNearFloat(z.dual(), 0.0f);
}

// Elementary function tests (analytic derivatives)


TEST(DualNumberFuncs, SinDerivative) {
    float x0 = 0.3f;
    dual_number x(x0, 1.0f);  // dx/dx = 1

    auto y = sin(x);
    // y = sin(x), dy/dx = cos(x)
    ExpectNearFloat(y.value(), std::sin(x0));
    ExpectNearFloat(y.dual(),  std::cos(x0));
}

TEST(DualNumberFuncs, CosDerivative) {
    float x0 = -0.7f;
    dual_number x(x0, 1.0f);

    auto y = cos(x);
    // y = cos(x), dy/dx = -sin(x)
    ExpectNearFloat(y.value(), std::cos(x0));
    ExpectNearFloat(y.dual(),  -std::sin(x0));
}

TEST(DualNumberFuncs, ExpDerivative) {
    float x0 = 1.1f;
    dual_number x(x0, 1.0f);

    auto y = exp(x);
    // y = exp(x), dy/dx = exp(x)
    ExpectNearFloat(y.value(), std::exp(x0));
    ExpectNearFloat(y.dual(),  std::exp(x0));
}

TEST(DualNumberFuncs, LnDerivative) {
    float x0 = 2.5f;
    dual_number x(x0, 1.0f);

    auto y = ln(x);
    // y = ln(x), dy/dx = 1/x
    ExpectNearFloat(y.value(), std::log(x0));
    ExpectNearFloat(y.dual(),  1.0f / x0);
}

TEST(DualNumberFuncs, ReluDerivative) {
    dual_number x_pos(2.0f, 3.0f);   // > 0
    dual_number x_neg(-1.0f, 4.0f);  // < 0

    auto y_pos = relu(x_pos);
    auto y_neg = relu(x_neg);

    // ReLU(x) = x (for x>0), derivative is 1 * x'
    ExpectNearFloat(y_pos.value(), 2.0f);
    ExpectNearFloat(y_pos.dual(),  3.0f);

    // ReLU(x) = 0 (for x<0), derivative is 0
    ExpectNearFloat(y_neg.value(), 0.0f);
    ExpectNearFloat(y_neg.dual(),  0.0f);
}

TEST(DualNumberFuncs, SigmoidDerivative) {
    float x0 = 0.4f;
    dual_number x(x0, 1.0f);

    auto y = sigmoid(x);

    float s = 1.0f / (1.0f + std::exp(-x0));
    // y.value() should equal s
    ExpectNearFloat(y.value(), s);
    // y.dual() should equal s(1-s)
    ExpectNearFloat(y.dual(),  s * (1.0f - s));
}

TEST(DualNumberFuncs, TanhDerivative) {
    float x0 = -0.9f;
    dual_number x(x0, 1.0f);

    auto y = tanh(x);

    float t = std::tanh(x0);
    ExpectNearFloat(y.value(), t);
    ExpectNearFloat(y.dual(),  1.0f - t * t);
}

// Composite function test vs numerical finite difference
// f(x) = sigmoid( sin(x) + x^2 )

// Helper: compute f(x) using dual numbers and return (value, derivative)
dual_number f_dual(float x0) {
    dual_number x(x0, 1.0f);               // derivative wrt x is 1
    dual_number y = sigmoid(sin(x) + x * x);
    return y;
}

float f_plain(float x) {
    return 1.0f / (1.0f + std::exp(- (std::sin(x) + x * x)));
}

TEST(DualNumberComposite, MatchesFiniteDifference) {
    float x0 = 0.2f;
    dual_number y = f_dual(x0);

    float h = 1e-3f;
    float fd = (f_plain(x0 + h) - f_plain(x0 - h)) / (2.0f * h);  // central diff

    ExpectNearFloat(y.value(), f_plain(x0), 1e-5f);
    ExpectNearFloat(y.dual(),  fd,         1e-3f);
}

// dual_vector simple test (element-wise sin)

TEST(DualVector, ElementwiseSin) {
    dual_vector xs;
    xs.emplace_back(0.0f, 1.0f);  // d/dx
    xs.emplace_back(1.0f, 1.0f);
    xs.emplace_back(2.0f, 1.0f);

    dual_vector ys = sin(xs);

    ASSERT_EQ(xs.size(), ys.size());

    for (std::size_t i = 0; i < xs.size(); ++i) {
        float v = xs[i].value();
        ExpectNearFloat(ys[i].value(), std::sin(v));
        ExpectNearFloat(ys[i].dual(),  std::cos(v));  // derivative of sin is cos
    }
}

// Google Test main (if you don't have a separate main in your test runner)

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}