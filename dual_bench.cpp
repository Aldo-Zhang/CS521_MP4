#include "dual_number.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

float f_plain(float x) {
    return 1.0f / (1.0f + std::exp(-(std::sin(x) + x * x)));
}

dual_number f_dual(const dual_number& x) {
    return sigmoid(sin(x) + x * x);
}

volatile float sink_val;  // volatile sinks to defeat dead-code elimination
volatile float sink_der;

int main() {
    std::vector<int> sizes = {1000, 5000, 10000, 50000, 100000};

    std::cout << "N,time_plain_ms,time_dual_ms,overhead_ratio\n";

    for (int N : sizes) {
        std::vector<float> xs(N);
        for (int i = 0; i < N; ++i) {
            xs[i] = 0.001f * i;
        }

        // ---- plain ----
        auto t0 = std::chrono::high_resolution_clock::now();
        float acc_plain = 0.0f;
        for (int i = 0; i < N; ++i) {
            acc_plain += f_plain(xs[i]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        // ---- dual (value + derivative) ----
        auto t2 = std::chrono::high_resolution_clock::now();
        float acc_val = 0.0f;
        float acc_der = 0.0f;
        for (int i = 0; i < N; ++i) {
            dual_number x(xs[i], 1.0f);
            dual_number y = f_dual(x);
            acc_val += y.value();
            acc_der += y.dual();   // force derivative to actually be computed
        }
        auto t3 = std::chrono::high_resolution_clock::now();

        using fsec = std::chrono::duration<float, std::milli>;
        float time_plain = fsec(t1 - t0).count();
        float time_dual  = fsec(t3 - t2).count();

        float ratio = time_dual / time_plain;

        // use the results so they are observable
        sink_val = acc_val + acc_plain;
        sink_der = acc_der;

        std::cout << N << "," << time_plain << "," << time_dual << "," << ratio << "\n";
    }
}