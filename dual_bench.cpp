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

int main() {
    std::vector<int> sizes = {1000, 5000, 10000, 50000, 100000};

    std::cout << "N,time_plain_ms,time_dual_ms,overhead_ratio\n";

    for (int N : sizes) {
        std::vector<float> xs(N);
        for (int i = 0; i < N; ++i) {
            xs[i] = 0.001f * i; // some deterministic inputs
        }

        // --- plain computation ---
        auto t0 = std::chrono::high_resolution_clock::now();
        float acc_plain = 0.0f;
        for (int i = 0; i < N; ++i) {
            acc_plain += f_plain(xs[i]); // accumulate so compiler can't remove it
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        // --- dual computation ---
        auto t2 = std::chrono::high_resolution_clock::now();
        float acc_dual = 0.0f;
        for (int i = 0; i < N; ++i) {
            dual_number x(xs[i], 1.0f); // derivative wrt x
            dual_number y = f_dual(x);
            acc_dual += y.value();      // again, stop dead-code elimination
        }
        auto t3 = std::chrono::high_resolution_clock::now();

        using fsec = std::chrono::duration<float, std::milli>;
        float time_plain = fsec(t1 - t0).count();
        float time_dual  = fsec(t3 - t2).count();
        float ratio = time_dual / time_plain;

        std::cout << N << "," << time_plain << "," << time_dual << "," << ratio << "\n";

        // print acc_ values so compiler can't optimize loops away
        if (acc_plain == 0.0f || acc_dual == 0.0f) {
            std::cerr << "dummy: " << acc_plain << " " << acc_dual << "\n";
        }
    }
}