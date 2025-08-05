#pragma once

template<typename T>
struct MatMulParams {
    T *A;
    T *B;
    T *C;
    T *D;
    float alpha;
    float beta;
    size_t M;
    size_t N;
    size_t K;

    MatMulParams(size_t M, size_t N, size_t K, float alpha, float beta) : alpha(alpha),
                                                                          beta(beta), M(M),
                                                                          N(N),
                                                                          K(K) {
        A = nullptr;
        B = nullptr;
        C = nullptr;
        D = nullptr;
    }
};
