#pragma once
// Heat-diffusion benchmark (Claude, 5 Nov 2026): t steps of the simple random walk on the periodic
// N x N grid, P = (S1 + S1^-1 + S2 + S2^-1) / 4, i.e. explicit time steps of the discrete heat
// equation. The target is Poly(P, x^t). Exact compilation needs t queries to P; the truncated Chebyshev
// method needs about sqrt(2 t ln(2/eps)) (Sachdeva & Vishnoi, Faster Algorithms via Approximation
// Theory, 2014; the mechanism behind quantum fast-forwarding, Apers & Sarlette 2019).
#include "symbol_route.h"

namespace bench {

// Each 2D shift is a single unitary leaf on the whole grid (axis 1 on the low qubits).
inline Term random_walk(std::size_t n) {
    const Eigen::Index N = Eigen::Index(1) << n;
    const Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(N, N);
    auto kron = [](const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) {   // A on the high qubits
        Eigen::MatrixXcd K(A.rows() * B.rows(), A.cols() * B.cols());
        for (Eigen::Index i = 0; i < A.rows(); ++i)
            for (Eigen::Index j = 0; j < A.cols(); ++j) K.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        return K;
    };
    return sum({0.25, 0.25, 0.25, 0.25},
               {unitary_leaf("S1", kron(I, cyclic_shift(N, 1))), unitary_leaf("S1^-1", kron(I, cyclic_shift(N, -1))),
                unitary_leaf("S2", kron(cyclic_shift(N, 1), I)), unitary_leaf("S2^-1", kron(cyclic_shift(N, -1), I))});
}

inline Term heat_steps(const Term& P, std::size_t t, double budget) {
    std::vector<double> c(t + 1, 0.0);
    c[t] = 1.0;
    return poly(P, c, budget);
}

// The matrix of the polynomial the truncated method actually builds: sum_k c_k T_k(M / alpha_M),
// by the recurrence T_{k+1} = 2 X T_k - T_{k-1} on matrices.
inline Eigen::MatrixXcd realised_matrix(const Term& poly_term) {
    const Info& I = info(poly_term);
    const Term& M = std::get<Poly>(poly_term->kind).a;
    const Eigen::MatrixXcd X = dense_of(M) / cost(M).alpha;
    const Eigen::Index D = X.rows();
    Eigen::MatrixXcd Tprev = Eigen::MatrixXcd::Identity(D, D), Tcur = X;
    Eigen::MatrixXcd R = I.realised_cheb[0] * Tprev;
    if (I.realised_cheb.size() > 1) R += I.realised_cheb[1] * Tcur;
    for (std::size_t k = 2; k < I.realised_cheb.size(); ++k) {
        Eigen::MatrixXcd Tnext = 2.0 * X * Tcur - Tprev;
        Tprev = Tcur;
        Tcur = Tnext;
        R += I.realised_cheb[k] * Tcur;
    }
    return R;
}

}  // namespace bench
