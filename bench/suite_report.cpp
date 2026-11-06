// Structured-operator suite (Claude, 5 Nov 2026). Scope: operators that commute with the shifts of a
// periodic grid (discretised constant-coefficient PDE operators). Each is written the natural way, as
// its Pauli decomposition, and then passed to optimise() (symbol route, kept only when cheaper).
// Costs are queries x alpha (Cobble's total cost), not gate counts.
#include "heat.h"
#include <cstdio>

using namespace bench;

namespace {
Eigen::MatrixXcd circulant(std::size_t n, const std::vector<std::pair<long, double>>& taps) {   // (T f)(x) = sum w f(x + s)
    const Eigen::Index N = Eigen::Index(1) << n;
    Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(N, N);
    for (auto [s, w] : taps) T += w * cyclic_shift(N, s);
    return T;
}
Eigen::MatrixXcd kron(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) {   // A on the high qubits
    Eigen::MatrixXcd K(A.rows() * B.rows(), A.cols() * B.cols());
    for (Eigen::Index i = 0; i < A.rows(); ++i)
        for (Eigen::Index j = 0; j < A.cols(); ++j) K.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
    return K;
}
void row(const char* name, const Eigen::MatrixXcd& A, std::size_t n1, std::size_t n2) {
    Term natural = pauli_sum(A, n1 + n2);
    Term best = optimise(natural, n1, n2);
    const double norm = Eigen::JacobiSVD<Eigen::MatrixXcd>(A).singularValues()(0);
    const Cost& a = cost(natural);
    const Cost& b = cost(best);
    const double ca = a.alpha * double(a.k), cb = b.alpha * double(b.k);
    std::printf("%-34s %2zu q | Pauli: k=%4llu alpha=%7.2f  cost=%9.1f | ours: k=%3llu alpha=%7.2f  cost=%8.1f | %5.2fx | alpha/||T||=%.2f | %s\n",
                name, n1 + n2, (unsigned long long)a.k, a.alpha, ca, (unsigned long long)b.k, b.alpha, cb, ca / cb,
                b.alpha / norm, same_matrix(best, natural) ? "same matrix" : "MISMATCH");
}
}

int main() {
    std::printf("SCOPE: translation-invariant operators on periodic grids, written as Pauli sums; cost = queries x alpha.\n\n");
    for (std::size_t n : {3u, 4u, 5u, 6u}) {
        const std::string N = std::to_string(1 << n);
        row(("1D Laplacian, " + N + " pts").c_str(), circulant(n, {{-1, 1}, {0, -2}, {1, 1}}), n, 0);
    }
    for (std::size_t n : {3u, 4u, 5u, 6u}) {
        const std::string N = std::to_string(1 << n);
        row(("1D upwind advection, " + N + " pts").c_str(), circulant(n, {{0, 1}, {-1, -1}}), n, 0);
    }
    for (std::size_t n : {3u, 4u, 5u, 6u}) {
        const std::string N = std::to_string(1 << n);
        row(("1D biharmonic, " + N + " pts").c_str(), circulant(n, {{-2, 1}, {-1, -4}, {0, 6}, {1, -4}, {2, 1}}), n, 0);
    }
    for (std::size_t n : {2u, 3u}) {
        const std::string N = std::to_string(1 << n);
        const Eigen::MatrixXcd L = circulant(n, {{-1, 1}, {0, -2}, {1, 1}}), I = Eigen::MatrixXcd::Identity(1 << n, 1 << n);
        const Eigen::MatrixXcd D = circulant(n, {{0, 1}, {-1, -1}});
        row(("2D Laplacian, " + N + "x" + N).c_str(), kron(L, I) + kron(I, L), n, n);
        row(("2D convection-diffusion, " + N + "x" + N).c_str(), -0.1 * (kron(L, I) + kron(I, L)) + kron(I, D) + 0.5 * kron(D, I), n, n);
        row(("2D biharmonic-type (L x L), " + N + "x" + N).c_str(), kron(L, L), n, n);
        row(("laplacian-filter target, " + N + "x" + N).c_str(), kron(-4.7 * I + 0.1 * L, -1.9 * I + 3.9 * L), n, n);
    }
}
