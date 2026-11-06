// What the literature changes (Claude, 5 Nov 2026, evening). Two tables:
//  1. Periodic operators: Pauli form against our compiled form, with the hand-built shift LCU
//     (alpha = sum |c|: Camps et al.; Kharazi et al.; Mahmud & Herrman), the floor ||T||, the
//     certificate and the derivative each stencil approximates (Mahmud & Herrman, arXiv:2607.11596).
//  2. Boundary conditions: Pauli form, Kharazi et al.'s hand-built encodings (arXiv:2407.18347,
//     Eqs. 52 and 54, built here as terms), and our boundary route.
// Two cost columns: Cobble's queries x alpha, and alpha x Toffolis per call under the model of gates.h.
#include "symbol_route.h"
#include <cstdio>

using namespace bench;

namespace {
using Taps = std::vector<std::pair<long, double>>;

Eigen::MatrixXcd circulant(std::size_t n, const Taps& taps) {
    const Eigen::Index N = Eigen::Index(1) << n;
    Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(N, N);
    for (auto [s, w] : taps) T += w * cyclic_shift(N, s);
    return T;
}
Eigen::MatrixXcd cut(std::size_t n, const Taps& taps) {
    const Eigen::Index N = Eigen::Index(1) << n;
    Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(N, N);
    for (auto [s, w] : taps)
        for (Eigen::Index x = 0; x < N; ++x)
            if (x + s >= 0 && x + s < N) T(x, x + s) += w;
    return T;
}
double norm2(const Eigen::MatrixXcd& A) { return Eigen::JacobiSVD<Eigen::MatrixXcd>(A).singularValues()(0); }

struct Col { double k, alpha, queries_cost, toffoli, gate_cost; };
Col col(const Term& t) {
    const Cost& c = cost(t);
    const double tof = gates::per_call(t);
    return {double(c.k), c.alpha, c.alpha * double(c.k), tof, c.alpha * tof};
}

void periodic(const char* name, const Eigen::MatrixXcd& A, std::size_t n1, std::size_t n2) {
    Term pauli = pauli_sum_sparse(A, n1 + n2);
    Term ours = optimise(pauli, n1, n2);
    SymbolResult r = symbol_route(pauli, n1, n2);
    const Col p = col(pauli), o = col(ours);
    std::printf("%-30s %2zu q | Pauli k=%3.0f a=%6.2f  a*k=%7.1f  a*Tof=%8.1f | ours k=%2.0f a=%6.2f  a*k=%6.1f  a*Tof=%7.1f |"
                " x%5.1f (q) x%5.1f (Tof) | hand-built a=%6.2f | ||T||=%6.2f %s | %s | %s\n",
                name, n1 + n2, p.k, p.alpha, p.queries_cost, p.gate_cost, o.k, o.alpha, o.queries_cost, o.gate_cost,
                p.queries_cost / o.queries_cost, p.gate_cost / o.gate_cost, r.coeffs.cwiseAbs().sum(), r.norm,
                r.certified ? "certified" : "NOT certified", r.moments.describe(n2 > 0).c_str(),
                same_matrix(ours, pauli) ? "same matrix" : "MISMATCH");
}

void boundary(const char* name, const Eigen::MatrixXcd& A, std::size_t n1, std::size_t n2, const Term& hand, double hand_alpha) {
    Term pauli = pauli_sum_sparse(A, n1 + n2);
    Term ours = optimise(pauli, n1, n2);
    const Col p = col(pauli), o = col(ours);
    char h[96];
    if (hand) {
        const Col c = col(hand);
        std::snprintf(h, sizeof h, "Kharazi k=%2.0f a=%5.2f a*k=%6.1f a*Tof=%7.1f", c.k, c.alpha, c.queries_cost, c.gate_cost);
    } else if (hand_alpha > 0) {
        std::snprintf(h, sizeof h, "Kharazi a=%5.2f (from the paper)", hand_alpha);
    } else {
        std::snprintf(h, sizeof h, "no hand-built encoding of this matrix found");
    }
    std::printf("%-34s %2zu q | Pauli k=%4.0f a=%6.2f  a*k=%8.1f  a*Tof=%9.1f | %-50s | ours k=%2.0f a=%5.2f  a*k=%6.1f  a*Tof=%7.1f |"
                " x%6.1f (q) x%6.1f (Tof) | ||T||=%5.2f | %s\n",
                name, n1 + n2, p.k, p.alpha, p.queries_cost, p.gate_cost, h, o.k, o.alpha, o.queries_cost, o.gate_cost,
                p.queries_cost / o.queries_cost, p.gate_cost / o.gate_cost, norm2(A),
                same_matrix(ours, pauli) ? "same matrix" : "MISMATCH");
}

// Kharazi et al., Eqs. 52 and 54, as terms (their S is ours: (S f)(x) = f(x + 1)).
Term kharazi(std::size_t n, bool neumann) {
    const Eigen::Index N = Eigen::Index(1) << n;
    auto R = [&](Eigen::Index p) { Eigen::MatrixXcd M = Eigen::MatrixXcd::Identity(N, N); M(p, p) = -1.0; return M; };
    auto reg = [&](const Term& u, double tof) { gates::registry()[u->id] = tof; return u; };
    const double refl = double(n) - 1;                                   // C^n Z on one grid point
    Term I = leaf(PauliString::from_string(std::string(n, 'I')));
    Term S = reg(unitary_leaf("S", cyclic_shift(N, 1)), gates::controlled_shift(n, 1));
    Term Si = reg(unitary_leaf("S^-1", cyclic_shift(N, -1)), gates::controlled_shift(n, -1));
    Term R0Si = reg(unitary_leaf("R0 S^-1", R(0) * cyclic_shift(N, -1)), refl + gates::controlled_shift(n, -1));
    Term RNS = reg(unitary_leaf("R_(N-1) S", R(N - 1) * cyclic_shift(N, 1)), refl + gates::controlled_shift(n, 1));
    if (!neumann) return sum({-2.0, 0.5, 0.5, 0.5, 0.5}, {I, Si, R0Si, S, RNS});
    Term R0RN = reg(unitary_leaf("R0 R_(N-1)", R(0) * R(N - 1)), 2 * refl);
    return sum({-1.5, -0.5, 0.5, 0.5, 0.5, 0.5}, {I, R0RN, R0Si, Si, RNS, S});
}
}  // namespace

int main() {
    std::printf("SCOPE: constant-coefficient stencils, 1D grids up to 64 points (6 qubits), 2D up to 16x16 (8 qubits). a = alpha.\n"
                "Cost 1 = queries x alpha (Cobble). Cost 2 = alpha x Toffolis per call: SELECT only, PREPARE not counted\n"
                "(that favours the Pauli form). Toffoli model and sources: bench/gates.h. Estimates, not compiled circuits.\n\n");
    std::printf("1. Periodic boundaries\n");
    const Taps lap{{-1, 1}, {0, -2}, {1, 1}}, adv{{0, 1}, {-1, -1}}, bih{{-2, 1}, {-1, -4}, {0, 6}, {1, -4}, {2, 1}};
    for (std::size_t n : {3u, 4u, 5u, 6u})
        for (auto [label, taps] : std::vector<std::pair<const char*, Taps>>{{"1D Laplacian", lap}, {"1D upwind advection", adv}, {"1D biharmonic", bih}})
            periodic((std::string(label) + ", " + std::to_string(1 << n) + " pts").c_str(), circulant(n, taps), n, 0);
    for (std::size_t n : {2u, 3u, 4u}) {
        const std::string N = std::to_string(1 << n);
        const Eigen::MatrixXcd L = circulant(n, lap), I = Eigen::MatrixXcd::Identity(1 << n, 1 << n), D = circulant(n, adv);
        periodic(("2D Laplacian, " + N + "x" + N).c_str(), kron_hi(L, I) + kron_hi(I, L), n, n);
        periodic(("2D convection-diffusion, " + N + "x" + N).c_str(), -0.1 * (kron_hi(L, I) + kron_hi(I, L)) + kron_hi(I, D) + 0.5 * kron_hi(D, I), n, n);
        periodic(("L x L, " + N + "x" + N).c_str(), kron_hi(L, L), n, n);
        periodic(("laplacian-filter target, " + N + "x" + N).c_str(), kron_hi(-4.7 * I + 0.1 * L, -1.9 * I + 3.9 * L), n, n);
    }

    std::printf("\n2. Other boundary conditions\n");
    const Taps f4{{-2, -1.0 / 12}, {-1, 16.0 / 12}, {0, -30.0 / 12}, {1, 16.0 / 12}, {2, -1.0 / 12}};
    for (std::size_t n : {3u, 4u, 5u, 6u}) {
        const std::string N = std::to_string(1 << n);
        const Eigen::Index M = Eigen::Index(1) << n;
        Eigen::MatrixXcd neu = second_difference(n, false), mixed = second_difference(n, false);
        neu(0, 0) = neu(M - 1, M - 1) = -1.0;
        mixed(0, 0) = -1.0;
        boundary(("1D Laplacian, Dirichlet, " + N + " pts").c_str(), second_difference(n, false), n, 0, kharazi(n, false), 0);
        boundary(("1D Laplacian, Neumann, " + N + " pts").c_str(), neu, n, 0, kharazi(n, true), 0);
        boundary(("1D Laplacian, Neumann|Dirichlet, " + N + " pts").c_str(), mixed, n, 0, nullptr, 0);
        boundary(("1D 4th-order d^2, Dirichlet, " + N + " pts").c_str(), cut(n, f4), n, 0, nullptr, 0);
    }
    for (std::size_t n : {2u, 3u, 4u}) {
        const std::string N = std::to_string(1 << n);
        const Eigen::MatrixXcd L = second_difference(n, false), I = Eigen::MatrixXcd::Identity(1 << n, 1 << n);
        boundary(("2D Laplacian, Dirichlet, " + N + "x" + N).c_str(), kron_hi(L, I) + kron_hi(I, L), n, n, nullptr, 8.0);
    }
}
