#pragma once
// Cobble's benchmarks rebuilt with GEO_Q.h (written by Claude, 5 Nov 2026).
// Sources: cobble/examples/*.py in Cobble's artifact (Zenodo 18898778); tables tex/tbl3.tex, tbl4.tex.
//
// Cobble's black boxes Basic("$X") are Hermitian unitaries with alpha = 1 (expr.py: hermitian = True).
// For costs, any Hermitian Pauli leaf behaves the same way (k = 1, alpha = 1, m = 0), so `box(n)`
// stands in for a black box on n qubits. For meaning checks, concrete matrices are used instead.
#include "GEO_Q.h"
#include <cmath>
#include <map>
#include <numbers>
#include <string>

namespace bench {
using namespace cobble;

inline Term box(std::size_t n = 1) { return leaf(PauliString::from_string(std::string(n, 'Z'))); }

// ---- laplacian-filter (cobble/examples/laplacian_filter.py) ------------------------------------
// prog = poly(Hx,f)(x)poly(Hy,p) + poly(Hx,g)(x)poly(Hy,p) + poly(Hx,f)(x)poly(Hy,q) + poly(Hx,g)(x)poly(Hy,q)
// targ = poly(Hx, f+g) (x) poly(Hy, p+q)
struct Pair { Term prog, targ; };

inline Pair laplacian_filter(const Term& Hx, const Term& Hy) {
    const std::vector<double> f{1.0, 0.5}, g{-2.9, 3.4}, p{1.1, 0.4}, q{-5.8, -0.3};
    auto kp = [&](const std::vector<double>& a, const std::vector<double>& b) { return tensor({poly(Hx, a), poly(Hy, b)}); };
    Term prog = sum({1, 1, 1, 1}, {kp(f, p), kp(g, p), kp(f, q), kp(g, q)});
    Term targ = tensor({poly(Hx, {-1.9, 3.9}), poly(Hy, {-4.7, 0.1})});
    return {prog, targ};
}

// The 1D second difference (1, -2, 1) on 2^n points, i.e. Delta+ Delta- (the "separable 2D stencil"
// of the benchmark's docstring, one axis at a time). Periodic by default (decision of 5 Nov): only
// then do the forward and backward differences commute, so that the discrete Dirac operator of
// Faustino, Kahler and Sommen (2007) squares to it exactly. periodic = false gives zero boundary values.
inline Eigen::MatrixXcd second_difference(std::size_t n, bool periodic = true) {
    const Eigen::Index N = Eigen::Index(1) << n;
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(N, N);
    for (Eigen::Index i = 0; i < N; ++i) {
        A(i, i) = -2.0;
        if (i + 1 < N) A(i, i + 1) = A(i + 1, i) = 1.0;
    }
    if (periodic) { A(0, N - 1) += 1.0; A(N - 1, 0) += 1.0; }
    return A;
}

// A matrix as a sum of Pauli strings: c_P = Tr(P^dagger A) / 2^n, over all 4^n strings
// (the n-qubit Pauli strings are an orthogonal basis of all 2^n x 2^n matrices).
inline Term pauli_sum(const Eigen::MatrixXcd& A, std::size_t n) {
    std::vector<Coeff> c;
    std::vector<Term> t;
    const char letters[4] = {'I', 'X', 'Y', 'Z'};
    for (std::uint64_t code = 0; code < (std::uint64_t(1) << (2 * n)); ++code) {
        std::string s(n, 'I');
        for (std::size_t j = 0; j < n; ++j) s[j] = letters[(code >> (2 * j)) & 3];
        const PauliString P = PauliString::from_string(s);
        const Coeff v = (matrix(P).adjoint() * A).trace() / double(std::uint64_t(1) << n);
        if (std::abs(v) > 1e-14) { c.push_back(v); t.push_back(leaf(P)); }
    }
    return sum(c, t);
}

// The same decomposition from the nonzero entries only (Claude, 5 Nov, evening), for sparse matrices on
// up to 8+ qubits: with P = i^k X^x Z^z (pauli.h), Tr(P^dagger A) = conj(i^k) sum_col (-1)^|z & col| A(col ^ x, col),
// so each nonzero entry (r, col) feeds only x = r ^ col. k = |x & z| mod 4 gives the letter string (Y = i X Z).
inline Term pauli_sum_sparse(const Eigen::MatrixXcd& A, std::size_t n) {
    const std::uint64_t D = std::uint64_t(1) << n;
    std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, Coeff>>> by_x;
    for (std::uint64_t c = 0; c < D; ++c)
        for (std::uint64_t r = 0; r < D; ++r)
            if (A(Eigen::Index(r), Eigen::Index(c)) != Coeff(0.0)) by_x[r ^ c].emplace_back(c, A(Eigen::Index(r), Eigen::Index(c)));
    const Coeff ik_conj[4] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};
    std::vector<Coeff> cs;
    std::vector<Term> ts;
    for (const auto& [x, entries] : by_x)
        for (std::uint64_t z = 0; z < D; ++z) {
            Coeff s = 0.0;
            for (const auto& [c, v] : entries) s += (std::popcount(z & c) % 2 ? -1.0 : 1.0) * v;
            PauliString P;
            P.n = n; P.x = x; P.z = z; P.k = unsigned(std::popcount(x & z)) % 4u;
            const Coeff coef = ik_conj[P.k] * s / double(D);
            if (std::abs(coef) > 1e-14) { cs.push_back(coef); ts.push_back(leaf(P)); }
        }
    return sum(cs, ts);
}

// ---- spectral-thresholding (cobble/examples/spectral_thresholding.py) --------------------------
// As written: (4/pi) sum_{i<10} T_{2i+1}/(2i+1). Corrected: with (-1)^i, the Chebyshev series of
// sign(x) (Brown AM34 Chebyshev page, "other examples"; Example 5A there drops the (-1)^i).
inline Term spectral_thresholding(const Term& X, bool corrected) {
    std::vector<double> c(20, 0.0);
    for (int i = 0; i < 10; ++i) c[std::size_t(2 * i + 1)] = ((corrected && i % 2) ? -1.0 : 1.0) * (4 / std::numbers::pi) / (2 * i + 1);
    return poly(X, polymax::from_chebyshev(c));
}

// ---- hamiltonian-simulation with real_phase = True (cobble/examples/hamiltonian_simulation.py) --
// cos part J_0(t) + 2 sum_{k=1..7} (-1)^k J_{2k}(t) T_{2k}; sin part 2 sum_{k=0..7} (-1)^k J_{2k+1}(t) T_{2k+1}; t = 7.
inline Term hamiltonian_simulation(const Term& X) {
    const double t = 7.0;
    std::vector<double> c(16, 0.0);
    c[0] = std::cyl_bessel_j(0.0, t);
    for (int k = 1; k <= 7; ++k) c[std::size_t(2 * k)] = 2 * ((k % 2) ? -1.0 : 1.0) * std::cyl_bessel_j(2.0 * k, t);
    for (int k = 0; k <= 7; ++k) c[std::size_t(2 * k + 1)] = 2 * ((k % 2) ? -1.0 : 1.0) * std::cyl_bessel_j(2.0 * k + 1, t);
    return poly(X, polymax::from_chebyshev(c));
}

inline const char* method_name(PolyMethod m) {
    static const char* names[6] = {"LCU", "Horner", "QSVT", "GQET", "Ko", "Truncated"};
    return names[int(m)];
}

}  // namespace bench
