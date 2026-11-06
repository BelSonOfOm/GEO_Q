#pragma once
// The symbol route: a first numerical version of the hunch (written by Claude, 5 Nov 2026, with
// choices made by Claude at Charbel's request for the Alice & Bob demo; the mathematical
// formulation is still Charbel's to write).
//
// Idea (Charbel, 5 Nov): read a term in a "fictitious" continuous space, simplify there, translate
// back into a cheaper term with the same matrix, like a function and its Fourier series.
//
// This version handles terms whose matrix commutes with the translations of a periodic grid
// Z_N1 x Z_N2 (qubits of axis 1 below those of axis 2, as in tensor({axis1, axis2})).
// 1. Such a matrix T is a sum of shifts: T = sum_{a,b} c_ab S1^a S2^b (circulant in each axis;
//    Gray, "Toeplitz and Circulant Matrices: A Review", 2006). The c_ab are the Fourier
//    coefficients of the symbol of T, read off from the first column of T.
// 2. Translate back: each shift is a unitary leaf (alpha = 1); if the coefficient table c_ab has
//    rank one, T = A (x) B and the term is tensor(sum over a, sum over b), otherwise a sum of tensors.
//    The cheaper of the two (alpha * k) is returned.
// 3. Floor: any block encoding has alpha >= ||T|| (Cobble, Def. 2.2), so alpha / ||T|| measures how
//    far the subnormalisation is from the best possible.
// Added the evening of 5 Nov, from the literature: the certificate and the moment order from the
// coefficients alone (Mahmud & Herrman, arXiv:2607.11596), the boundary route (after Kharazi et al.,
// arXiv:2407.18347), and Toffoli estimates for the leaves (gates.h).
#include "benchmarks.h"
#include "gates.h"
#include "rewrite.h"
#include <map>
#include <sstream>

namespace bench {

// The cyclic shift on N points raised to the power a: (S^a f)(x) = f(x + a mod N).
inline Eigen::MatrixXcd cyclic_shift(Eigen::Index N, long a) {
    Eigen::MatrixXcd S = Eigen::MatrixXcd::Zero(N, N);
    for (Eigen::Index x = 0; x < N; ++x) S(x, ((x + a) % N + N) % N) = 1.0;
    return S;
}

inline Eigen::MatrixXcd dense_of(const Term& t) {
    const Eigen::Index D = Eigen::Index(1) << t->qubits;
    Eigen::MatrixXcd M(D, D);
    for (Eigen::Index j = 0; j < D; ++j) M.col(j) = act(t, Eigen::VectorXcd::Unit(D, j));
    return M;
}

inline Eigen::MatrixXcd kron_hi(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) {   // A on the high qubits
    Eigen::MatrixXcd K(A.rows() * B.rows(), A.cols() * B.cols());
    for (Eigen::Index i = 0; i < A.rows(); ++i)
        for (Eigen::Index j = 0; j < A.cols(); ++j) K.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
    return K;
}

// S1^a S2^b on the grid Z_N1 x Z_N2 (axis 1 on the low n1 qubits; n2 = 0 for one axis).
inline Eigen::MatrixXcd grid_shift(std::size_t n1, std::size_t n2, long a, long b) {
    return kron_hi(cyclic_shift(Eigen::Index(1) << n2, b), cyclic_shift(Eigen::Index(1) << n1, a));
}

// The shift as one unitary leaf (5 Nov, evening: a 2D shift counts as one query, as a Pauli string
// does; before, it was a tensor of two leaves and counted as two). The identity is a Pauli leaf.
// Its controlled Toffoli cost goes into gates::registry().
inline Term shift_term(std::size_t n1, std::size_t n2, long a, long b) {
    const long N1 = long(1) << n1, N2 = long(1) << n2;
    if (((a % N1) + N1) % N1 == 0 && ((b % N2) + N2) % N2 == 0) return leaf(PauliString::from_string(std::string(n1 + n2, 'I')));
    Term u = unitary_leaf("S^(" + std::to_string(a) + "," + std::to_string(b) + ")", grid_shift(n1, n2, a, b));
    gates::registry()[u->id] = gates::controlled_shift(n1, a) + (n2 ? gates::controlled_shift(n2, b) : 0.0);
    return u;
}

// Representative of a in (-N/2, N/2]: the offset a stencil coefficient sits at.
inline long centred(long a, long N) {
    long r = ((a % N) + N) % N;
    return r > N / 2 ? r - N : r;
}

// ||T|| from the coefficients alone: T = sum c_ab S1^a S2^b is diagonalised by the Fourier basis,
// S^a e_k = w^(ka) e_k (Gray 2006, circulant matrices), so its eigenvalues are
// lambda(k1, k2) = sum c_ab exp(2 pi i (a k1 / N1 + b k2 / N2)) and ||T|| = max |lambda| (T is normal).
inline double symbol_norm(const Eigen::MatrixXcd& C) {
    const Eigen::Index N1 = C.rows(), N2 = C.cols();
    std::vector<std::tuple<long, long, Coeff>> nz;
    for (Eigen::Index a = 0; a < N1; ++a)
        for (Eigen::Index b = 0; b < N2; ++b)
            if (C(a, b) != Coeff(0.0)) nz.emplace_back(long(a), long(b), C(a, b));
    double best = 0.0;
    for (Eigen::Index k1 = 0; k1 < N1; ++k1)
        for (Eigen::Index k2 = 0; k2 < N2; ++k2) {
            Coeff l = 0.0;
            for (auto [a, b, c] : nz)
                l += c * std::polar(1.0, 2 * std::numbers::pi * (double(a * k1) / double(N1) + double(b * k2) / double(N2)));
            best = std::max(best, std::abs(l));
        }
    return best;
}

// Moment order (Mahmud & Herrman, arXiv:2607.11596): the lowest total degree j at which a moment
// mu_(j1,j2) = sum c_ab a^j1 b^j2 is nonzero, offsets taken in (-N/2, N/2]. By Taylor's formula,
// (S1^a S2^b f)(x) = f(x + a h e1 + b h e2), so T ~ h^m sum_(j1 + j2 = m) mu_(j1,j2) / (j1! j2!) d1^j1 d2^j2:
// the stencil approximates an order-m derivative.
struct Moments {
    int order = -1;                                      // -1: none nonzero up to degree 8
    std::vector<std::tuple<int, int, Coeff>> leading;    // (j1, j2, mu / (j1! j2!))
    std::string describe(bool two_axes) const {
        if (order < 0) return "no nonzero moment up to degree 8";
        std::ostringstream o;
        o << "order " << order << ":";
        bool first = true;
        for (auto [j1, j2, c] : leading) {
            if (std::abs(c.imag()) <= 1e-12 * std::abs(c)) {
                const double v = c.real();
                o << (first ? (v < 0 ? " -" : " ") : (v < 0 ? " - " : " + ")) << std::abs(v);
            } else {
                o << (first ? " " : " + ") << "(" << c.real() << (c.imag() < 0 ? "-" : "+") << std::abs(c.imag()) << "i)";
            }
            auto d = [&](const char* name, int j) { if (j) o << " " << name << (j > 1 ? "^" + std::to_string(j) : ""); };
            d(two_axes ? "d1" : "d", j1);
            d("d2", j2);
            first = false;
        }
        return o.str();
    }
};

inline Moments moments(const Eigen::MatrixXcd& C) {
    Moments M;
    const long N1 = long(C.rows()), N2 = long(C.cols());
    double scale = 0.0;
    for (long a = 0; a < N1; ++a)
        for (long b = 0; b < N2; ++b) scale += std::abs(C(a, b));
    auto fact = [](int j) { double f = 1; for (int i = 2; i <= j; ++i) f *= i; return f; };
    for (int j = 0; j <= 8; ++j) {
        for (int j1 = j; j1 >= 0; --j1) {
            const int j2 = j - j1;
            if (N2 == 1 && j2 > 0) continue;
            Coeff mu = 0.0;
            double size = 0.0;
            for (long a = 0; a < N1; ++a)
                for (long b = 0; b < N2; ++b) {
                    if (C(a, b) == Coeff(0.0)) continue;
                    const double w = std::pow(double(centred(a, N1)), j1) * std::pow(double(centred(b, N2)), j2);
                    mu += C(a, b) * w;
                    size += std::abs(C(a, b)) * std::abs(w);
                }
            if (std::abs(mu) > 1e-9 * std::max(size, scale)) M.leading.emplace_back(j1, j2, mu / (fact(j1) * fact(j2)));
        }
        if (!M.leading.empty()) { M.order = j; return M; }
    }
    return M;
}

struct SymbolResult {
    bool translation_invariant = false;
    Term term;                 // the translated-back term (null if not translation-invariant)
    double norm = 0.0;         // ||T||, the floor for alpha (dense SVD)
    std::size_t shifts = 0;    // number of nonzero Fourier coefficients
    bool separable = false;
    Eigen::MatrixXcd coeffs;   // c_ab, a on axis 1, b on axis 2
    double symbol_norm = 0.0;  // ||T|| from the coefficients alone
    bool certified = false;    // sum |c_ab| = ||T||: the shift LCU is optimal (Mahmud & Herrman, Prop. V.4)
    Moments moments;
};

// T acts on n1 + n2 qubits: axis 1 on the low n1 qubits, axis 2 on the high n2 qubits.
inline SymbolResult symbol_route(const Term& t, std::size_t n1, std::size_t n2) {
    SymbolResult r;
    detail::require(t->qubits == n1 + n2, "symbol_route: qubit counts do not add up.");
    const Eigen::Index N1 = Eigen::Index(1) << n1, N2 = Eigen::Index(1) << n2;
    const Eigen::MatrixXcd T = dense_of(t);
    const Eigen::MatrixXcd X1 = grid_shift(n1, n2, 1, 0), X2 = grid_shift(n1, n2, 0, 1);
    const double scale = std::max(T.norm(), 1e-300);
    r.translation_invariant = (T * X1 - X1 * T).norm() <= 1e-10 * scale && (T * X2 - X2 * T).norm() <= 1e-10 * scale;
    r.norm = Eigen::JacobiSVD<Eigen::MatrixXcd>(T).singularValues()(0);
    if (!r.translation_invariant) return r;

    // Fourier coefficients: T = sum c_ab S1^a S2^b gives (T delta_0)(x) = c_{-x}.
    Eigen::MatrixXcd C(N1, N2);
    for (Eigen::Index a = 0; a < N1; ++a)
        for (Eigen::Index b = 0; b < N2; ++b) C(a, b) = T(((N1 - a) % N1) + N1 * ((N2 - b) % N2), 0);
    const double cut = 1e-12 * std::max(C.cwiseAbs().maxCoeff(), 1e-300);
    for (Eigen::Index a = 0; a < N1; ++a)
        for (Eigen::Index b = 0; b < N2; ++b)
            if (std::abs(C(a, b)) <= cut) C(a, b) = 0.0;
    r.coeffs = C;
    r.symbol_norm = symbol_norm(C);
    r.certified = std::abs(r.symbol_norm - C.cwiseAbs().sum()) <= 1e-9 * C.cwiseAbs().sum();
    r.moments = moments(C);

    // All nonzero (a, b): T = sum c_ab S1^a S2^b, each shift one leaf (one axis: b = 0 throughout).
    std::vector<Coeff> coeffs;
    std::vector<Term> terms;
    for (Eigen::Index a = 0; a < N1; ++a)
        for (Eigen::Index b = 0; b < N2; ++b)
            if (C(a, b) != Coeff(0.0)) {
                coeffs.push_back(C(a, b));
                terms.push_back(shift_term(n1, n2, long(a), long(b)));
            }
    r.shifts = coeffs.size();
    Term best = sum(coeffs, terms);
    if (n2 == 0) { r.separable = true; r.term = best; return r; }

    // Separable: the coefficient table has rank one, C = u v^T, so T = (sum_a u_a S1^a) (x) (sum_b v_b S2^b).
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& sv = svd.singularValues();
    if (sv.size() < 2 || sv(1) <= 1e-12 * sv(0)) {
        const Eigen::VectorXcd u = svd.matrixU().col(0) * sv(0), v = svd.matrixV().col(0).conjugate();
        std::vector<Coeff> cu, cv;
        std::vector<Term> tu, tv;
        for (Eigen::Index a = 0; a < N1; ++a)
            if (std::abs(u(a)) > 1e-12 * u.cwiseAbs().maxCoeff()) { cu.push_back(u(a)); tu.push_back(shift_term(n1, 0, long(a), 0)); }
        for (Eigen::Index b = 0; b < N2; ++b)
            if (std::abs(v(b)) > 1e-12 * v.cwiseAbs().maxCoeff()) { cv.push_back(v(b)); tv.push_back(shift_term(n2, 0, long(b), 0)); }
        Term sep = tensor({sum(cu, tu), sum(cv, tv)});
        const auto total = [](const Term& x) { return cost(x).alpha * double(cost(x).k); };
        if (total(sep) < total(best)) { best = sep; r.separable = true; }
    }
    r.term = best;
    return r;
}

// ---- The boundary route (Dirichlet, Neumann, Robin, mixed) ---------------------------------------
// Every matrix on the grid is a sum over shifts with diagonal weights, T = sum_g D_g S^g, where
// D_g(x, x) = T(x, x + g): the shifts of the grid partition the entries of a matrix. A translation-
// invariant T has constant D_g (the symbol route); a stencil with boundary conditions has D_g
// constant except near the edges. Each D_g is written with reflections, after Kharazi et al.
// (arXiv:2407.18347, Eqs. 52 and 54): with a base value b and P_v the projector on the points where
// D_g takes the value v,
//     D_g = b I + sum_(v != b) (v - b) P_v = (b + sum (v - b) / 2) I - sum ((v - b) / 2) R_v,
// where R_v = I - 2 P_v is unitary. Kharazi et al.'s (R_0 + I) / 2 cutting the wrap-around of a
// shift is the case v = 0; their R_0 R_(N-1) term is the case of one value at both ends. Writing it
// for any stencil, any boundary pattern and two axes, and choosing b automatically, is ours: b is the
// value (or 0) giving the smallest alpha for that D_g, ties to fewer leaves, then to fewer Toffolis in
// the reflections (gates.h; e.g. R_0 rather than R on all the other points). Each R_v S^g is one
// unitary leaf. Dense construction: at most 8 qubits and 256 leaves.
struct DiagonalResult { Term term; std::size_t diagonals = 0, reflections = 0; };

inline DiagonalResult diagonal_route(const Term& t, std::size_t n1, std::size_t n2) {
    DiagonalResult r;
    if (t->qubits != n1 + n2 || t->qubits > 8) return r;
    const Eigen::Index N1 = Eigen::Index(1) << n1, N2 = Eigen::Index(1) << n2, D = N1 * N2;
    const Eigen::MatrixXcd T = dense_of(t);
    const double tol = 1e-12 * std::max(T.cwiseAbs().maxCoeff(), 1e-300);

    struct Plan { long a, b; Coeff base, coefI; std::vector<Coeff> vals; std::vector<int> which; };
    std::vector<Plan> plans;
    std::size_t leaves = 0;
    for (Eigen::Index a = 0; a < N1; ++a)
        for (Eigen::Index b = 0; b < N2; ++b) {
            Plan p{long(a), long(b), 0.0, 0.0, {}, std::vector<int>(std::size_t(D))};
            bool nonzero = false;
            for (Eigen::Index x = 0; x < D; ++x) {
                const Eigen::Index y = (x % N1 + a) % N1 + N1 * ((x / N1 + b) % N2);
                const Coeff v = T(x, y);
                nonzero = nonzero || std::abs(v) > tol;
                std::size_t j = 0;
                while (j < p.vals.size() && std::abs(p.vals[j] - v) > tol) ++j;
                if (j == p.vals.size()) p.vals.push_back(v);
                p.which[std::size_t(x)] = int(j);
            }
            if (!nonzero) continue;
            std::vector<double> gate(p.vals.size(), 0.0);   // Toffolis of each value's controlled reflection
            for (std::size_t j = 0; j < p.vals.size(); ++j) {
                std::vector<bool> in(std::size_t(D), false);
                for (Eigen::Index x = 0; x < D; ++x) in[std::size_t(x)] = p.which[std::size_t(x)] == int(j);
                gate[j] = gates::controlled_reflection(in);
            }
            std::vector<Coeff> bases = p.vals;
            bases.push_back(0.0);
            double best_alpha = 1e300, best_gate = 1e300;
            std::size_t best_leaves = 0;
            for (const Coeff& base : bases) {
                Coeff cI = base;
                double al = 0.0, gt = 0.0;
                std::size_t lv = 0;
                for (std::size_t j = 0; j < p.vals.size(); ++j)
                    if (std::abs(p.vals[j] - base) > tol) { cI += (p.vals[j] - base) / 2.0; al += std::abs(p.vals[j] - base) / 2; gt += gate[j]; ++lv; }
                if (std::abs(cI) > tol) ++lv;
                al += std::abs(cI);
                const bool same_alpha = std::abs(al - best_alpha) <= 1e-12 * std::max(al, best_alpha);
                if ((!same_alpha && al < best_alpha) || (same_alpha && (lv < best_leaves || (lv == best_leaves && gt < best_gate)))) {
                    best_alpha = al; best_leaves = lv; best_gate = gt; p.base = base; p.coefI = cI;
                }
            }
            leaves += best_leaves;
            plans.push_back(std::move(p));
        }
    if (plans.empty() || leaves > 256) return r;

    std::vector<Coeff> cs;
    std::vector<Term> ts;
    for (const Plan& p : plans) {
        if (std::abs(p.coefI) > tol) { cs.push_back(p.coefI); ts.push_back(shift_term(n1, n2, p.a, p.b)); }
        for (std::size_t j = 0; j < p.vals.size(); ++j) {
            if (std::abs(p.vals[j] - p.base) <= tol) continue;
            std::vector<bool> in(std::size_t(D), false);
            Eigen::MatrixXcd R = Eigen::MatrixXcd::Identity(D, D);
            for (Eigen::Index x = 0; x < D; ++x)
                if (p.which[std::size_t(x)] == int(j)) { in[std::size_t(x)] = true; R(x, x) = -1.0; }
            Term u = unitary_leaf("R S^(" + std::to_string(p.a) + "," + std::to_string(p.b) + ")", R * grid_shift(n1, n2, p.a, p.b));
            gates::registry()[u->id] = gates::controlled_reflection(in) + gates::controlled_shift(n1, p.a) +
                                       (n2 ? gates::controlled_shift(n2, p.b) : 0.0);
            cs.push_back(-(p.vals[j] - p.base) / 2.0);
            ts.push_back(u);
            ++r.reflections;
        }
        ++r.diagonals;
    }
    r.term = sum(cs, ts);
    return r;
}

// Never worse than the usual form (Charbel, 5 Nov; the same guarantee Cobble gives for its polynomial
// methods, Sec. 6.2): among candidate terms that are the same matrix as the first one, return the
// cheapest by alpha * k, ties to fewer workspace qubits, then to the earlier candidate.
inline Term cheapest(const std::vector<Term>& candidates) {
    detail::require(!candidates.empty(), "cheapest: no candidates.");
    Term best = candidates[0];
    for (std::size_t j = 1; j < candidates.size(); ++j) {
        const Term& c = candidates[j];
        if (!c) continue;
        detail::require(same_matrix(c, candidates[0]), "cheapest: a candidate is not the same matrix.");
        const double a = cost(c).alpha * double(cost(c).k), b = cost(best).alpha * double(cost(best).k);
        if (a < b || (a == b && cost(c).m < cost(best).m)) best = c;
    }
    return best;
}

// The cheapest of: the usual form, Cobble's rewrites of it (GEO_Q/rewrite.h), the symbol route and the
// boundary route (the routes need the dense matrix, so only up to 12 qubits).
inline Term optimise(const Term& usual, std::size_t n1, std::size_t n2) {
    const Term rewritten = cobble::optimize(usual);
    if (usual->qubits != n1 + n2 || usual->qubits > 12 || parameter_count(usual) > 0) return rewritten;
    return cheapest({usual, rewritten, symbol_route(usual, n1, n2).term, diagonal_route(usual, n1, n2).term});
}

}  // namespace bench
