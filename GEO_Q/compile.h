#pragma once
// Code generation: GEO_Q terms -> gate circuits that run on the simulator of Basics.h
// (written by Claude, 6 Nov 2026, at Charbel's request: "do all the coding").
//
// What a compiled circuit means. A term t on n data qubits compiles to a circuit C on n + m qubits
// (data qubits 0..n-1, workspace qubits n..n+m-1) and a number alpha such that, with the workspace
// started and ended in |0...0>,
//        alpha * (<0^m| (x) I) C (|0^m> (x) I) = [[t]]                    (Cobble, Def. 2.2).
// Each node is compiled to a standalone circuit; a parent embeds its children by renaming qubits and
// adding controls (controlling every gate of a circuit controls the whole circuit).
//
// Constructions and their sources:
//   Pauli leaf        one-qubit gates X, Y, Z, plus the string's phase.
//   Unitary leaf      if it is a shift on a grid times a diagonal of signs (the leaves of the symbol and
//                     boundary routes), increments/decrements made of multi-controlled X gates and
//                     multi-controlled Z gates (one per subcube, bench/gates.h); otherwise one dense gate.
//   Rotation          exp(-i theta P / 2) on the qubits where P acts, angle given at run time.
//   Sum               PREPARE - SELECT - PREPARE^dagger (linear combination of unitaries; Childs & Wiebe
//                     2012; Gilyen et al. 2019, Lemma 52), PREPARE as a tree of controlled R_y rotations
//                     (Mottonen et al. 2004, nonnegative amplitudes). Complex coefficients: their phases go
//                     into SELECT.
//   Product           factors one after the other on a shared workspace, plus a counter of ceil(log2 n)
//                     qubits counting the factors that succeeded; at the end the counter is moved back to 0
//                     exactly when all did. Workspace ceil(log2 n) + max m_j, Cobble's Table 1.
//   Tensor            factors side by side, each with its own workspace.
//   Direct sum        each block controlled on the selector qubits; a block whose alpha is smaller gets a
//                     one-qubit R_y with top-left entry alpha_j / alpha_max (arXiv:2403.19512, App. B).
//   Adjoint           the circuit reversed, every gate replaced by its inverse.
//   Poly              QSVT (Gilyen, Su, Low & Wiebe 2019, Thm. 17 and Cor. 18) on a Hermitian base: the even
//                     and odd parts separately, each made real with one extra qubit (the H - controlled phase
//                     sign - H construction of Cor. 18), the two parts joined by one more qubit. QSP phases are
//                     found numerically here (Levenberg-Marquardt on the 2 x 2 model, many random starts;
//                     the optimisation approach of Dong, Meng, Whaley & Lin, PRA 103, 042419, 2021, in the
//                     reflection convention). On a non-Hermitian base: the sum of powers a_0 I + a_1 M + ...
//                     compiled as a Sum of Products (Cobble's LCU method). When the cost model chose GQET, Ko
//                     or Horner, QSVT or the sum of powers is built instead, and Circuit::notes says so.
#include "GEO_Q.h"
#include "Basics.h"
#include <functional>
#include <limits>

namespace cobble {

struct Gate {
    Eigen::MatrixXcd U;                        // 2^k x 2^k; bit j of its index is qubit targets[j]
    std::vector<std::size_t> targets;
    std::vector<std::size_t> on1, on0;         // controls: these qubits must be 1 / must be 0
    long param = -1;                           // >= 0: U = exp(-i s theta[param] G / 2), s = -1 if inverted
    Eigen::MatrixXcd G;                        // its generator (G^2 = I)
    bool inverted = false;
    std::string label;
};

struct Circuit {
    std::size_t data = 0, workspace = 0;
    double alpha = 1.0;
    std::vector<Gate> gates;
    std::uint64_t leaf_calls = 0;              // uses of leaves (Pauli strings, unitary leaves, rotations)
    double eps = 0.0;                          // bound on ||alpha * block - [[t]]|| from the fitted QSP phases
                                               // (0 when no Poly is compiled by QSVT); same propagation rules
                                               // as the cost model's eps column
    std::string notes;                         // what was built where it differs from the cost model's choice
    std::size_t qubits() const { return data + workspace; }
};

namespace codegen {

inline Eigen::MatrixXcd pauli2(char c) {
    Eigen::MatrixXcd M(2, 2);
    if (c == 'X') M << 0, 1, 1, 0;
    else if (c == 'Y') M << 0, Coeff(0, -1), Coeff(0, 1), 0;
    else if (c == 'Z') M << 1, 0, 0, -1;
    else M = Eigen::MatrixXcd::Identity(2, 2);
    return M;
}
inline Eigen::MatrixXcd diag2(Coeff a, Coeff b) { Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(2, 2); M(0, 0) = a; M(1, 1) = b; return M; }
inline Eigen::MatrixXcd ry(double t) {
    Eigen::MatrixXcd M(2, 2);
    M << std::cos(t / 2), -std::sin(t / 2), std::sin(t / 2), std::cos(t / 2);
    return M;
}
inline Eigen::MatrixXcd hadamard() { Eigen::MatrixXcd M(2, 2); M << 1, 1, 1, -1; return M / std::sqrt(2.0); }

inline Gate plain(const Eigen::MatrixXcd& U, std::vector<std::size_t> targets, std::string label,
                  std::vector<std::size_t> on1 = {}, std::vector<std::size_t> on0 = {}) {
    Gate g;
    g.U = U; g.targets = std::move(targets); g.on1 = std::move(on1); g.on0 = std::move(on0); g.label = std::move(label);
    return g;
}

// Append child's gates to out, child qubit q becoming map[q], with extra controls.
inline void embed(Circuit& out, const Circuit& child, const std::vector<std::size_t>& map,
                  const std::vector<std::size_t>& on1 = {}, const std::vector<std::size_t>& on0 = {}) {
    for (const Gate& g : child.gates) {
        Gate h = g;
        for (auto& q : h.targets) q = map[q];
        for (auto& q : h.on1) q = map[q];
        for (auto& q : h.on0) q = map[q];
        h.on1.insert(h.on1.end(), on1.begin(), on1.end());
        h.on0.insert(h.on0.end(), on0.begin(), on0.end());
        out.gates.push_back(std::move(h));
    }
    out.leaf_calls += child.leaf_calls;
}

inline Circuit inverse(const Circuit& c) {
    Circuit r = c;
    r.gates.assign(c.gates.rbegin(), c.gates.rend());
    for (Gate& g : r.gates) {
        if (g.param >= 0) g.inverted = !g.inverted;
        else g.U = g.U.adjoint().eval();
    }
    return r;
}

// Controls selecting the value v on the qubits qs (bit j of v on qs[j]).
inline void pattern(const std::vector<std::size_t>& qs, std::uint64_t v, std::vector<std::size_t>& on1, std::vector<std::size_t>& on0) {
    for (std::size_t j = 0; j < qs.size(); ++j) ((v >> j) & 1 ? on1 : on0).push_back(qs[j]);
}

// +1 (or -1) modulo 2^L on the qubits qs (qs[0] lowest): the bit qs[i] flips when every lower bit is 1
// (for +1) or 0 (for -1), highest bit first so that the lower bits are still the old ones.
inline void step(std::vector<Gate>& out, const std::vector<std::size_t>& qs, bool up,
                 const std::vector<std::size_t>& on1 = {}, const std::vector<std::size_t>& on0 = {}) {
    for (std::size_t i = qs.size(); i-- > 0;) {
        std::vector<std::size_t> c1 = on1, c0 = on0;
        for (std::size_t j = 0; j < i; ++j) (up ? c1 : c0).push_back(qs[j]);
        out.push_back(plain(pauli2('X'), {qs[i]}, up ? "inc" : "dec", c1, c0));
    }
}

// x -> x + a modulo 2^L on qs: one +-2^k step per binary digit of the representative of a in (-N/2, N/2].
inline void add_constant(std::vector<Gate>& out, const std::vector<std::size_t>& qs, long a,
                         const std::vector<std::size_t>& on1 = {}, const std::vector<std::size_t>& on0 = {}) {
    if (qs.empty()) return;
    const long N = long(1) << qs.size();
    long r = ((a % N) + N) % N;
    if (r > N / 2) r -= N;
    const bool up = r >= 0;
    const unsigned long u = (unsigned long)std::labs(r);
    for (std::size_t k = 0; k < qs.size(); ++k)
        if ((u >> k) & 1UL) step(out, std::vector<std::size_t>(qs.begin() + long(k), qs.end()), up, on1, on0);
}

// Gates for a unitary leaf on qubits 0..n-1. Recognises U = D P with P |y> = |p(y)> a shift on a
// two-axis grid (axis 1 on the low n1 bits) and D diagonal; otherwise one dense gate.
inline void unitary_gates(const Eigen::MatrixXcd& U, std::size_t n, const std::string& name, std::vector<Gate>& out) {
    const Eigen::Index N = U.rows();
    std::vector<Eigen::Index> p(static_cast<std::size_t>(N));
    std::vector<Coeff> d(static_cast<std::size_t>(N));
    bool monomial = true;
    for (Eigen::Index y = 0; y < N && monomial; ++y) {
        Eigen::Index row = -1;
        for (Eigen::Index x = 0; x < N; ++x)
            if (std::abs(U(x, y)) > 1e-12) { if (row >= 0) { monomial = false; break; } row = x; }
        if (row < 0) monomial = false;
        else { p[std::size_t(y)] = row; d[std::size_t(row)] = U(row, y); }
    }
    std::vector<std::size_t> all(n);
    for (std::size_t q = 0; q < n; ++q) all[q] = q;
    if (monomial) {
        for (std::size_t n1 = 0; n1 <= n; ++n1) {
            const Eigen::Index N1 = Eigen::Index(1) << n1, N2 = N / N1;
            const Eigen::Index a = p[0] % N1, b = p[0] / N1;
            bool ok = true;
            for (Eigen::Index y = 0; y < N && ok; ++y)
                ok = p[std::size_t(y)] == ((y % N1 + a) % N1) + N1 * ((y / N1 + b) % N2);
            if (!ok) continue;
            const std::vector<std::size_t> lo(all.begin(), all.begin() + long(n1)), hi(all.begin() + long(n1), all.end());
            add_constant(out, lo, long(a));
            add_constant(out, hi, long(b));
            bool signs = true, trivial = true;
            for (const Coeff& c : d) {
                signs = signs && (std::abs(c - 1.0) < 1e-12 || std::abs(c + 1.0) < 1e-12);
                trivial = trivial && std::abs(c - 1.0) < 1e-12;
            }
            if (trivial) return;
            if (signs) {   // one multi-controlled Z per subcube of the points with sign -1
                std::vector<bool> neg(static_cast<std::size_t>(N));
                for (Eigen::Index x = 0; x < N; ++x) neg[std::size_t(x)] = std::abs(d[std::size_t(x)] + 1.0) < 1e-12;
                std::vector<std::pair<std::uint64_t, std::uint64_t>> cubes;   // (mask, value)
                std::function<void(const std::vector<bool>&, int, std::uint64_t, std::uint64_t)> cover =
                    [&](const std::vector<bool>& f, int bits, std::uint64_t mask, std::uint64_t value) {
                        bool any = false, every = true;
                        for (bool b : f) { any = any || b; every = every && b; }
                        if (!any) return;
                        if (every) { cubes.emplace_back(mask, value); return; }
                        const std::size_t h = f.size() / 2;
                        const std::vector<bool> f0(f.begin(), f.begin() + long(h)), f1(f.begin() + long(h), f.end());
                        const std::uint64_t bit = std::uint64_t(1) << (bits - 1);
                        if (f0 == f1) { cover(f0, bits - 1, mask, value); return; }
                        cover(f0, bits - 1, mask | bit, value);
                        cover(f1, bits - 1, mask | bit, value | bit);
                    };
                cover(neg, int(n), 0, 0);
                for (auto [mask, value] : cubes) {
                    if (mask == 0) { out.push_back(plain(diag2(-1.0, -1.0), {0}, "-1")); continue; }
                    std::size_t t = 0;
                    while (!((mask >> t) & 1)) ++t;
                    std::vector<std::size_t> c1, c0;
                    for (std::size_t q = t + 1; q < n; ++q)
                        if ((mask >> q) & 1) (((value >> q) & 1) ? c1 : c0).push_back(q);
                    out.push_back(plain(((value >> t) & 1) ? diag2(1.0, -1.0) : diag2(-1.0, 1.0), {t}, "mcZ", c1, c0));
                }
                return;
            }
            Eigen::MatrixXcd D = Eigen::MatrixXcd::Zero(N, N);
            for (Eigen::Index x = 0; x < N; ++x) D(x, x) = d[std::size_t(x)];
            out.push_back(plain(D, all, name + " (diagonal)"));
            return;
        }
    }
    out.push_back(plain(U, all, name));
}

// ---- QSP phases (reflection convention) -------------------------------------------------------
// g(phi, x) = <0| prod_{j=1..d} (e^{i phi_j Z} R(x)) |0>,  R(x) = [[x, s], [s, -x]], s = sqrt(1 - x^2)
// (Gilyen et al. 2019, Cor. 8 and Thm. 17). Wanted: Re g(phi, x) = f(x) on [-1, 1], f of parity d.
using M2 = Eigen::Matrix2cd;
inline M2 R(double x) { const double s = std::sqrt(std::max(0.0, 1 - x * x)); M2 m; m << x, s, s, -x; return m; }
inline M2 ph(double phi) { M2 m = M2::Zero(); m(0, 0) = std::polar(1.0, phi); m(1, 1) = std::polar(1.0, -phi); return m; }

inline Coeff g(const std::vector<double>& phi, double x) {
    M2 m = M2::Identity();
    for (double p : phi) m = m * ph(p) * R(x);
    return m(0, 0);
}

// Re g(phi, x) and its gradient in phi.
inline double re_g_grad(const std::vector<double>& phi, double x, std::vector<double>& grad) {
    const std::size_t d = phi.size();
    std::vector<M2> pre(d + 1), suf(d + 1);
    pre[0] = M2::Identity();
    for (std::size_t j = 0; j < d; ++j) pre[j + 1] = pre[j] * ph(phi[j]) * R(x);
    suf[d] = M2::Identity();
    for (std::size_t j = d; j-- > 0;) suf[j] = ph(phi[j]) * R(x) * suf[j + 1];
    M2 Z = M2::Zero(); Z(0, 0) = Coeff(0, 1); Z(1, 1) = Coeff(0, -1);   // i Z
    grad.assign(d, 0.0);
    for (std::size_t j = 0; j < d; ++j) grad[j] = (pre[j] * Z * ph(phi[j]) * R(x) * suf[j + 1])(0, 0).real();
    return pre[d](0, 0).real();
}

// Dong, Meng, Whaley & Lin (PRA 103, 042419, 2021): W convention, U(x) = e^{i p_0 Z} prod_{j=1..d} W(x) e^{i p_j Z},
// W(x) = [[x, i s], [i s, x]], target Re <0|U(x)|0> = f(x) (at the starting phases <0|U|0> = i T_d(x), whose real
// part is 0); symmetric phases p_j = p_{d-j}, started at (pi/4, 0, ..., 0, pi/4), fitted at the d~ = ceil((d+1)/2) nodes cos((2k-1) pi / (4 d~)) by Newton's method
// (square system), damped when a step does not decrease the residual.
inline M2 W(double x) { const double s = std::sqrt(std::max(0.0, 1 - x * x)); M2 m; m << x, Coeff(0, s), Coeff(0, s), x; return m; }

inline std::vector<double> dong_phases(const std::function<double(double)>& f0, std::size_t d, bool& ok) {
    const std::size_t dt = (d + 2) / 2;                       // ceil((d + 1) / 2) reduced phases
    std::vector<double> xs(dt);
    for (std::size_t k = 1; k <= dt; ++k) xs[k - 1] = std::cos((2.0 * double(k) - 1) * std::numbers::pi / (4.0 * double(dt)));
    double scale = 0.5;
    std::function<double(double)> f = [&](double x) { return scale * f0(x); };
    auto full = [&](const Eigen::VectorXd& r) {
        std::vector<double> p(d + 1);
        for (std::size_t j = 0; j <= d; ++j) p[j] = r[Eigen::Index(std::min(j, d - j))];
        return p;
    };
    M2 iZ = M2::Zero(); iZ(0, 0) = Coeff(0, 1); iZ(1, 1) = Coeff(0, -1);
    auto residual = [&](const Eigen::VectorXd& r, Eigen::VectorXd& res, Eigen::MatrixXd* J) {
        const std::vector<double> p = full(r);
        res.resize(Eigen::Index(dt));
        if (J) J->setZero(Eigen::Index(dt), Eigen::Index(dt));
        for (std::size_t k = 0; k < dt; ++k) {
            const double x = xs[k];
            std::vector<M2> pre(d + 2), suf(d + 2);         // pre[j] = everything left of e^{i p_j Z}
            pre[0] = M2::Identity();
            for (std::size_t j = 0; j <= d; ++j) pre[j + 1] = pre[j] * ph(p[j]) * (j < d ? W(x) : M2::Identity());
            suf[d + 1] = M2::Identity();
            for (std::size_t j = d + 1; j-- > 0;) suf[j] = ph(p[j]) * (j < d ? W(x) : M2::Identity()) * suf[j + 1];
            res[Eigen::Index(k)] = pre[d + 1](0, 0).real() - f(x);
            if (J)
                for (std::size_t j = 0; j <= d; ++j) {
                    const M2 Wj = j < d ? W(x) : M2::Identity();
                    const double dj = (pre[j] * iZ * ph(p[j]) * Wj * suf[j + 1])(0, 0).real();
                    (*J)(Eigen::Index(k), Eigen::Index(std::min(j, d - j))) += dj;
                }
        }
    };
    // Continuation in the target's size: s f for s = 1/2 first (inside the region where the iteration provably
    // converges, ||f|| <= 0.861: Dong, Lin, Ni & Wang 2022), then s raised step by step to 1, each Newton solve
    // started from the previous solution (standard numerical continuation; Allgower & Georg 1990).
    Eigen::VectorXd r = Eigen::VectorXd::Zero(Eigen::Index(dt));
    r[0] = std::numbers::pi / 4;
    Eigen::VectorXd res;
    Eigen::MatrixXd J;
    auto newton = [&]() {
        residual(r, res, &J);
        double lam = 0.0;
        for (int it = 0; it < 100 && res.cwiseAbs().maxCoeff() > 1e-14; ++it) {
            Eigen::MatrixXd A = J.transpose() * J;
            A.diagonal().array() += lam;
            const Eigen::VectorXd step = A.ldlt().solve(-J.transpose() * res);
            const Eigen::VectorXd trial = r + step;
            Eigen::VectorXd rt;
            residual(trial, rt, nullptr);
            if (rt.norm() < res.norm()) { r = trial; residual(r, res, &J); lam = lam / 10; }
            else { lam = std::max(1e-10, lam * 10); if (lam > 1e8) break; }
        }
        return res.cwiseAbs().maxCoeff() <= 1e-12;
    };
    ok = newton();
    double step_s = 0.1;
    while (ok && scale < 1.0) {
        const Eigen::VectorXd saved = r;
        const double prev = scale;
        scale = std::min(1.0, scale + step_s);
        if (newton()) { step_s = std::min(0.2, step_s * 1.5); continue; }
        r = saved;                                   // too far: go back, smaller step
        scale = prev;
        step_s /= 4;
        if (step_s < 1e-7) { ok = false; break; }
        ok = true;
    }
    ok = ok && scale == 1.0;
    return full(r);
}

// Reflection-convention phases phi_1..phi_d and a phase kappa with Re(kappa g(phi, x)) = f(x) on [-1, 1].
// From Dong et al.'s phases p (Re g_W = f), using R(x) = -i e^{i pi Z / 4} W(x) e^{i pi Z / 4} (checked by
// multiplying out): phi_1 = p_0 - pi/4, phi_{j+1} = p_j - pi/2, and g_R = omega g_W(p) with
// omega = (-i)^d e^{-i (p_d - pi/4)}, so kappa = 1 / omega. The fallback, if that fails, is a direct least-squares fit of
// Re g_R = f from many random starts (kappa = 1). err_out: largest error on a fine grid of [-1, 1].
inline std::vector<double> phases(const std::function<double(double)>& f, std::size_t d, Coeff& kappa, double& err_out) {
    kappa = 1.0;
    err_out = 0.0;
    if (d == 0) return {};
    auto grid_error = [&](const std::vector<double>& phi, Coeff k) {
        double e = 0.0;
        for (int i = 0; i <= 400; ++i) {
            const double x = -1.0 + 2.0 * i / 400.0;
            e = std::max(e, std::abs((k * g(phi, x)).real() - f(x)));
        }
        return e;
    };
    bool ok = false;
    const std::vector<double> p = dong_phases(f, d, ok);
    std::vector<double> phi(d);
    phi[0] = p[0] - std::numbers::pi / 4;
    for (std::size_t j = 1; j < d; ++j) phi[j] = p[j] - std::numbers::pi / 2;
    const Coeff omega = std::pow(Coeff(0, -1), double(d)) * std::polar(1.0, -(p[d] - std::numbers::pi / 4));
    kappa = 1.0 / omega;
    err_out = grid_error(phi, kappa);
    if (err_out <= 1e-7) return phi;   // the circuit's eps records this error

    // Fallback: Levenberg-Marquardt on Re g_R = f from random starts.
    std::vector<double> xs;
    for (std::size_t k = 0; k < d + 2; ++k) xs.push_back(std::cos(std::numbers::pi * (2.0 * double(k) + 1) / (4.0 * double(d + 2))));
    xs.push_back(1.0);
    std::mt19937_64 rng(1105 + d);
    std::uniform_real_distribution<double> u(-std::numbers::pi, std::numbers::pi);
    std::vector<double> best = phi;
    Coeff best_k = kappa;
    double best_err = err_out;
    for (int start = 0; start < 40 && best_err > 1e-11; ++start) {
        std::vector<double> q(d);
        for (double& v : q) v = u(rng) * (start < 20 ? 0.25 : 1.0);
        auto residuals = [&](const std::vector<double>& pp, Eigen::VectorXd& rr, Eigen::MatrixXd* JJ) {
            rr.resize(Eigen::Index(xs.size()));
            if (JJ) JJ->resize(Eigen::Index(xs.size()), Eigen::Index(d));
            std::vector<double> gr;
            for (std::size_t k = 0; k < xs.size(); ++k) {
                rr[Eigen::Index(k)] = re_g_grad(pp, xs[k], gr) - f(xs[k]);
                if (JJ) for (std::size_t j = 0; j < d; ++j) (*JJ)(Eigen::Index(k), Eigen::Index(j)) = gr[j];
            }
        };
        Eigen::VectorXd rr;
        Eigen::MatrixXd JJ;
        residuals(q, rr, &JJ);
        double mu = 1e-3, c2 = rr.squaredNorm();
        for (int it = 0; it < 500 && rr.cwiseAbs().maxCoeff() > 1e-13; ++it) {
            Eigen::MatrixXd A = JJ.transpose() * JJ;
            A.diagonal().array() += mu * (A.diagonal().array() + 1e-12);
            const Eigen::VectorXd st = A.ldlt().solve(-JJ.transpose() * rr);
            std::vector<double> trial(d);
            for (std::size_t j = 0; j < d; ++j) trial[j] = q[j] + st[Eigen::Index(j)];
            Eigen::VectorXd rt;
            residuals(trial, rt, nullptr);
            if (rt.squaredNorm() < c2) { q = trial; residuals(q, rr, &JJ); c2 = rr.squaredNorm(); mu = std::max(mu / 3, 1e-15); }
            else { mu *= 4; if (mu > 1e12) break; }
        }
        const double e = grid_error(q, 1.0);
        if (e < best_err) { best_err = e; best = q; best_k = 1.0; }
    }
    err_out = best_err;
    kappa = best_k;
    if (best_err <= 1e-6) return best;
    throw std::runtime_error("codegen: QSP phase finding did not converge (degree " + std::to_string(d) +
                             ", best error " + std::to_string(best_err) + ").");
}

// ---- Compilation ------------------------------------------------------------------------------

inline Circuit build(const Term& t);

inline Circuit build_sum(const std::vector<Coeff>& cs, const std::vector<Term>& ts, std::size_t n) {
    std::vector<Circuit> kids;
    std::size_t mk = 0;
    double lambda = 0.0;
    for (std::size_t j = 0; j < ts.size(); ++j) {
        kids.push_back(build(ts[j]));
        mk = std::max(mk, kids.back().workspace);
        lambda += std::abs(cs[j]) * kids.back().alpha;
    }
    const std::size_t r = detail::ceil_log2(ts.size());
    Circuit c;
    c.data = n;
    c.workspace = r + mk;
    c.alpha = lambda;
    for (std::size_t j = 0; j < ts.size(); ++j) c.eps += std::abs(cs[j]) * kids[j].eps;
    std::vector<std::size_t> idx(r);
    for (std::size_t j = 0; j < r; ++j) idx[j] = n + j;
    // PREPARE: amplitudes sqrt(|c_j| alpha_j / lambda) by a tree of controlled R_y (top bit first).
    std::vector<double> amp(std::size_t(1) << r, 0.0);
    for (std::size_t j = 0; j < ts.size(); ++j) amp[j] = std::sqrt(std::abs(cs[j]) * kids[j].alpha / lambda);
    std::vector<Gate> prep;
    for (std::size_t t = r; t-- > 0;) {
        for (std::uint64_t h = 0; h < (std::uint64_t(1) << (r - 1 - t)); ++h) {
            double n0 = 0, n1 = 0;
            for (std::uint64_t i = 0; i < amp.size(); ++i)
                if ((i >> (t + 1)) == h) ((i >> t) & 1 ? n1 : n0) += amp[i] * amp[i];
            if (n0 + n1 <= 0) continue;
            std::vector<std::size_t> on1, on0;
            for (std::size_t b = t + 1; b < r; ++b) ((h >> (b - t - 1)) & 1 ? on1 : on0).push_back(idx[b]);
            prep.push_back(plain(ry(2 * std::atan2(std::sqrt(n1), std::sqrt(n0))), {idx[t]}, "prep", on1, on0));
        }
    }
    c.gates = prep;
    // SELECT: term j (with the phase of c_j) when the index register holds j.
    for (std::size_t j = 0; j < ts.size(); ++j) {
        if (cs[j] == Coeff(0.0)) continue;
        std::vector<std::size_t> on1, on0;
        pattern(idx, j, on1, on0);
        std::vector<std::size_t> map(kids[j].qubits());
        for (std::size_t q = 0; q < n; ++q) map[q] = q;
        for (std::size_t q = 0; q < kids[j].workspace; ++q) map[n + q] = n + r + q;
        embed(c, kids[j], map, on1, on0);
        const Coeff phase = cs[j] / std::abs(cs[j]);
        if (std::abs(phase - 1.0) > 1e-15) c.gates.push_back(plain(diag2(phase, phase), {0}, "phase", on1, on0));
    }
    for (auto it = prep.rbegin(); it != prep.rend(); ++it) { Gate g2 = *it; g2.U = g2.U.adjoint().eval(); c.gates.push_back(g2); }
    return c;
}

inline Circuit build_product(const std::vector<Term>& ts, std::size_t n) {
    std::vector<Circuit> kids;
    std::size_t mk = 0;
    for (const Term& f : ts) { kids.push_back(build(f)); mk = std::max(mk, kids.back().workspace); }
    const std::size_t cnt = detail::ceil_log2(ts.size());
    Circuit c;
    c.data = n;
    c.workspace = cnt + mk;
    c.alpha = 1.0;
    std::vector<std::size_t> counter(cnt), shared(mk);
    for (std::size_t j = 0; j < cnt; ++j) counter[j] = n + j;
    for (std::size_t j = 0; j < mk; ++j) shared[j] = n + cnt + j;
    double with_eps = 1.0;
    for (const Circuit& k : kids) with_eps *= k.alpha + k.eps;
    std::size_t checks = 0;
    for (std::size_t i = ts.size(); i-- > 0;) {      // the last factor acts first
        std::vector<std::size_t> map(kids[i].qubits());
        for (std::size_t q = 0; q < n; ++q) map[q] = q;
        for (std::size_t q = 0; q < kids[i].workspace; ++q) map[n + q] = shared[q];
        embed(c, kids[i], map);
        c.alpha *= kids[i].alpha;
        if (i > 0 && kids[i].workspace > 0) {        // count this factor's success: workspace all 0
            step(c.gates, counter, true, {}, shared);
            ++checks;
        }
    }
    if (checks > 0) add_constant(c.gates, counter, -long(checks));
    c.eps = with_eps - c.alpha;
    return c;
}

inline Circuit build_poly(const Poly& P, const Info& I, std::size_t n) {
    const Circuit base = build(P.a);
    const long d = detail::true_degree(P.coeff);
    const bool hermitian = info(P.a).hermitian;
    Circuit c;
    c.data = n;
    std::string chosen;
    {
        static const char* names[6] = {"LCU", "Horner", "QSVT", "GQET", "Ko", "Truncated"};
        chosen = names[int(I.method)];
    }
    // Chebyshev coefficients of the polynomial actually built, in x = M / alpha_M.
    std::vector<double> cheb;
    if (I.method == PolyMethod::Truncated) {
        cheb = I.realised_cheb;
    } else if (hermitian && I.method != PolyMethod::LCU && I.method != PolyMethod::Horner) {
        std::vector<double> q(std::size_t(d) + 1);
        for (long j = 0; j <= d; ++j) q[std::size_t(j)] = P.coeff[std::size_t(j)] * std::pow(base.alpha, double(j));
        cheb = d <= detail::exact_degree_limit ? polymax::to_chebyshev(q) : polymax::chebyshev_by_sampling(q);
    }
    if (cheb.empty()) {
        // Sum of powers: a_0 I + a_1 M + a_2 M M + ...
        std::vector<Coeff> cs;
        std::vector<Term> ts;
        for (long j = 0; j <= d; ++j) {
            if (P.coeff[std::size_t(j)] == 0.0) continue;
            cs.push_back(P.coeff[std::size_t(j)]);
            if (j == 0) ts.push_back(leaf(PauliString::from_string(std::string(n, 'I'))));
            else if (j == 1) ts.push_back(P.a);
            else ts.push_back(product(std::vector<Term>(std::size_t(j), P.a)));
        }
        c = build_sum(cs, ts, n);
        c.notes = "sum of powers (Cobble's LCU method)" + std::string(chosen == "LCU" ? "" : "; the cost model chose " + chosen);
        return c;
    }
    // QSVT on the even and odd parts.
    std::vector<double> ce(cheb.size(), 0.0), co(cheb.size(), 0.0);
    for (std::size_t k = 0; k < cheb.size(); ++k) (k % 2 ? co : ce)[k] = cheb[k];
    ce = detail::clean(ce);
    co = detail::clean(co);
    const long de = detail::true_degree(ce), dq = detail::true_degree(co);
    const double ae = de >= 0 ? detail::cheb_max(ce) : 0.0, ao = dq >= 0 ? detail::cheb_max(co) : 0.0;
    const std::size_t mM = base.workspace;
    const std::size_t rq = n + mM;                    // the qubit that takes the real part
    const bool both = de >= 0 && dq >= 0;
    const std::size_t sq = rq + 1;                    // the qubit that joins the two parts
    c.workspace = mM + 1 + (both ? 1 : 0);
    c.alpha = ae + ao;
    std::vector<std::size_t> anc(mM);
    for (std::size_t j = 0; j < mM; ++j) anc[j] = n + j;
    std::vector<std::size_t> id(base.qubits());
    for (std::size_t q = 0; q < base.qubits(); ++q) id[q] = q;
    const Circuit baseinv = inverse(base);

    // Error carried in from the base (as in the cost model): sum over monomials |a_j| ((alpha + eps)^j - alpha^j),
    // plus each part's fitting error times its alpha.
    if (base.eps > 0.0)
        for (long j = 1; j <= d; ++j)
            c.eps += std::abs(P.coeff[std::size_t(j)]) * (std::pow(base.alpha + base.eps, double(j)) - std::pow(base.alpha, double(j)));
    if (I.method == PolyMethod::Truncated) c.eps += I.cost.eps;   // the truncation itself
    auto part = [&](const std::vector<double>& cp, long dp, double a, const std::vector<std::size_t>& on1, const std::vector<std::size_t>& on0) {
        auto f = [&](double x) { return polymax::clenshaw(cp, x) / a; };
        double fit = 0.0;
        Coeff kappa = 1.0;
        const std::vector<double> phi = phases(f, std::size_t(dp), kappa, fit);
        c.eps += a * fit;
        c.gates.push_back(plain(hadamard(), {rq}, "H", on1, on0));
        if (std::abs(kappa - 1.0) > 1e-15)              // Re(kappa g): kappa on the r = 0 branch, conj(kappa) on r = 1
            c.gates.push_back(plain(diag2(kappa, std::conj(kappa)), {rq}, "kappa", on1, on0));
        if (dp == 0 && cp[0] < 0) c.gates.push_back(plain(diag2(-1.0, -1.0), {rq}, "-1", on1, on0));
        for (std::size_t j = std::size_t(dp); j-- > 0;) {   // time order: W_d, phase_d, W_{d-1}, phase_{d-1}, ...
            const bool forward = ((std::size_t(dp) - 1 - j) % 2) == 0;
            embed(c, forward ? base : baseinv, id, on1, on0);
            const double p = phi[j];
            c.gates.push_back(plain(diag2(std::polar(1.0, -p), std::polar(1.0, p)), {rq}, "qsvt phase", on1, on0));
            std::vector<std::size_t> z0 = on0;
            z0.insert(z0.end(), anc.begin(), anc.end());
            c.gates.push_back(plain(diag2(std::polar(1.0, 2 * p), std::polar(1.0, -2 * p)), {rq}, "qsvt phase", on1, z0));
        }
        c.gates.push_back(plain(hadamard(), {rq}, "H", on1, on0));
    };
    if (both) {
        const double t = 2 * std::atan2(std::sqrt(ao), std::sqrt(ae));
        c.gates.push_back(plain(ry(t), {sq}, "prep"));
        part(ce, de, ae, {}, {sq});
        part(co, dq, ao, {sq}, {});
        c.gates.push_back(plain(ry(-t), {sq}, "prep"));
    } else if (de >= 0) {
        part(ce, de, ae, {}, {});
    } else {
        part(co, dq, ao, {}, {});
    }
    c.notes = "QSVT" + std::string(chosen == "QSVT" || chosen == "Truncated" ? "" : "; the cost model chose " + chosen);
    if (chosen == "Truncated") c.notes = "QSVT on the truncated Chebyshev series";
    return c;
}

inline Circuit build(const Term& t) {
    const std::size_t n = t->qubits;
    if (const Leaf* l = std::get_if<Leaf>(&t->kind)) {
        Circuit c;
        c.data = n;
        const PauliString& P = l->p;
        c.leaf_calls = (P.x | P.z) ? 1 : 0;                   // the identity emits no gate and costs no query
        unsigned ys = 0;
        for (std::size_t q = 0; q < n; ++q) {
            const bool x = (P.x >> q) & 1, z = (P.z >> q) & 1;
            if (x && z) ++ys;
            if (x || z) c.gates.push_back(plain(pauli2(x && z ? 'Y' : (x ? 'X' : 'Z')), {q}, "pauli"));
        }
        static const Coeff ik[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        const Coeff phase = ik[(P.k + 4 - ys % 4) % 4];   // i^k X^x Z^z = i^(k - #Y) (product of letters)
        if (phase != Coeff(1.0)) c.gates.push_back(plain(diag2(phase, phase), {0}, "phase"));
        return c;
    }
    if (const Unitary* u = std::get_if<Unitary>(&t->kind)) {
        Circuit c;
        c.data = n;
        c.leaf_calls = 1;
        unitary_gates(u->U, n, u->name, c.gates);
        return c;
    }
    if (const Rotation* r = std::get_if<Rotation>(&t->kind)) {
        Circuit c;
        c.data = n;
        c.leaf_calls = 1;
        std::vector<std::size_t> support;
        Eigen::MatrixXcd G = Eigen::MatrixXcd::Identity(1, 1);
        unsigned ys = 0;
        for (std::size_t q = 0; q < n; ++q) {
            const bool x = (r->p.x >> q) & 1, z = (r->p.z >> q) & 1;
            if (!(x || z)) continue;
            if (x && z) ++ys;
            support.push_back(q);
            const Eigen::MatrixXcd L = pauli2(x && z ? 'Y' : (x ? 'X' : 'Z'));
            Eigen::MatrixXcd K(G.rows() * 2, G.cols() * 2);   // new qubit = new highest bit of G's index
            for (Eigen::Index i = 0; i < 2; ++i)
                for (Eigen::Index j = 0; j < 2; ++j) K.block(i * G.rows(), j * G.cols(), G.rows(), G.cols()) = L(i, j) * G;
            G = K;
        }
        const double sign = ((r->p.k + 4 - ys % 4) % 4) == 0 ? 1.0 : -1.0;   // Hermitian: the phase is +-1
        if (support.empty()) { support.push_back(0); G = Eigen::MatrixXcd::Identity(2, 2); }
        Gate g;
        g.targets = support;
        g.param = long(r->param);
        g.G = sign * G;
        g.label = "rot";
        c.gates.push_back(g);
        return c;
    }
    if (const Adjoint* a = std::get_if<Adjoint>(&t->kind)) return inverse(build(a->a));
    if (const Sum* s = std::get_if<Sum>(&t->kind)) return build_sum(s->c, s->t, n);
    if (const Product* p = std::get_if<Product>(&t->kind)) return build_product(p->t, n);
    if (const Tensor* x = std::get_if<Tensor>(&t->kind)) {
        std::vector<Circuit> kids;
        std::size_t ws = 0;
        for (const Term& f : x->t) { kids.push_back(build(f)); ws += kids.back().workspace; }
        Circuit c;
        c.data = n;
        c.workspace = ws;
        double with_eps = 1.0;
        for (const Circuit& k : kids) with_eps *= k.alpha + k.eps;
        std::size_t lo = 0, wlo = 0;
        for (const Circuit& k : kids) {
            std::vector<std::size_t> map(k.qubits());
            for (std::size_t q = 0; q < k.data; ++q) map[q] = lo + q;
            for (std::size_t q = 0; q < k.workspace; ++q) map[k.data + q] = n + wlo + q;
            embed(c, k, map);
            c.alpha *= k.alpha;
            lo += k.data;
            wlo += k.workspace;
        }
        c.eps = with_eps - c.alpha;
        return c;
    }
    if (const DirectSum* d = std::get_if<DirectSum>(&t->kind)) {
        std::vector<Circuit> kids;
        double top = 0.0;
        for (const Term& b : d->t) { kids.push_back(build(b)); top = std::max(top, kids.back().alpha); }
        const std::size_t nb = d->t[0]->qubits, sel = n - nb;
        std::size_t ws = 0;
        std::vector<bool> resc;
        for (const Circuit& k : kids) {
            resc.push_back(top - k.alpha > detail::alpha_tolerance * top);
            ws = std::max(ws, k.workspace + (resc.back() ? 1 : 0));
        }
        Circuit c;
        c.data = n;
        c.workspace = ws;
        c.alpha = top;
        std::vector<std::size_t> selq(sel);
        for (std::size_t j = 0; j < sel; ++j) selq[j] = nb + j;
        for (std::size_t j = 0; j < kids.size(); ++j) {
            std::vector<std::size_t> on1, on0;
            pattern(selq, j, on1, on0);
            std::vector<std::size_t> map(kids[j].qubits());
            for (std::size_t q = 0; q < nb; ++q) map[q] = q;
            for (std::size_t q = 0; q < kids[j].workspace; ++q) map[nb + q] = n + q;
            embed(c, kids[j], map, on1, on0);
            if (resc[j]) c.gates.push_back(plain(ry(2 * std::acos(kids[j].alpha / top)), {n + kids[j].workspace}, "rescale", on1, on0));
            c.eps = std::max(c.eps, kids[j].eps + (resc[j] ? 0.0 : top - kids[j].alpha));
        }
        return c;
    }
    const Poly& P = std::get<Poly>(t->kind);
    return build_poly(P, info(t), n);
}

}  // namespace codegen

// Compile a term to a circuit (see the top of this file for what it means).
inline Circuit compile(const Term& t) { return codegen::build(t); }

// ---- Running a circuit on the simulator (Basics.h) --------------------------------------------

// Run c on the register q (c.qubits() qubits) at angles theta. shift_gate / shift move one gate's angle
// (used by the parameter-shift rule).
inline void run(const Circuit& c, Qubits& q, const std::vector<double>& theta = {}, long shift_gate = -1, double shift = 0.0) {
    detail::require(q.num_qubits() == c.qubits(), "run: register of the wrong size.");
    const Eigen::MatrixXcd X = codegen::pauli2('X');
    for (std::size_t i = 0; i < c.gates.size(); ++i) {
        const Gate& g = c.gates[i];
        Eigen::MatrixXcd U = g.U;
        if (g.param >= 0) {
            detail::require(std::size_t(g.param) < theta.size(), "run: not enough parameter values.");
            double a = theta[std::size_t(g.param)] + (long(i) == shift_gate ? shift : 0.0);
            if (g.inverted) a = -a;
            U = std::cos(a / 2) * Eigen::MatrixXcd::Identity(g.G.rows(), g.G.cols()) - Coeff(0, std::sin(a / 2)) * g.G;
        }
        for (std::size_t z : g.on0) q.applyOn(X, {z});
        if (g.on1.empty() && g.on0.empty()) q.applyOn(U, g.targets);
        else {
            std::vector<std::size_t> ctrl = g.on1;
            ctrl.insert(ctrl.end(), g.on0.begin(), g.on0.end());
            q.applyControlledOn(U, ctrl, g.targets);
        }
        for (std::size_t z : g.on0) q.applyOn(X, {z});
    }
}

// alpha times the top-left block of the circuit, read off the simulator column by column:
// it must equal the term's matrix.
inline Eigen::MatrixXcd block(const Circuit& c, const std::vector<double>& theta = {}) {
    const Eigen::Index N = Eigen::Index(1) << c.data, F = Eigen::Index(1) << c.qubits();
    Eigen::MatrixXcd B(N, N);
    for (Eigen::Index j = 0; j < N; ++j) {
        Qubits q(Eigen::VectorXcd::Unit(F, j));
        run(c, q, theta);
        B.col(j) = c.alpha * q.state_vector().head(N);
    }
    return B;
}

// <psi| U(theta)^dagger O U(theta) |psi> for a circuit that is unitary on its data qubits (its workspace
// always comes back to 0, e.g. products and tensors of rotations, Pauli strings and unitary leaves).
inline double expectation(const Circuit& c, const Term& O, const Eigen::VectorXcd& psi, const std::vector<double>& theta,
                          long shift_gate = -1, double shift = 0.0) {
    detail::require(O->qubits == c.data && psi.size() == Eigen::Index(1) << c.data, "expectation: sizes do not match.");
    Eigen::VectorXcd full = Eigen::VectorXcd::Zero(Eigen::Index(1) << c.qubits());
    full.head(psi.size()) = psi / psi.norm();
    Qubits q(full);
    run(c, q, theta, shift_gate, shift);
    const Eigen::VectorXcd out = q.state_vector();
    const Eigen::VectorXcd phi = out.head(psi.size());
    detail::require(std::abs(phi.squaredNorm() - 1.0) < 1e-10,
                    "expectation: the circuit is a block encoding, not a unitary on its data (workspace not returned to 0).");
    return phi.dot(act(O, phi)).real();
}

// Gradient of the expectation by the parameter-shift rule (Mitarai et al. 2018; Schuld et al. 2019): for a
// gate exp(-i a G / 2) with G^2 = I, dE/da = (E(a + pi/2) - E(a - pi/2)) / 2; a parameter used by several
// gates collects one such term per gate (chain rule). Needs uncontrolled rotations.
inline std::vector<double> gradient(const Circuit& c, const Term& O, const Eigen::VectorXcd& psi, const std::vector<double>& theta) {
    std::vector<double> grad(theta.size(), 0.0);
    for (std::size_t i = 0; i < c.gates.size(); ++i) {
        const Gate& g = c.gates[i];
        if (g.param < 0) continue;
        detail::require(g.on1.empty() && g.on0.empty(), "gradient: the parameter-shift rule needs uncontrolled rotations.");
        const double hp = expectation(c, O, psi, theta, long(i), std::numbers::pi / 2);
        const double hm = expectation(c, O, psi, theta, long(i), -std::numbers::pi / 2);
        grad[std::size_t(g.param)] += (hp - hm) / 2;   // the shift is applied before the sign flip, so this holds for inverted gates too
    }
    return grad;
}

// exp(-i theta_param (P_1 + ... + P_k) / 2) for commuting Pauli strings: exactly the product of the
// rotations, all with the same angle (e^{A+B} = e^A e^B when AB = BA). One trainable generator, e.g. a
// symmetry-averaged one.
inline Term pauli_evolution(const std::vector<PauliString>& ps, std::size_t param) {
    detail::require(!ps.empty(), "pauli_evolution: no Pauli strings.");
    for (std::size_t i = 0; i < ps.size(); ++i)
        for (std::size_t j = i + 1; j < ps.size(); ++j)
            detail::require(commute(ps[i], ps[j]), "pauli_evolution: the Pauli strings must commute.");
    if (ps.size() == 1) return rotation(ps[0], param);
    std::vector<Term> rs;
    for (const PauliString& p : ps) rs.push_back(rotation(p, param));
    return product(rs);
}

}  // namespace cobble
