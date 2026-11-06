#pragma once
// Maxima of real polynomials, for the subnormalisation of two Poly methods:
//   QSVT: max over [-1, 1] of |q(x)|                         (Cobble, Sec. 5.2)
//   GQET: max over |z| = 1 of |sum_n c_n z^n|, c = Chebyshev coefficients of q
//                                                            (Sunderhauf 2023, Eqs. 7-8)
// A polynomial is a coefficient vector in the monomial basis: q[k] is the coefficient of x^k.
#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

namespace polymax {

using Poly = std::vector<double>;

// Drop trailing zero coefficients, so that the last entry is the leading coefficient.
inline Poly trim(Poly q) {
    while (!q.empty() && q.back() == 0.0) q.pop_back();
    return q;
}

// q(x), by Horner's rule.
inline double eval(const Poly& q, double x) {
    double y = 0.0;
    for (auto it = q.rbegin(); it != q.rend(); ++it) y = y * x + *it;
    return y;
}

inline Poly derivative(const Poly& q) {
    Poly d;
    for (std::size_t k = 1; k < q.size(); ++k) d.push_back(double(k) * q[k]);
    return d;
}

// All complex roots of q: the eigenvalues of its companion matrix (Edelman & Murakami 1995).
inline Eigen::VectorXcd roots(const Poly& q_in) {
    const Poly q = trim(q_in);
    if (q.size() < 2) return Eigen::VectorXcd();
    const int n = int(q.size()) - 1;
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n, n);
    for (int i = 1; i < n; ++i) C(i, i - 1) = 1.0;
    for (int i = 0; i < n; ++i) C(i, n - 1) = -q[i] / q[n];
    return Eigen::EigenSolver<Eigen::MatrixXd>(C, false).eigenvalues();
}

// Problem A: max over [-1, 1] of |q(x)|.
// Candidates: the two endpoints and the critical points (roots of q') inside (-1, 1).
// The real part of every root is tried, so a real critical point that rounding pushed
// slightly off the real axis is still caught; extra candidates inside [-1, 1] can never
// raise the result above the true maximum.
inline double max_abs_interval(const Poly& q) {
    double best = std::max(std::abs(eval(q, -1.0)), std::abs(eval(q, 1.0)));
    const Eigen::VectorXcd r = roots(derivative(q));
    for (Eigen::Index i = 0; i < r.size(); ++i) {
        const double x = r[i].real();
        if (x > -1.0 && x < 1.0) best = std::max(best, std::abs(eval(q, x)));
    }
    return best;
}

// Monomial coefficients of T_0, ..., T_d, from T_0 = 1, T_1 = x, T_{n+1} = 2x T_n - T_{n-1}.
inline std::vector<Poly> chebyshev_table(std::size_t d) {
    std::vector<Poly> T(d + 1, Poly(d + 1, 0.0));
    T[0][0] = 1.0;
    if (d >= 1) T[1][1] = 1.0;
    for (std::size_t n = 1; n < d; ++n)
        for (std::size_t k = 0; k <= d; ++k)
            T[n + 1][k] = (k >= 1 ? 2.0 * T[n][k - 1] : 0.0) - T[n - 1][k];
    return T;
}

// Chebyshev coefficients c of q, so that q = sum_n c[n] T_n.
// T_n has degree n, so the top coefficient is fixed first, then removed, and so on downwards.
inline Poly to_chebyshev(const Poly& q_in) {
    Poly q = trim(q_in);
    if (q.empty()) return {};
    const std::size_t d = q.size() - 1;
    const auto T = chebyshev_table(d);
    Poly c(d + 1, 0.0);
    for (std::size_t n = d + 1; n-- > 0;) {
        c[n] = q[n] / T[n][n];
        for (std::size_t k = 0; k <= n; ++k) q[k] -= c[n] * T[n][k];
    }
    return c;
}

// Monomial coefficients of sum_n c[n] T_n.
inline Poly from_chebyshev(const Poly& c) {
    if (c.empty()) return {};
    const std::size_t d = c.size() - 1;
    const auto T = chebyshev_table(d);
    Poly q(d + 1, 0.0);
    for (std::size_t n = 0; n <= d; ++n)
        for (std::size_t k = 0; k <= d; ++k) q[k] += c[n] * T[n][k];
    return q;
}

// Problem B: max over |z| = 1 of |P(z)|, P(z) = sum_n c[n] z^n with real c.
// For z = e^{i theta}:  |P(z)|^2 = r_0 + 2 sum_{m>=1} r_m cos(m theta),  r_m = sum_j c_j c_{j+m},
// and cos(m theta) = T_m(cos theta). With x = cos theta, which runs over all of [-1, 1],
// |P|^2 becomes the polynomial g = r_0 T_0 + 2 sum_{m>=1} r_m T_m, so the answer is sqrt(max of g on [-1, 1]).
inline double max_abs_circle(const Poly& c) {
    if (c.empty()) return 0.0;
    const std::size_t d = c.size() - 1;
    Poly g_cheb(d + 1, 0.0);
    for (std::size_t m = 0; m <= d; ++m) {
        double r = 0.0;
        for (std::size_t j = 0; j + m <= d; ++j) r += c[j] * c[j + m];
        g_cheb[m] = (m == 0) ? r : 2.0 * r;
    }
    return std::sqrt(max_abs_interval(from_chebyshev(g_cheb)));
}

// Chebyshev coefficients of q from its values at the N = deg + 1 Chebyshev nodes x_j = cos(pi (j + 1/2) / N):
// c_k = (2/N) sum_j q(x_j) cos(k theta_j), with c_0 halved; exact for degree < N by the discrete
// orthogonality of T_0, ..., T_{N-1} at these nodes (Mason & Handscomb, Chebyshev Polynomials, 2003,
// Ch. 6). Stable at high degree, unlike to_chebyshev, whose monomial table grows like 2^n.
inline Poly chebyshev_by_sampling(const Poly& q_in) {
    const Poly q = trim(q_in);
    if (q.empty()) return {};
    const std::size_t N = q.size();
    std::vector<double> theta(N), val(N);
    for (std::size_t j = 0; j < N; ++j) {
        theta[j] = std::acos(-1.0) * (double(j) + 0.5) / double(N);
        val[j] = eval(q, std::cos(theta[j]));
    }
    Poly c(N, 0.0);
    for (std::size_t k = 0; k < N; ++k) {
        double sum = 0.0;
        for (std::size_t j = 0; j < N; ++j) sum += val[j] * std::cos(double(k) * theta[j]);
        c[k] = 2.0 * sum / double(N);
    }
    c[0] /= 2.0;
    return c;
}

// sum_k c[k] T_k(x), by Clenshaw's recurrence b_k = c_k + 2x b_{k+1} - b_{k+2} (stable at high degree).
inline double clenshaw(const Poly& c, double x) {
    double b1 = 0.0, b2 = 0.0;
    for (std::size_t k = c.size(); k-- > 1;) {
        const double b0 = c[k] + 2.0 * x * b1 - b2;
        b2 = b1;
        b1 = b0;
    }
    return c.empty() ? 0.0 : c[0] + x * b1 - b2;
}

}  // namespace polymax
