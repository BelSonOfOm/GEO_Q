// Heat diffusion: exact compilation (what Cobble does) against the truncated Chebyshev method with an
// error budget (Claude, 5 Nov 2026).
#include "heat.h"
#include <cmath>
#include <cstdio>

using namespace bench;

int main() {
    const std::size_t n = 3;
    const double eps = 1e-6;
    Term P = random_walk(n);
    std::printf("random walk on the periodic 8 x 8 grid: k_P = %llu, alpha_P = %.3f, Hermitian: %s\n",
                (unsigned long long)cost(P).k, cost(P).alpha, info(P).hermitian ? "yes" : "no");
    std::printf("target P^t, error budget eps = %.0e (in operator norm)\n\n", eps);
    std::printf("     t | exact: method   k     alpha    alpha*k | truncated: K     k     alpha     error bound   alpha*k | gain | sqrt-law k\n");
    for (std::size_t t : {16u, 32u, 64u, 128u, 256u, 512u, 1024u}) {
        Term exact = heat_steps(P, t, 0.0), approx = heat_steps(P, t, eps);
        const Cost& a = cost(exact);
        const Cost& b = cost(approx);
        const Info& I = info(approx);
        const double law = double(cost(P).k) * std::sqrt(2.0 * double(t) * std::log(2.0 / eps));
        std::printf("%6zu | %-9s %7llu %9.4f %10.1f | %-9s %4zu %5llu %9.4f %13.2e %9.1f | %4.1fx | %7.0f\n", t,
                    method_name(info(exact).method), (unsigned long long)a.k, a.alpha, a.alpha * double(a.k),
                    method_name(I.method), I.realised_cheb.size() - 1, (unsigned long long)b.k, b.alpha, b.eps,
                    b.alpha * double(b.k), (a.alpha * double(a.k)) / (b.alpha * double(b.k)), law);
    }
}
