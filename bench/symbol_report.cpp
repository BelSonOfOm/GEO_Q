// The symbol route against Cobble on laplacian-filter (Claude, 5 Nov 2026). Prints, for each form,
// k, alpha, alpha * k, the floor ||T|| and whether it is the same matrix as Cobble's input.
#include "symbol_route.h"
#include <cstdio>

using namespace bench;

static void line(const char* what, const Term& t, const Term& reference, double floor) {
    const Cost& c = cost(t);
    std::printf("  %-44s k = %3llu  alpha = %9.3f  alpha*k = %10.2f  alpha/||T|| = %5.3f  same matrix: %s\n", what,
                (unsigned long long)c.k, c.alpha, c.alpha * double(c.k), c.alpha / floor,
                same_matrix(t, reference) ? "yes" : "NO");
}

int main() {
    for (std::size_t n : {2u, 3u}) {
        const Eigen::Index N = Eigen::Index(1) << n;
        std::printf("laplacian-filter, periodic stencil, %lld x %lld grid (%zu qubits)\n", (long long)N, (long long)N, 2 * n);
        Term Hp = pauli_sum(second_difference(n), n);                         // Hx as Pauli strings
        Term I = leaf(PauliString::from_string(std::string(n, 'I')));
        Term S = unitary_leaf("S", cyclic_shift(N, 1)), Sinv = unitary_leaf("S^-1", cyclic_shift(N, -1));
        Term Hs = sum({-2.0, 1.0, 1.0}, {I, S, Sinv});                          // Hx as shifts
        Pair P = laplacian_filter(Hp, Hp), Q = laplacian_filter(Hs, Hs);
        SymbolResult r = symbol_route(P.prog, n, n);
        line("Cobble input (Pauli leaves)", P.prog, P.prog, r.norm);
        line("Cobble optimised (Pauli leaves)", P.targ, P.prog, r.norm);
        line("Cobble optimised, given shift leaves", Q.targ, P.prog, r.norm);
        line(r.separable ? "symbol route (separable)" : "symbol route", r.term, P.prog, r.norm);
        line("chosen (never worse than Cobble)", optimise(P.targ, n, n), P.prog, r.norm);
        std::printf("  floor ||T|| = %.3f, nonzero Fourier coefficients: %zu\n\n", r.norm, r.shifts);

        std::printf("2D Laplacian Hx(x)I + I(x)Hy, %lld x %lld grid\n", (long long)N, (long long)N);
        Term L2 = sum({1.0, 1.0}, {tensor({Hp, I}), tensor({I, Hp})});
        SymbolResult r2 = symbol_route(L2, n, n);
        line("written with Pauli leaves", L2, L2, r2.norm);
        line(r2.separable ? "symbol route (separable)" : "symbol route", r2.term, L2, r2.norm);
        line("chosen (never worse)", optimise(L2, n, n), L2, r2.norm);
        std::printf("  floor ||T|| = %.3f, nonzero Fourier coefficients: %zu\n\n", r2.norm, r2.shifts);
    }
}
