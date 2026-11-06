// Cost report for the rebuilt benchmarks (Claude, 5 Nov 2026). Prints every Poly method's row and
// the one chosen, with alpha * k (Cobble's total cost).
#include "benchmarks.h"
#include <cstdio>

using namespace bench;

static void row(const char* what, const Cost& c) {
    std::printf("    %-8s k = %5llu   alpha = %12.4f   m = %3zu   alpha*k = %14.2f\n", what,
                (unsigned long long)c.k, c.alpha, c.m, c.alpha * double(c.k));
}
static void poly_report(const char* name, const Term& t) {
    const Info& I = info(t);
    std::printf("%s (Hermitian base: %s)\n", name, info(std::get<Poly>(t->kind).a).hermitian ? "yes" : "no");
    for (int j = 0; j < 6; ++j)
        if (I.rows[std::size_t(j)]) row(method_name(PolyMethod(j)), *I.rows[std::size_t(j)]);
    std::printf("    chosen: %s\n\n", method_name(I.method));
}
static void term_report(const char* name, const Term& t) {
    std::printf("%s: %zu data qubits\n", name, t->qubits);
    row("total", cost(t));
    std::printf("\n");
}

int main() {
    poly_report("spectral thresholding, as written", spectral_thresholding(box(), false));
    poly_report("spectral thresholding, corrected", spectral_thresholding(box(), true));
    poly_report("hamiltonian simulation", hamiltonian_simulation(box()));

    Pair Lb = laplacian_filter(box(), box());
    term_report("laplacian-filter, black boxes, unoptimised", Lb.prog);
    term_report("laplacian-filter, black boxes, Cobble target", Lb.targ);
    for (std::size_t n : {2u, 3u}) {
        Term Hx = pauli_sum(second_difference(n), n);
        std::printf("1D stencil on %zu points: %zu Pauli strings, alpha = %.4f\n\n",
                    std::size_t(1) << n, std::get<Sum>(Hx->kind).t.size(), cost(Hx).alpha);
        Pair L = laplacian_filter(Hx, Hx);
        term_report("laplacian-filter, concrete, unoptimised", L.prog);
        term_report("laplacian-filter, concrete, Cobble target", L.targ);
        std::printf("    same matrix: %s\n\n", same_matrix(L.prog, L.targ) ? "yes" : "NO");
    }
}
