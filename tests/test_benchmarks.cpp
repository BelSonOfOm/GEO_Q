// Known answers from Cobble's own published tables (artifact tex/tbl3.tex and tbl4.tex), reproduced
// with GEO_Q.h, plus the meaning check of laplacian-filter on concrete stencils (Claude, 5 Nov 2026).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../bench/benchmarks.h"

using namespace bench;

TEST_CASE("Table 3, laplacian-filter: 8 x 59.3 unoptimised, 2 x 27.8 optimised") {
    Pair L = laplacian_filter(box(), box());
    CHECK(cost(L.prog).k == 8);
    CHECK(cost(L.prog).alpha == doctest::Approx(59.28));   // 1.5*1.5 + 6.3*1.5 + 1.5*6.1 + 6.3*6.1
    CHECK(cost(L.targ).k == 2);
    CHECK(cost(L.targ).alpha == doctest::Approx(27.84));   // 5.8 * 4.8
}

TEST_CASE("Table 4, spectral thresholding as written: LCU 99 x 5.27e5, Horner 19 x 5.27e5, QSVT 19 x 2.7") {
    const Term P = spectral_thresholding(box(), false);   // kept alive: info() lives in the node
    const Info& I = info(P);
    CHECK(I.rows[1]->k == 19);
    CHECK(I.rows[1]->alpha == doctest::Approx(5.27e5).epsilon(1e-3));
    CHECK(I.rows[2]->k == 19);
    CHECK(I.rows[2]->alpha == doctest::Approx(51.6 / 19).epsilon(2e-3));
    MESSAGE("LCU queries here: " << I.rows[0]->k << " (Cobble: 99)");
}

TEST_CASE("Table 4, Hamiltonian simulation: LCU 120 x 1066, Horner 15 x 1066, QSVT 29 x 2.0") {
    const Term P = hamiltonian_simulation(box());
    const Info& I = info(P);
    CHECK(I.rows[0]->k == 120);
    CHECK(I.rows[0]->alpha == doctest::Approx(1066).epsilon(1e-3));
    CHECK(I.rows[1]->k == 15);
    CHECK(I.rows[2]->k == 29);
    CHECK(I.rows[2]->alpha == doctest::Approx(2.0).epsilon(1e-2));
}

TEST_CASE("laplacian-filter on concrete stencils: prog and targ are the same matrix") {
    for (std::size_t n : {2u, 3u}) for (bool periodic : {true, false}) {
        Term Hx = pauli_sum(second_difference(n, periodic), n);
        CHECK(same_matrix(Hx, pauli_sum(second_difference(n, periodic), n)));
        Pair L = laplacian_filter(Hx, Hx);
        CHECK(same_matrix(L.prog, L.targ));
        CHECK_FALSE(same_matrix(L.prog, sum({1.0, 1e-6}, {L.targ, L.targ})));
    }
}
