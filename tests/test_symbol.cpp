// Checks for bench/symbol_route.h (written by Claude, 5 Nov 2026).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../bench/symbol_route.h"

using namespace bench;

TEST_CASE("cyclic shifts") {
    auto S = cyclic_shift(8, 1);
    Eigen::VectorXcd f = Eigen::VectorXcd::LinSpaced(8, 0, 7);
    CHECK((S * f)(3).real() == doctest::Approx(4));          // (S f)(3) = f(4)
    CHECK((S * f)(7).real() == doctest::Approx(0));          // wraps round
    CHECK((cyclic_shift(8, -1) * S - Eigen::MatrixXcd::Identity(8, 8)).norm() < 1e-14);
}

TEST_CASE("symbol route on laplacian-filter: same matrix, alpha at the floor") {
    for (std::size_t n : {2u, 3u}) {
        Term H = pauli_sum(second_difference(n), n);
        Pair P = laplacian_filter(H, H);
        SymbolResult r = symbol_route(P.prog, n, n);
        REQUIRE(r.translation_invariant);
        CHECK(r.separable);
        CHECK(same_matrix(r.term, P.prog));
        CHECK(cost(r.term).alpha >= r.norm * (1 - 1e-9));           // never below ||T|| (Def. 2.2)
        CHECK(cost(r.term).alpha == doctest::Approx(r.norm));       // and here exactly at it
    }
}

TEST_CASE("symbol route: non-separable case and a refusal") {
    Term H = pauli_sum(second_difference(2), 2);
    Term I = leaf(PauliString::from_string("II"));
    Term L2 = sum({1.0, 1.0}, {tensor({H, I}), tensor({I, H})});
    SymbolResult r = symbol_route(L2, 2, 2);
    REQUIRE(r.translation_invariant);
    CHECK_FALSE(r.separable);
    CHECK(same_matrix(r.term, L2));
    // Z on one qubit is not translation-invariant, so the route declines.
    SymbolResult z = symbol_route(tensor({leaf(PauliString::from_string("ZI")), I}), 2, 2);
    CHECK_FALSE(z.translation_invariant);
    CHECK(z.term == nullptr);
}

TEST_CASE("optimise is never worse than the usual form") {
    for (std::size_t n : {2u, 3u}) {
        Term H = pauli_sum(second_difference(n), n);
        Term I = leaf(PauliString::from_string(std::string(n, 'I')));
        Pair P = laplacian_filter(H, H);
        Term L2 = sum({1.0, 1.0}, {tensor({H, I}), tensor({I, H})});
        for (const Term& t : {P.targ, L2}) {
            Term o = optimise(t, n, n);
            CHECK(same_matrix(o, t));
            CHECK(cost(o).alpha * double(cost(o).k) <= cost(t).alpha * double(cost(t).k));
        }
    }
    // Not translation-invariant: the symbol route declines. Since 5 Nov (evening) the boundary route
    // may still rewrite it (here Z (x) I, a tensor of two leaves, becomes one reflection leaf: k = 1, not 2);
    // the guarantee is the same matrix and never a higher cost.
    Term z = tensor({leaf(PauliString::from_string("ZI")), leaf(PauliString::from_string("II"))});
    CHECK_FALSE(symbol_route(z, 2, 2).translation_invariant);
    CHECK(same_matrix(optimise(z, 2, 2), z));
    CHECK(cost(optimise(z, 2, 2)).alpha * double(cost(optimise(z, 2, 2)).k) <= cost(z).alpha * double(cost(z).k));
}

TEST_CASE("one-axis grids: the 1D Laplacian on 32 points in three shifts, alpha at the floor") {
    Term H = pauli_sum(second_difference(5), 5);
    SymbolResult r = symbol_route(H, 5, 0);
    REQUIRE(r.translation_invariant);
    CHECK(same_matrix(r.term, H));
    CHECK(cost(r.term).k == 2);                 // S and S^-1; the identity costs no query (6 Nov, Cobble's Const)
    CHECK(cost(r.term).alpha == doctest::Approx(4.0));
    CHECK(r.norm == doctest::Approx(4.0));
    CHECK(cost(optimise(H, 5, 0)).alpha * double(cost(optimise(H, 5, 0)).k) == doctest::Approx(8.0));
}
