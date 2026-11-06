// Checks for the costs (k, alpha, m) and the Hermiticity judgment in GEO_Q.h (written by Claude,
// 5 Nov 2026). Expected values are worked out by hand from Cobble's Tables 1-2 and the decisions
// of 4-5 Nov; the derivation of each is in the comment next to it.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "GEO_Q.h"

using namespace cobble;

namespace {
Term L(const char* s) { return leaf(PauliString::from_string(s)); }
void check(const Term& t, std::uint64_t k, double alpha, std::size_t m) {
    CHECK(cost(t).k == k);
    CHECK(cost(t).alpha == doctest::Approx(alpha));
    CHECK(cost(t).m == m);
}
}

TEST_CASE("leaves, adjoints, sums, products, tensor products") {
    check(L("XZ"), 1, 1.0, 0);                                   // a unitary encodes itself
    CHECK(info(L("XZ")).hermitian);
    PauliString iX = PauliString::from_string("X");
    iX.k = 1;                                                    // i X is anti-Hermitian
    CHECK_FALSE(info(leaf(iX)).hermitian);

    Term S = sum({0.5, 0.5}, {L("X"), L("Z")});
    check(S, 2, 1.0, 1);                                         // k 1+1, alpha .5+.5, one index qubit
    CHECK(info(S).hermitian);
    Term C = sum({Coeff(1, 1), Coeff(2, 0)}, {L("X"), L("Z")});
    check(C, 2, std::sqrt(2.0) + 2.0, 1);                        // |1+i| + |2|
    CHECK_FALSE(info(C).hermitian);                              // complex coefficient
    check(adjoint(S), 2, 1.0, 1);

    check(product({S, S}), 4, 1.0, 2);                           // k 2+2, alpha 1*1, 1 flag + max(1, 1)
    check(product({S, S, S}), 6, 1.0, 3);                        // ceil(log2 3) = 2 flags + 1
    check(tensor({S, S}), 4, 1.0, 2);                            // own workspace each: 1 + 1
    check(tensor({L("X"), S}), 3, 1.0, 1);
}

TEST_CASE("direct sums: tolerance and rescaling") {
    Term threeZ = sum({3.0}, {L("Z")});                          // alpha 3, no index qubit
    Term D = direct_sum({L("X"), threeZ});
    check(D, 2, 3.0, 1);                                         // X rescaled: one rotation qubit
    CHECK(cost(D).rescaled == 1);
    Term E = direct_sum({L("X"), sum({1.0 + 1e-13}, {L("X")})}); // within 1e-12: nothing rescaled
    check(E, 2, 1.0 + 1e-13, 0);
    CHECK(cost(E).rescaled == 0);
    Term F = direct_sum({L("X"), sum({1.0 + 1e-11}, {L("X")})}); // outside: rescaled
    CHECK(cost(F).rescaled == 1);
    CHECK(cost(direct_sum({D, D})).rescaled == 2);               // counted through the tree
}

TEST_CASE("Poly: every row of 4x^2 - 4x^4 of a Hermitian leaf") {
    Term P = poly(L("Z"), {0, 0, 4, 0, -4});
    const Info& I = info(P);
    // LCU: queries 2 + 4, alpha l1 = 8, ancillas ceil(log2 4) + 4
    CHECK(I.rows[0]->k == 6); CHECK(I.rows[0]->alpha == doctest::Approx(8)); CHECK(I.rows[0]->m == 6);
    // Horner: queries 4, alpha 8, ancillas 2*4
    CHECK(I.rows[1]->k == 4); CHECK(I.rows[1]->alpha == doctest::Approx(8)); CHECK(I.rows[1]->m == 8);
    // QSVT, even polynomial: queries 4, alpha = max |4x^2 - 4x^4| = 1, one ancilla
    CHECK(I.rows[2]->k == 4); CHECK(I.rows[2]->alpha == doctest::Approx(1)); CHECK(I.rows[2]->m == 1);
    // GQET: 4x^2 - 4x^4 = (T_0 - T_4)/2, so max |(1 - z^4)/2| on the circle = 1
    CHECK(I.rows[3]->k == 4); CHECK(I.rows[3]->alpha == doctest::Approx(1)); CHECK(I.rows[3]->m == 1);
    // Ko: degree 4 already a power of two; 2*(16 - 1) queries; q(cos t) = sin^2(2t), max 1 on the
    // 16 points, so alpha = sqrt 2; ancillas log2 4 + 4
    CHECK(I.rows[4]->k == 30); CHECK(I.rows[4]->alpha == doctest::Approx(std::sqrt(2.0))); CHECK(I.rows[4]->m == 6);
    // cheapest alpha*k: QSVT and GQET tie at 4 with one ancilla each; the order picks QSVT
    CHECK(I.method == PolyMethod::QSVT);
    check(P, 4, 1.0, 1);
    CHECK(I.hermitian);
}

TEST_CASE("Poly: mixed parity, the scaling alpha_M, and a non-Hermitian base") {
    Term f = poly(L("Z"), {1.0, 0.5});                           // 1 + 0.5 x, as in laplacian-filter
    const Info& I = info(f);
    CHECK(I.rows[2]->k == 1);                                    // parts of degree 0 and 1
    CHECK(I.rows[2]->alpha == doctest::Approx(1.5));             // max|1| + max|0.5 x|
    CHECK(I.rows[2]->m == 2);
    CHECK(I.method == PolyMethod::LCU);                          // alpha*k = 1.5 for LCU, QSVT, GQET; LCU and GQET
    check(f, 1, 1.5, 1);                                         // have 1 ancilla; the order picks LCU

    Term S = sum({0.5, 1.5}, {L("X"), L("Z")});                  // alpha_M = 2
    Term g = poly(S, {0, 1, 0, 1});                              // x + x^3, so q(x) = 2x + 8x^3
    CHECK(info(g).rows[0]->alpha == doctest::Approx(10.0));      // l1 of q
    CHECK(info(g).rows[2]->alpha == doctest::Approx(10.0));      // odd, increasing: max at x = 1
    CHECK(info(g).rows[0]->k == 2 * (1 + 3));                    // k_M = 2

    PauliString iX = PauliString::from_string("X");
    iX.k = 1;
    Term h = poly(leaf(iX), {0, 0, 1});
    CHECK(info(h).rows[0].has_value());
    CHECK_FALSE(info(h).rows[2].has_value());                    // QSVT, GQET, Ko need Hermitian M
    CHECK_FALSE(info(h).rows[4].has_value());
}

TEST_CASE("Poly degree: list kept as written, true degree used, constants rejected") {
    CHECK_THROWS_AS(poly(L("Z"), {1.0, 0.0}), std::invalid_argument);
    CHECK_THROWS_AS(poly(L("Z"), {}), std::invalid_argument);
    Term t = poly(L("Z"), {0.0, 1.0, 0.0});
    CHECK(std::get<Poly>(t->kind).coeff.size() == 3);            // stored as written
    check(t, 1, 1.0, 1);                                         // but costed as degree 1
}

TEST_CASE("Hermiticity of products and the memo") {
    Term S = sum({0.5, 1.0}, {L("XY"), L("ZI")});
    Term A = sum({Coeff(0, 1), Coeff(1, 0)}, {L("XI"), L("IZ")});   // not Hermitian
    CHECK(info(product({A, S, adjoint(A)})).hermitian);             // A S A^dagger
    CHECK_FALSE(info(product({A, S})).hermitian);
    CHECK(&info(S) == &info(S));                                    // computed once, kept in the node
}
