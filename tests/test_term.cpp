// Checks for the Cobble term builders in GEO_Q.h (written by Claude, 4 Nov 2026): hash-consing,
// the canonical label (equal up to flattening and reordering), and rejection of malformed terms.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "GEO_Q.h"

using namespace cobble;

namespace {
Term L(const char* s) { return leaf(PauliString::from_string(s)); }
}

TEST_CASE("identical trees are the same node") {
    Term A = L("XI"), B = L("ZZ");
    CHECK(L("XI").get() == A.get());
    CHECK(sum({1, 2}, {A, B}).get() == sum({1, 2}, {A, B}).get());
    CHECK(sum({Coeff(1, 0.0), Coeff(2, 0)}, {A, B}).get() == sum({Coeff(1, -0.0), Coeff(2, 0)}, {A, B}).get());
}

TEST_CASE("the canonical label: equal up to flattening and reordering, nothing more") {
    Term A = L("XI"), B = L("ZZ"), C = L("YX"), D = L("IZ");
    // reordered sums: different trees, same label
    CHECK(sum({1, 2}, {A, B})->id != sum({2, 1}, {B, A})->id);
    CHECK(sum({1, 2}, {A, B})->canon == sum({2, 1}, {B, A})->canon);
    // nested and flat sums, coefficients multiplied through
    Term nested = sum({1, 1}, {sum({1, 1}, {sum({1, 1}, {A, B}), C}), D});
    Term flat = sum({1, 1, 1, 1}, {A, B, C, D});
    CHECK(nested->id != flat->id);
    CHECK(nested->canon == flat->canon);
    CHECK(sum({2}, {sum({1, 3}, {A, B})})->canon == sum({2, 6}, {A, B})->canon);
    // no fusion: 2A + 3A is not labelled like 5A
    CHECK(sum({2, 3}, {A, A})->canon != sum({5}, {A})->canon);
    // products and tensor products: flattened, never reordered
    CHECK(product({product({A, B}), C})->canon == product({A, product({B, C})})->canon);
    CHECK(product({A, B})->canon != product({B, A})->canon);
    CHECK(tensor({tensor({A, B}), C})->canon == tensor({A, tensor({B, C})})->canon);
    CHECK(tensor({A, B})->canon != tensor({B, A})->canon);
    // adjoints and polynomials follow their child's label
    CHECK(adjoint(sum({1, 1}, {A, B}))->canon == adjoint(sum({1, 1}, {B, A}))->canon);
    CHECK(poly(sum({1, 1}, {A, B}), {0, 0, 1})->canon == poly(sum({1, 1}, {B, A}), {0, 0, 1})->canon);
    // exact coefficients: 0.1 + 0.2 is not 0.3 in binary
    CHECK(sum({0.1 + 0.2}, {A})->id != sum({0.3}, {A})->id);
}

TEST_CASE("qubit counts") {
    Term A = L("XI"), B = L("ZZ"), C = L("YX"), D = L("IZ");
    CHECK(tensor({A, B})->qubits == 4);
    CHECK(direct_sum({A, B})->qubits == 3);
    CHECK(direct_sum({A, B, C, D})->qubits == 4);
    CHECK(sum({1, 1}, {A, B})->qubits == 2);
}

TEST_CASE("malformed terms are rejected") {
    Term A = L("XI"), B = L("ZZ"), C = L("YX");
    CHECK_THROWS_AS(sum({1, 1}, {A, L("X")}), std::invalid_argument);
    CHECK_THROWS_AS(sum({0, 0}, {A, B}), std::invalid_argument);
    CHECK_THROWS_AS(product({A}), std::invalid_argument);
    CHECK_THROWS_AS(direct_sum({A, B, C}), std::invalid_argument);
    CHECK_THROWS_AS(sum({std::nan("")}, {A}), std::invalid_argument);
    CHECK_THROWS_AS(poly(A, {}), std::invalid_argument);
}

TEST_CASE("unused nodes are freed") {
    Term C = L("YX"), D = L("IZ");
    std::uint64_t old_id;
    {
        Term tmp = sum({7, 7}, {C, D});
        old_id = tmp->id;
    }
    CHECK(sum({7, 7}, {C, D})->id != old_id);
}
