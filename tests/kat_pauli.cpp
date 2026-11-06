// Known-answer tests for pauli.h, from textbook results (Nielsen & Chuang, Quantum Computation and
// Quantum Information, 10th anniversary ed.). Chosen by Claude on 4 Nov 2026 at Charbel's request,
// in place of his own hand calculations. Each check names its source.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "pauli.h"

namespace {

PauliString P(const char* s) { return PauliString::from_string(s); }

// The same Pauli string times i^m.
PauliString times_i(PauliString p, unsigned m) {
    p.k = (p.k + m) % 4;
    return p;
}

bool same(const PauliString& a, const PauliString& b) {
    return a.n == b.n && a.x == b.x && a.z == b.z && a.k == b.k;
}

}  // namespace

TEST_CASE("the Pauli matrices (N&C section 2.1.3)") {
    using C = std::complex<double>;
    Eigen::Matrix2cd X, Y, Z;
    X << 0, 1, 1, 0;
    Y << 0, C(0, -1), C(0, 1), 0;
    Z << 1, 0, 0, -1;
    CHECK((matrix(P("I")) - Eigen::Matrix2cd::Identity()).norm() == 0.0);
    CHECK((matrix(P("X")) - X).norm() == 0.0);
    CHECK((matrix(P("Y")) - Y).norm() == 0.0);
    CHECK((matrix(P("Z")) - Z).norm() == 0.0);
}

TEST_CASE("products of single-qubit Paulis (N&C exercises 2.40 and 2.41)") {
    // [X,Y] = 2iZ, [Y,Z] = 2iX, [Z,X] = 2iY  and  {X,Y} = {Y,Z} = {Z,X} = 0,  X^2 = Y^2 = Z^2 = I
    // together give  XY = iZ, YZ = iX, ZX = iY  and the reverses with -i (that is, i^3).
    CHECK(same(P("X") * P("Y"), times_i(P("Z"), 1)));
    CHECK(same(P("Y") * P("Z"), times_i(P("X"), 1)));
    CHECK(same(P("Z") * P("X"), times_i(P("Y"), 1)));
    CHECK(same(P("Y") * P("X"), times_i(P("Z"), 3)));
    CHECK(same(P("Z") * P("Y"), times_i(P("X"), 3)));
    CHECK(same(P("X") * P("Z"), times_i(P("Y"), 3)));
    CHECK(same(P("X") * P("X"), P("I")));
    CHECK(same(P("Y") * P("Y"), P("I")));
    CHECK(same(P("Z") * P("Z"), P("I")));
}

TEST_CASE("commutation of single-qubit Paulis (N&C exercise 2.41)") {
    const char* L[] = {"X", "Y", "Z"};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) CHECK(commute(P(L[a]), P(L[b])) == (a == b));
    for (int a = 0; a < 3; ++a) CHECK(commute(P("I"), P(L[a])));
}

TEST_CASE("two qubits") {
    // X1 X2 and Z1 Z2 stabilise the Bell state (|00> + |11>)/sqrt2 (N&C section 10.5.1); a stabiliser
    // group is abelian, so they commute.
    CHECK(commute(P("XX"), P("ZZ")));
    // X on qubit 0 and Z on qubit 0, identity on qubit 1: they anticommute, as X and Z do (exercise 2.41).
    CHECK_FALSE(commute(P("XI"), P("ZI")));
    // Mixed product (A (x) B)(C (x) D) = (AC) (x) (BD) with the products above:
    // "XZ" . "ZX" = (X Z) (x) (Z X) = (-iY) (x) (iY) = Y (x) Y.
    CHECK(same(P("XZ") * P("ZX"), P("YY")));
}

TEST_CASE("Hermitian or not") {
    // The Pauli matrices are Hermitian (N&C section 2.1.3); XY = iZ is not, since (iZ)^dagger = -iZ.
    CHECK(is_hermitian(P("X")));
    CHECK(is_hermitian(P("Y")));
    CHECK(is_hermitian(P("Z")));
    CHECK(is_hermitian(P("XYZI")));
    CHECK_FALSE(is_hermitian(P("X") * P("Y")));
}
