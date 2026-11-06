// Checks for act() and same_matrix() in GEO_Q.h (written by Claude, 5 Nov 2026): the meaning of a
// term applied to a vector, against dense matrices built independently from the definitions.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "GEO_Q.h"
#include <random>

using namespace cobble;
using Mat = Eigen::MatrixXcd;

namespace {
Term L(const char* s) { return leaf(PauliString::from_string(s)); }

Mat kron(const Mat& A, const Mat& B) {   // A on the high qubits, B on the low ones
    Mat K(A.rows() * B.rows(), A.cols() * B.cols());
    for (Eigen::Index i = 0; i < A.rows(); ++i)
        for (Eigen::Index j = 0; j < A.cols(); ++j) K.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
    return K;
}

// The matrix of a term, straight from the definitions (Cobble Sec. 4.1).
Mat dense(const Term& t) {
    if (const Leaf* l = std::get_if<Leaf>(&t->kind)) return matrix(l->p);
    if (const Adjoint* a = std::get_if<Adjoint>(&t->kind)) return dense(a->a).adjoint();
    if (const Sum* s = std::get_if<Sum>(&t->kind)) {
        Mat M = Mat::Zero(1 << t->qubits, 1 << t->qubits);
        for (std::size_t j = 0; j < s->t.size(); ++j) M += s->c[j] * dense(s->t[j]);
        return M;
    }
    if (const Product* p = std::get_if<Product>(&t->kind)) {
        Mat M = dense(p->t[0]);
        for (std::size_t j = 1; j < p->t.size(); ++j) M = M * dense(p->t[j]);
        return M;
    }
    if (const DirectSum* d = std::get_if<DirectSum>(&t->kind)) {
        const Eigen::Index B = 1 << d->t[0]->qubits;
        Mat M = Mat::Zero(1 << t->qubits, 1 << t->qubits);
        for (std::size_t j = 0; j < d->t.size(); ++j) M.block(Eigen::Index(j) * B, Eigen::Index(j) * B, B, B) = dense(d->t[j]);
        return M;
    }
    if (const Tensor* x = std::get_if<Tensor>(&t->kind)) {
        Mat M = dense(x->t[0]);
        for (std::size_t j = 1; j < x->t.size(); ++j) M = kron(dense(x->t[j]), M);   // t[0] lowest
        return M;
    }
    if (const Unitary* u = std::get_if<Unitary>(&t->kind)) return u->U;
    const Poly& P = std::get<Poly>(t->kind);
    const Mat A = dense(P.a);
    Mat M = Mat::Zero(A.rows(), A.cols()), pw = Mat::Identity(A.rows(), A.cols());
    for (double a : P.coeff) { M += a * pw; pw = pw * A; }
    return M;
}

Eigen::VectorXcd random_vector(std::size_t qubits, std::mt19937& rng) {
    std::normal_distribution<double> g;
    Eigen::VectorXcd x(1 << qubits);
    for (Eigen::Index i = 0; i < x.size(); ++i) x[i] = Coeff(g(rng), g(rng));
    return x;
}

void agree(const Term& t) {
    static std::mt19937 rng(5);
    const Mat M = dense(t);
    for (int r = 0; r < 3; ++r) {
        const Eigen::VectorXcd x = random_vector(t->qubits, rng);
        CHECK((act(t, x) - M * x).norm() <= 1e-10 * (1 + (M * x).norm()));
        CHECK((act(t, x, true) - M.adjoint() * x).norm() <= 1e-10 * (1 + (M.adjoint() * x).norm()));
    }
}
}

TEST_CASE("act agrees with the dense matrix, for every kind of node") {
    Term A = L("XY"), B = L("ZI"), C = L("YZ");
    agree(A);
    agree(adjoint(sum({Coeff(1, 2), Coeff(0.5, 0)}, {A, B})));
    agree(sum({Coeff(1, -1), Coeff(0.3, 0), Coeff(0, 2)}, {A, B, C}));
    agree(product({sum({1.0, 2.0}, {A, B}), C, sum({Coeff(0, 1), Coeff(1, 0)}, {B, C})}));
    agree(direct_sum({A, sum({2.0}, {B})}));
    agree(direct_sum({A, B, C, product({A, C})}));
    agree(tensor({L("X"), L("Z")}));
    agree(tensor({sum({1.0, 0.5}, {L("X"), L("Y")}), L("ZX"), sum({Coeff(0, 1), Coeff(2, 0)}, {L("Z"), L("Y")})}));
    agree(poly(sum({0.3, 0.4}, {A, B}), {0.5, -1.0, 0.0, 2.0}));
    agree(adjoint(product({tensor({L("X"), sum({Coeff(1, 1), Coeff(1, 0)}, {L("Y"), L("Z")})}), direct_sum({L("Y"), L("Z")})})));
    Mat S = Mat::Zero(4, 4);                                   // cyclic shift on 4 points: (S f)(x) = f(x + 1)
    for (int x = 0; x < 4; ++x) S(x, (x + 1) % 4) = 1.0;
    Term U = unitary_leaf("S", S);
    agree(U);
    agree(sum({Coeff(1, 0), Coeff(0.5, 0)}, {U, adjoint(U)}));
    agree(tensor({U, L("X")}));
}

TEST_CASE("unitary leaves") {
    Mat S = Mat::Zero(4, 4);
    for (int x = 0; x < 4; ++x) S(x, (x + 1) % 4) = 1.0;
    Term U = unitary_leaf("S", S);
    CHECK(U->qubits == 2);
    CHECK(cost(U).k == 1); CHECK(cost(U).alpha == 1.0); CHECK(cost(U).m == 0);
    CHECK_FALSE(info(U).hermitian);                            // S is not S^dagger
    CHECK(info(sum({1, 1}, {U, adjoint(U)})).hermitian == false);   // Cobble-style judgment: children must be Hermitian
    CHECK(unitary_leaf("other name", S).get() == U.get());    // same matrix, same node
    CHECK_THROWS_AS(unitary_leaf("bad", Mat::Ones(4, 4)), std::invalid_argument);
    CHECK_THROWS_AS(unitary_leaf("bad", Mat::Identity(3, 3)), std::invalid_argument);
}

TEST_CASE("same_matrix") {
    Term A = L("XY"), B = L("ZI");
    CHECK(same_matrix(sum({1, 2}, {A, B}), sum({2, 1}, {B, A})));
    CHECK(same_matrix(product({A, A}), L("II")));                 // a Pauli string squares to I
    CHECK_FALSE(same_matrix(A, B));
    CHECK_FALSE(same_matrix(sum({1.0, 1e-6}, {A, B}), A));         // a small but real difference
    CHECK_THROWS_AS(same_matrix(A, L("X")), std::invalid_argument);
}
