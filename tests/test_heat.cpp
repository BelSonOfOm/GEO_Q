// Checks for the error column, the truncated Chebyshev method and the heat benchmark (Claude, 5 Nov 2026).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../bench/heat.h"

using namespace bench;

TEST_CASE("the walk operator: alpha 1, four queries, Hermitian as an adjoint-symmetric sum") {
    Term P = random_walk(2);
    CHECK(cost(P).k == 4);
    CHECK(cost(P).alpha == doctest::Approx(1.0));
    CHECK(info(P).hermitian);
    CHECK(cost(P).eps == 0.0);
}

TEST_CASE("exact compilation is unchanged by the new method: budget 0 gives k = 4t, error 0") {
    Term P = random_walk(2);
    for (std::size_t t : {8u, 40u, 100u}) {
        Term e = heat_steps(P, t, 0.0);
        CHECK_FALSE(info(e).rows[5].has_value());
        CHECK(cost(e).k == 4 * t);
        CHECK(cost(e).alpha == doctest::Approx(1.0).epsilon(1e-9));
        CHECK(cost(e).eps == 0.0);
    }
}

TEST_CASE("truncated: the realised polynomial is within the reported error, which fits the budget") {
    Term P = random_walk(2);                                   // 4 x 4 grid, 16 x 16 matrices
    const Eigen::MatrixXcd X = dense_of(P);
    for (std::size_t t : {20u, 60u, 200u}) {
        for (double budget : {1e-3, 1e-8}) {
            Term a = heat_steps(P, t, budget);
            REQUIRE(info(a).rows[5].has_value());
            const Cost& row = *info(a).rows[5];
            CHECK(row.eps <= budget * (1 + 1e-9));
            Eigen::MatrixXcd Pt = Eigen::MatrixXcd::Identity(X.rows(), X.cols());
            for (std::size_t s = 0; s < t; ++s) Pt = Pt * X;
            const double err = Eigen::JacobiSVD<Eigen::MatrixXcd>(Pt - realised_matrix(a)).singularValues()(0);
            CHECK(err <= row.eps * (1 + 1e-6) + 1e-12);
            CHECK(row.k <= 4 * t);                            // never more queries than exact
            // chosen only when strictly cheaper (t = 20 at 1e-8 keeps the full degree: a tie, so QSVT)
            if (row.alpha * double(row.k) < 4.0 * double(t)) CHECK(info(a).method == PolyMethod::Truncated);
        }
    }
}

TEST_CASE("square-root law: four times the steps, about twice the degree") {
    Term P = random_walk(2);
    auto K = [&](std::size_t t) { Term a = heat_steps(P, t, 1e-6); return double(info(a).realised_cheb.size() - 1); };
    for (std::size_t t : {64u, 256u}) {
        const double r = K(4 * t) / K(t);
        CHECK(r > 1.6);
        CHECK(r < 2.4);
    }
}

TEST_CASE("error propagation through sums and tensor products") {
    Term P = random_walk(2);
    Term a = heat_steps(P, 50, 1e-4);
    const double e = cost(a).eps;
    REQUIRE(e > 0.0);
    CHECK(cost(sum({3.0}, {a})).eps == doctest::Approx(3 * e));                    // sum: |c| eps
    Term I = leaf(PauliString::from_string("II"));
    CHECK(cost(tensor({a, I})).eps == doctest::Approx(e));                          // (alpha + eps) * 1 - alpha * 1
    CHECK(cost(product({a, a})).eps ==
          doctest::Approx((cost(a).alpha + e) * (cost(a).alpha + e) - cost(a).alpha * cost(a).alpha));
}
