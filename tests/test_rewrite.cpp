// Checks for Cobble's rewrite optimiser ported to GEO_Q (GEO_Q/rewrite.h; written by Claude, 6 Nov 2026).
// Known answers: Cobble's Table 3 (tex/tbl3.tex), "Unoptimized" and "Optimized" queries x subnormalisation,
// for the four of its five benchmarks that GEO_Q can write (ols-ridge is not ported), built as in Cobble's
// cobble/examples/*.py with Pauli strings as the black boxes (Hermitian unitaries, alpha = 1, one query).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../bench/symbol_route.h"

using namespace cobble;

namespace {
Term L(const char* s) { return leaf(PauliString::from_string(s)); }
void costs(const Term& t, std::uint64_t k, double alpha) {   // alpha as printed in Table 3, one decimal
    CHECK(cost(t).k == k);
    CHECK(std::abs(cost(t).alpha - alpha) <= 0.05 + 1e-12);
}
}  // namespace

TEST_CASE("Table 3: simulation-example, 8 x 2.6 -> 4 x 2.0") {
    const Term X = L("X"), Y = L("Y");
    const Term XX = tensor({X, X}), YY = tensor({Y, Y});
    const Term A = sum({1.0, 1.0}, {XX, YY}), B = sum({1.0, -1.0}, {XX, YY});
    const Term H = sum({1.0, 0.3}, {A, B});
    costs(H, 8, 2.6);
    const Term o = optimize(H);
    costs(o, 4, 2.0);
    CHECK(same_matrix(o, H));
}

TEST_CASE("Table 3: regression-example, 12 x 16.0 -> 8 x 1.0") {
    const Term A = L("X"), B = L("Z");
    const Term S = sum({1.0, -1.0}, {A, B});
    const Term S2 = product({S, S});
    const Term f = sum({1.0, 1.0}, {S, sum({0.5}, {S2})});
    const Term g = sum({1.0, 1.0}, {S, sum({-0.5}, {S2})});
    const Term prog = product({f, g});
    costs(prog, 12, 16.0);
    const Term o = optimize(prog);
    costs(o, 8, 1.0);
    CHECK(same_matrix(o, prog));
    REQUIRE(std::holds_alternative<Poly>(o->kind));          // Cobble's target: poly(A - B, x^2 - x^4 / 4)
    const auto& c = std::get<Poly>(o->kind).coeff;
    CHECK(c.size() == 5);
    CHECK(c[2] == doctest::Approx(1.0));
    CHECK(c[4] == doctest::Approx(-0.25));
}

TEST_CASE("Table 3: penalized-coupler, 6 x 8.2 -> 3 x 6.2 (the identity is Cobble's Const: no query)") {
    const Term Z = L("Z"), I = L("I");
    const Term Z0 = tensor({Z, I}), Z1 = tensor({I, Z}), ZZ = tensor({Z, Z}), II = tensor({I, I});
    const Term H = sum({2.0, 2.0, 1.2, -1.0}, {Z0, Z1, ZZ, sum({1.0, -1.0, 1.0}, {Z0, Z1, II})});
    costs(H, 6, 8.2);
    const Term o = optimize(H);
    costs(o, 3, 6.2);
    CHECK(same_matrix(o, H));
}

TEST_CASE("Table 3: laplacian-filter, 8 x 59.3 -> 2 x 27.8 (Cobble's input rewritten into Cobble's target)") {
    const bench::Pair P = bench::laplacian_filter(bench::box(), bench::box());
    costs(P.prog, 8, 59.3);
    const Term o = optimize(P.prog);
    costs(o, 2, 27.8);
    CHECK(same_matrix(o, P.targ));                                // Cobble's target (up to the last bit of
    CHECK(cost(o).alpha == doctest::Approx(cost(P.targ).alpha));  // 1.0 - 2.9 against -1.9)
    CHECK(std::holds_alternative<Tensor>(o->kind));
}

TEST_CASE("Cobble's simple examples rewrite to Cobble's targets") {
    const Term A = L("X"), B = L("Z"), I = L("I");
    CHECK(rewrite(sum({0.5, 0.5}, {A, A}))->canon == A->canon);                                    // simple_1
    CHECK(rewrite(sum({1.0, -1.0}, {I, product({adjoint(A), A})}))->canon == poly(A, {1.0, 0.0, -1.0})->canon);   // simple_2
    const Term S = sum({1.0, -1.0}, {A, B});                                                       // simple_3
    const Term f = sum({1.0, 1.0}, {S, sum({0.5}, {product({S, S})})});
    const Term g = sum({1.0, 1.0}, {S, sum({-0.25}, {product({S, S})})});
    const Term o = rewrite(product({f, g}));
    REQUIRE(std::holds_alternative<Poly>(o->kind));
    const auto& c = std::get<Poly>(o->kind).coeff;
    const std::vector<double> want{0.0, 0.0, 1.0, 0.25, -0.125};
    REQUIRE(c.size() == want.size());
    for (std::size_t j = 0; j < c.size(); ++j) CHECK(c[j] == doctest::Approx(want[j]));
}

TEST_CASE("random terms: rewrites keep the matrix, optimize is never worse") {
    std::mt19937 rng(42);
    std::normal_distribution<double> gs;
    const char* ps[6] = {"XI", "ZZ", "IY", "YX", "II", "ZI"};
    auto rleaf = [&]() { return L(ps[rng() % 6]); };
    std::function<Term(int)> rterm = [&](int depth) -> Term {
        if (depth == 0) return rleaf();
        switch (rng() % 5) {
            case 0: return sum({gs(rng), Coeff(gs(rng), 0.5 * gs(rng)), gs(rng)}, {rterm(depth - 1), rterm(depth - 1), rleaf()});
            case 1: return product({rterm(depth - 1), rterm(depth - 1)});
            case 2: return adjoint(rterm(depth - 1));
            case 3: { const Term b = sum({gs(rng), gs(rng)}, {rleaf(), rleaf()}); return poly(b, {gs(rng), gs(rng), gs(rng)}); }
            default: return sum({1.0, 1.0}, {rterm(depth - 1), rterm(depth - 1)});
        }
    };
    for (int trial = 0; trial < 150; ++trial) {
        const Term t = rterm(3);
        const Term r = rewrite(t);
        if (!r) continue;
        CHECK(same_matrix(r, t, 2, 1e-8));
        const Term o = optimize(t);
        CHECK(cost(o).alpha * double(cost(o).k) <= cost(t).alpha * double(cost(t).k) * (1 + 1e-12));
    }
}

TEST_CASE("rewrites around rotations and direct sums") {
    const Term R = rotation(PauliString::from_string("XY"), 0);
    const Term t = sum({0.5, 0.5}, {product({R, L("ZI")}), product({R, L("ZI")})});
    const Term o = optimize(t);
    CHECK(cost(o).k < cost(t).k);
    CHECK(same_matrix(bind(o, {0.7}), bind(t, {0.7})));
    const Term A = sum({0.3, 0.7}, {L("X"), L("Z")});
    const Term d = direct_sum({A, A});                            // if x then A else A => A (x) I
    const Term od = rewrite(d);
    CHECK(std::holds_alternative<Tensor>(od->kind));
    CHECK(same_matrix(od, d));
}
