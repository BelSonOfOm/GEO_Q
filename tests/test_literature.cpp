// Checks for what was added from the literature on 5 Nov 2026 (written by Claude):
// the certificate and moment order (Mahmud & Herrman, arXiv:2607.11596), the boundary route
// (after Kharazi et al., arXiv:2407.18347) and the Toffoli model (bench/gates.h).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../bench/symbol_route.h"
#include <random>

using namespace bench;

namespace {
Eigen::MatrixXcd circulant(std::size_t n, const std::vector<std::pair<long, double>>& taps) {
    const Eigen::Index N = Eigen::Index(1) << n;
    Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(N, N);
    for (auto [s, w] : taps) T += w * cyclic_shift(N, s);
    return T;
}
// Zero boundary values: the stencil with every wrap-around entry removed.
Eigen::MatrixXcd cut(std::size_t n, const std::vector<std::pair<long, double>>& taps) {
    const Eigen::Index N = Eigen::Index(1) << n;
    Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(N, N);
    for (auto [s, w] : taps)
        for (Eigen::Index x = 0; x < N; ++x)
            if (x + s >= 0 && x + s < N) T(x, x + s) += w;
    return T;
}
Eigen::MatrixXcd reflection(Eigen::Index N, Eigen::Index point) {   // R = I - 2 Pi_point
    Eigen::MatrixXcd R = Eigen::MatrixXcd::Identity(N, N);
    R(point, point) = -1.0;
    return R;
}
double norm2(const Eigen::MatrixXcd& A) { return Eigen::JacobiSVD<Eigen::MatrixXcd>(A).singularValues()(0); }
}  // namespace

TEST_CASE("controlled increment = increment on one more qubit, then X on the control (gates.h)") {
    for (std::size_t m : {1u, 2u, 3u, 4u}) {
        const Eigen::Index M = Eigen::Index(1) << m;
        const Eigen::MatrixXcd inc = cyclic_shift(M, -1);          // S^-1 |x> = |x + 1>
        const Eigen::MatrixXcd inc1 = cyclic_shift(2 * M, -1);     // on m + 1 qubits, control = lowest bit
        Eigen::MatrixXcd Xc = Eigen::MatrixXcd::Zero(2 * M, 2 * M);
        for (Eigen::Index z = 0; z < 2 * M; ++z) Xc(z ^ 1, z) = 1.0;
        Eigen::MatrixXcd ctrl = Eigen::MatrixXcd::Zero(2 * M, 2 * M);   // |0><0| (x) I + |1><1| (x) inc
        for (Eigen::Index x = 0; x < M; ++x) {
            ctrl(2 * x, 2 * x) = 1.0;
            for (Eigen::Index y = 0; y < M; ++y) ctrl(2 * y + 1, 2 * x + 1) = inc(y, x);
        }
        CHECK((Xc * inc1 - ctrl).norm() < 1e-12);
    }
    CHECK(gates::controlled_shift(5, 1) == 5);
    CHECK(gates::controlled_shift(5, -1) == 5);
    CHECK(gates::controlled_shift(5, 2) == 4);
    CHECK(gates::controlled_shift(5, 3) == 9);
    CHECK(gates::controlled_shift(5, 31) == 5);                  // 31 = -1 on 32 points
}

TEST_CASE("subcube cover: disjoint, and the product of its reflections is I - 2P") {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 50; ++trial) {
        const std::size_t D = 64;
        std::vector<bool> in(D);
        for (std::size_t x = 0; x < D; ++x) in[x] = (rng() % 3) == 0;
        std::vector<int> hits(D, 0);
        for (const auto& q : gates::cubes(in))
            for (std::size_t x = 0; x < D; ++x)
                if ((x & q.mask) == q.value) ++hits[x];
        for (std::size_t x = 0; x < D; ++x) CHECK(hits[x] == (in[x] ? 1 : 0));
    }
    std::vector<bool> edge(8, false);
    edge[7] = true;                                              // one point of a 3-qubit line
    CHECK(gates::cubes(edge).size() == 1);
    CHECK(gates::controlled_reflection(edge) == 2);              // C^3 Z: 2 temporary ANDs
    std::vector<bool> line(64, false);                           // x1 = 7 on an 8 x 8 grid
    for (std::size_t x2 = 0; x2 < 8; ++x2) line[7 + 8 * x2] = true;
    CHECK(gates::cubes(line).size() == 1);
    CHECK(gates::controlled_reflection(line) == 2);
}

TEST_CASE("norm from the coefficients alone equals the dense norm") {
    std::mt19937 rng(11);
    std::normal_distribution<double> g;
    for (std::size_t n1 : {2u, 3u})
        for (std::size_t n2 : {0u, 2u}) {
            const Eigen::Index N1 = Eigen::Index(1) << n1, N2 = Eigen::Index(1) << n2;
            Eigen::MatrixXcd C = Eigen::MatrixXcd::Zero(N1, N2);
            Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(N1 * N2, N1 * N2);
            for (int j = 0; j < 4; ++j) {
                const long a = long(rng() % N1), b = long(rng() % N2);
                const Coeff c(g(rng), g(rng));
                C(a, b) += c;
                T += c * grid_shift(n1, n2, a, b);
            }
            CHECK(symbol_norm(C) == doctest::Approx(norm2(T)));
        }
}

TEST_CASE("certificate (Mahmud & Herrman, Prop. V.4)") {
    for (const auto& taps : std::vector<std::vector<std::pair<long, double>>>{
             {{-1, 1}, {0, -2}, {1, 1}}, {{0, 1}, {-1, -1}}, {{-2, 1}, {-1, -4}, {0, 6}, {1, -4}, {2, 1}}}) {
        SymbolResult r = symbol_route(pauli_sum(circulant(4, taps), 4), 4, 0);
        REQUIRE(r.translation_invariant);
        CHECK(r.certified);
        CHECK(cost(r.term).alpha == doctest::Approx(r.norm));
    }
    // 1 + S - S^2: no frequency where the three terms share a phase, so sum |c| = 3 > ||T||.
    SymbolResult r = symbol_route(pauli_sum(circulant(3, {{0, 1}, {1, 1}, {2, -1}}), 3), 3, 0);
    REQUIRE(r.translation_invariant);
    CHECK_FALSE(r.certified);
    CHECK(cost(r.term).alpha > r.norm * 1.01);
    CHECK(r.symbol_norm == doctest::Approx(r.norm));
}

TEST_CASE("moment order: which derivative a stencil approximates") {
    auto m1 = [](const std::vector<std::pair<long, double>>& taps) { return symbol_route(pauli_sum(circulant(4, taps), 4), 4, 0).moments; };
    Moments lap = m1({{-1, 1}, {0, -2}, {1, 1}});
    CHECK(lap.order == 2);
    CHECK(std::get<2>(lap.leading[0]).real() == doctest::Approx(1.0));   // h^2 f''
    Moments adv = m1({{0, 1}, {-1, -1}});
    CHECK(adv.order == 1);
    CHECK(std::get<2>(adv.leading[0]).real() == doctest::Approx(1.0));   // f(x) - f(x - h) ~ h f'
    Moments bih = m1({{-2, 1}, {-1, -4}, {0, 6}, {1, -4}, {2, 1}});
    CHECK(bih.order == 4);
    CHECK(std::get<2>(bih.leading[0]).real() == doctest::Approx(1.0));   // h^4 f''''
    const Eigen::MatrixXcd L = circulant(2, {{-1, 1}, {0, -2}, {1, 1}}), I = Eigen::MatrixXcd::Identity(4, 4);
    Moments lap2 = symbol_route(pauli_sum(kron_hi(L, I) + kron_hi(I, L), 4), 2, 2).moments;
    CHECK(lap2.order == 2);
    CHECK(lap2.leading.size() == 2);                                      // d1^2 + d2^2
}

TEST_CASE("Kharazi et al.'s Dirichlet and Neumann formulas, checked against the matrices") {
    for (std::size_t n : {2u, 3u, 4u}) {
        const Eigen::Index N = Eigen::Index(1) << n;
        Term I = leaf(PauliString::from_string(std::string(n, 'I')));
        Term S = unitary_leaf("S", cyclic_shift(N, 1)), Si = unitary_leaf("S^-1", cyclic_shift(N, -1));
        Term R0Si = unitary_leaf("R0 S^-1", reflection(N, 0) * cyclic_shift(N, -1));
        Term RNS = unitary_leaf("R_(N-1) S", reflection(N, N - 1) * cyclic_shift(N, 1));
        Term R0RN = unitary_leaf("R0 R_(N-1)", reflection(N, 0) * reflection(N, N - 1));
        Term LD = sum({-2.0, 0.5, 0.5, 0.5, 0.5}, {I, Si, R0Si, S, RNS});                 // Eq. 52
        Term LN = sum({-1.5, -0.5, 0.5, 0.5, 0.5, 0.5}, {I, R0RN, R0Si, Si, RNS, S});      // Eq. 54
        Eigen::MatrixXcd neumann = second_difference(n, false);
        neumann(0, 0) = neumann(N - 1, N - 1) = -1.0;
        CHECK((dense_of(LD) - second_difference(n, false)).norm() < 1e-12);
        CHECK((dense_of(LN) - neumann).norm() < 1e-12);
        CHECK(cost(LD).alpha == doctest::Approx(4.0));
        CHECK(cost(LN).alpha == doctest::Approx(4.0));
    }
}

TEST_CASE("boundary route: same matrix, and the published subnormalisations found automatically") {
    for (std::size_t n : {3u, 4u, 5u}) {
        const Eigen::Index N = Eigen::Index(1) << n;
        Term dir = pauli_sum(second_difference(n, false), n);
        DiagonalResult d = diagonal_route(dir, n, 0);
        REQUIRE(d.term);
        CHECK(same_matrix(d.term, dir));
        CHECK(cost(d.term).alpha == doctest::Approx(4.0));       // Kharazi et al.: lambda_D = 4
        CHECK(cost(d.term).k == 4);                              // -2 I costs no query (6 Nov)
        Eigen::MatrixXcd neu = second_difference(n, false);
        neu(0, 0) = neu(N - 1, N - 1) = -1.0;
        DiagonalResult e = diagonal_route(pauli_sum(neu, n), n, 0);
        REQUIRE(e.term);
        CHECK(same_matrix(e.term, pauli_sum(neu, n)));
        CHECK(cost(e.term).alpha == doctest::Approx(4.0));       // lambda_N = 4
        // fourth-order stencil (-1, 16, -30, 16, -1) / 12 with zero boundary values: no hand-built version
        Eigen::MatrixXcd f4 = cut(n, {{-2, -1.0 / 12}, {-1, 16.0 / 12}, {0, -30.0 / 12}, {1, 16.0 / 12}, {2, -1.0 / 12}});
        DiagonalResult h = diagonal_route(pauli_sum(f4, n), n, 0);
        REQUIRE(h.term);
        CHECK(same_matrix(h.term, pauli_sum(f4, n)));
        CHECK(cost(h.term).alpha == doctest::Approx(64.0 / 12));
    }
    // 2D Dirichlet Laplacian: Kharazi et al. give 4d = 8.
    const Eigen::MatrixXcd L = second_difference(2, false), I = Eigen::MatrixXcd::Identity(4, 4);
    Term dir2 = pauli_sum(kron_hi(L, I) + kron_hi(I, L), 4);
    DiagonalResult d2 = diagonal_route(dir2, 2, 2);
    REQUIRE(d2.term);
    CHECK(same_matrix(d2.term, dir2));
    CHECK(cost(d2.term).alpha == doctest::Approx(8.0));
}

TEST_CASE("optimise stays never worse, including on unstructured matrices") {
    std::mt19937 rng(3);
    std::normal_distribution<double> g;
    Eigen::MatrixXcd A(16, 16);
    for (Eigen::Index i = 0; i < 16; ++i)
        for (Eigen::Index j = 0; j < 16; ++j) A(i, j) = Coeff(g(rng), g(rng));
    Term t = pauli_sum(A, 4);
    Term o = optimise(t, 4, 0);
    CHECK(same_matrix(o, t));
    CHECK(cost(o).alpha * double(cost(o).k) <= cost(t).alpha * double(cost(t).k));
    Term dir = pauli_sum(second_difference(4, false), 4);
    Term od = optimise(dir, 4, 0);
    CHECK(same_matrix(od, dir));
    CHECK(cost(od).alpha * double(cost(od).k) < cost(dir).alpha * double(cost(dir).k));
}

TEST_CASE("Toffoli estimate: Pauli form = unary iteration only; shifts add their increments") {
    Term P = pauli_sum(second_difference(4), 4);
    CHECK(gates::per_call(P) == doctest::Approx(double(std::get<Sum>(P->kind).t.size()) - 1));
    SymbolResult r = symbol_route(P, 4, 0);                     // -2 I + S + S^-1
    CHECK(gates::per_call(r.term) == doctest::Approx(2 + 0 + 4 + 4));
}

TEST_CASE("sparse Pauli decomposition = the full one") {
    std::mt19937 rng(5);
    std::normal_distribution<double> g;
    for (std::size_t n : {2u, 3u, 4u}) {
        const Eigen::Index D = Eigen::Index(1) << n;
        Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(D, D);
        for (int j = 0; j < 3 * int(D); ++j) A(Eigen::Index(rng() % D), Eigen::Index(rng() % D)) = Coeff(g(rng), g(rng));
        Term full = pauli_sum(A, n), sparse = pauli_sum_sparse(A, n);
        CHECK(same_matrix(full, sparse));
        CHECK((dense_of(sparse) - A).norm() < 1e-10);
        CHECK(cost(full).k == cost(sparse).k);
        CHECK(cost(full).alpha == doctest::Approx(cost(sparse).alpha));
    }
}
