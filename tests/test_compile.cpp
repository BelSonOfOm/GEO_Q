// Checks for code generation (GEO_Q/compile.h) and rotations (written by Claude, 6 Nov 2026).
// Every compiled circuit is run on the simulator of Basics.h; alpha times its top-left block must equal the
// term's matrix, computed independently by act (GEO_Q.h). Where the cost model applies, the circuit's
// alpha, workspace and leaf calls must equal Cobble's cost rules.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "compile.h"
#include "../bench/heat.h"

using cobble::Term;
using cobble::Circuit;
using cobble::Coeff;

namespace {
std::mt19937 rng(606);
std::normal_distribution<double> gauss;

PauliString random_pauli(std::size_t n, bool hermitian = true) {
    const char L[4] = {'I', 'X', 'Y', 'Z'};
    std::string s(n, 'I');
    for (auto& c : s) c = L[rng() % 4];
    PauliString p = PauliString::from_string(s);
    if (!hermitian) p.k = (p.k + unsigned(rng() % 4)) % 4;
    else if (rng() % 2) p.k = (p.k + 2) % 4;   // a sign
    return p;
}
Term random_sum(std::size_t n, int terms, bool complex_coeffs, bool hermitian_strings = true) {
    std::vector<Coeff> c;
    std::vector<Term> t;
    for (int j = 0; j < terms; ++j) {
        c.push_back(complex_coeffs ? Coeff(gauss(rng), gauss(rng)) : Coeff(gauss(rng), 0.0));
        t.push_back(cobble::leaf(random_pauli(n, hermitian_strings)));
    }
    return cobble::sum(c, t);
}
// Compile, run on the simulator, compare with the term's matrix.
Circuit check(const Term& t, const std::vector<double>& theta = {}, double tol = 1e-9) {
    const Circuit c = cobble::compile(t);
    const Eigen::MatrixXcd B = cobble::block(c, theta);
    const Eigen::MatrixXcd T = bench::dense_of(theta.empty() ? t : cobble::bind(t, theta));
    // Frobenius norm <= sqrt(2^n) x operator norm, and the circuit's own error bound eps (fitted QSP phases)
    CHECK((B - T).norm() <= tol * std::max(1.0, T.norm()) + std::sqrt(double(B.rows())) * c.eps);
    CHECK(c.eps <= 1e-6 * std::max(1.0, c.alpha));
    return c;
}
void same_costs(const Term& t, const Circuit& c) {
    CHECK(c.alpha == doctest::Approx(cobble::cost(t).alpha));
    CHECK(c.workspace == cobble::cost(t).m);
    CHECK(c.leaf_calls == cobble::cost(t).k);
}
bool only_small_gates(const Circuit& c) {
    for (const auto& g : c.gates)
        if (g.targets.size() > 1) return false;
    return true;
}
}  // namespace

TEST_CASE("Pauli leaves, with every phase") {
    for (int trial = 0; trial < 40; ++trial) {
        const Term t = cobble::leaf(random_pauli(1 + rng() % 4, false));
        same_costs(t, check(t));
    }
}

TEST_CASE("shift and reflection leaves become increments and multi-controlled Z gates") {
    for (long a = -3; a <= 4; ++a) {
        const Term s = cobble::unitary_leaf("S", bench::cyclic_shift(8, a));
        const Circuit c = check(s);
        CHECK(only_small_gates(c));
    }
    for (long a = -1; a <= 2; ++a)
        for (long b = -1; b <= 1; ++b) {
            const Circuit c = check(cobble::unitary_leaf("S2", bench::grid_shift(2, 2, a, b)));
            CHECK(only_small_gates(c));
        }
    // the leaves of the boundary route (reflections times shifts), Dirichlet and Neumann, 1D and 2D
    for (std::size_t n : {3u, 4u}) {
        Eigen::MatrixXcd neu = bench::second_difference(n, false);
        neu(0, 0) = neu((1 << n) - 1, (1 << n) - 1) = -1.0;
        for (const Eigen::MatrixXcd& A : {bench::second_difference(n, false), neu}) {
            const bench::DiagonalResult d = bench::diagonal_route(bench::pauli_sum_sparse(A, n), n, 0);
            REQUIRE(d.term);
            const Circuit c = check(d.term);
            same_costs(d.term, c);
            CHECK(only_small_gates(c));
        }
    }
    const Eigen::MatrixXcd L = bench::second_difference(2, false), I = Eigen::MatrixXcd::Identity(4, 4);
    const bench::DiagonalResult d2 = bench::diagonal_route(bench::pauli_sum_sparse(bench::kron_hi(L, I) + bench::kron_hi(I, L), 4), 2, 2);
    REQUIRE(d2.term);
    same_costs(d2.term, check(d2.term));
}

TEST_CASE("sums (PREPARE - SELECT - PREPARE^dagger), complex coefficients included") {
    for (int trial = 0; trial < 10; ++trial) {
        const Term t = random_sum(1 + rng() % 3, 1 + int(rng() % 7), trial % 2 == 1);
        same_costs(t, check(t));
    }
    // the symbol route's output: the periodic Laplacian as shifts
    const Term H = bench::pauli_sum(bench::second_difference(4), 4);
    const bench::SymbolResult r = bench::symbol_route(H, 4, 0);
    const Circuit c = check(r.term);
    same_costs(r.term, c);
    CHECK(c.alpha == doctest::Approx(4.0));
}

TEST_CASE("products (shared workspace and a success counter), tensors, adjoints, direct sums") {
    for (int trial = 0; trial < 6; ++trial) {
        const std::size_t n = 1 + rng() % 2;
        const Term A = random_sum(n, 3, true), B = random_sum(n, 2, false), C = random_sum(n, 5, true);
        const Term P = cobble::product({A, B, C});
        same_costs(P, check(P));
        const Term Q = cobble::product({A, cobble::leaf(random_pauli(n)), C});
        same_costs(Q, check(Q));
        const Term T = cobble::tensor({A, B});
        same_costs(T, check(T));
        const Term D = cobble::adjoint(P);
        same_costs(D, check(D));
        const Term S = cobble::direct_sum({A, C});                      // different alphas: one block rescaled
        same_costs(S, check(S, {}, 1e-9));
        const Term S4 = cobble::direct_sum({A, B, C, cobble::leaf(random_pauli(n))});
        same_costs(S4, check(S4));
    }
}

TEST_CASE("polynomials by QSVT on a Hermitian base, with numerically found phases") {
    const Term H = random_sum(2, 4, false);   // Hermitian: real coefficients on Hermitian strings
    REQUIRE(cobble::info(H).hermitian);
    for (const auto& coeff : std::vector<std::vector<double>>{{0, 1, 0, -0.5}, {0.3, 0, 1, 0, -0.2}, {0.1, 0.5, 0.2, 0.7}, {0, 0, 0, 0, 0, 1}}) {
        const Term P = cobble::poly(H, coeff);
        const Circuit c = check(P, {}, 1e-8);
        if (cobble::info(P).method == cobble::PolyMethod::QSVT) same_costs(P, c);
        INFO(c.notes);
    }
}

TEST_CASE("Cobble's benchmarks compile and run: spectral thresholding (degree 19), Hamiltonian simulation (degree 15)") {
    const Term H = random_sum(2, 3, false);
    REQUIRE(cobble::info(H).hermitian);
    const Term st = bench::spectral_thresholding(H, true);
    const Circuit c1 = check(st, {}, 1e-8);
    if (cobble::info(st).method == cobble::PolyMethod::QSVT) same_costs(st, c1);
    const Term hs = bench::hamiltonian_simulation(H);
    const Circuit c2 = check(hs, {}, 1e-8);
    CHECK(c2.notes.find("QSVT") != std::string::npos);   // GQET is chosen by the cost model; QSVT is built
    // laplacian-filter on a 4 x 4 grid, Cobble's input and Cobble's optimised form
    const Term Hx = bench::pauli_sum(bench::second_difference(2), 2);
    const bench::Pair LF = bench::laplacian_filter(Hx, Hx);
    check(LF.prog, {}, 1e-8);
    check(LF.targ, {}, 1e-8);
}

TEST_CASE("truncated Chebyshev: heat steps on the 4 x 4 grid, within the error budget") {
    const Term W = bench::random_walk(2);
    const double budget = 1e-6;
    const Term P = bench::heat_steps(W, 40, budget);
    REQUIRE(cobble::info(P).method == cobble::PolyMethod::Truncated);
    const Circuit c = cobble::compile(P);
    const Eigen::MatrixXcd B = cobble::block(c);
    const Eigen::MatrixXcd T = bench::dense_of(P);
    CHECK((B - T).norm() <= 4 * c.eps + 1e-8);   // Frobenius norm on 16 x 16 <= 4 x operator norm; eps includes the truncation
    CHECK(c.alpha == doctest::Approx(cobble::cost(P).alpha));
    CHECK(c.leaf_calls == cobble::cost(P).k);
}

TEST_CASE("polynomials on a non-Hermitian base: the sum of powers") {
    const Term M = random_sum(2, 3, true, false);
    REQUIRE_FALSE(cobble::info(M).hermitian);
    const Term P = cobble::poly(M, {0.5, -1.0, 0.25});
    check(P, {}, 1e-9);
}

TEST_CASE("rotations: bind, compile, expectation values and parameter-shift gradients") {
    // A 4-qubit periodic-chain ansatz with shared angles: exp(-i t0 sum Z_i Z_(i+1) / 2), then
    // exp(-i t1 sum X_i / 2), twice (the shape of a symmetry-averaged TFIM-type circuit).
    const std::size_t n = 4;
    std::vector<PauliString> zz, xs;
    for (std::size_t i = 0; i < n; ++i) {
        std::string s(n, 'I');
        s[i] = 'Z';
        s[(i + 1) % n] = 'Z';
        zz.push_back(PauliString::from_string(s));
        std::string x(n, 'I');
        x[i] = 'X';
        xs.push_back(PauliString::from_string(x));
    }
    const Term U = cobble::product({cobble::pauli_evolution(xs, 3), cobble::pauli_evolution(zz, 2),
                                    cobble::pauli_evolution(xs, 1), cobble::pauli_evolution(zz, 0)});
    CHECK(cobble::parameter_count(U) == 4);
    CHECK_THROWS(cobble::act(U, Eigen::VectorXcd::Zero(16)));   // no matrix before the angles are given
    const std::vector<double> theta{0.3, -1.1, 0.7, 2.0};
    const Circuit c = check(U, theta);

    // exp(-i t sum ZZ / 2) is diagonal: compare with the formula directly.
    const Term E = cobble::pauli_evolution(zz, 0);
    const Eigen::MatrixXcd B = cobble::block(cobble::compile(E), {0.9});
    for (Eigen::Index x = 0; x < 16; ++x) {
        double s = 0;
        for (std::size_t i = 0; i < n; ++i) s += (((x >> i) & 1) == ((x >> ((i + 1) % n)) & 1)) ? 1 : -1;
        CHECK(std::abs(B(x, x) - std::polar(1.0, -0.9 * s / 2)) < 1e-12);
    }

    // Observable: sum Z_i; state: a random normalised vector.
    std::vector<Coeff> ones(n, 1.0);
    std::vector<Term> zs;
    for (std::size_t i = 0; i < n; ++i) {
        std::string s(n, 'I');
        s[i] = 'Z';
        zs.push_back(cobble::leaf(PauliString::from_string(s)));
    }
    const Term O = cobble::sum(ones, zs);
    Eigen::VectorXcd psi(16);
    for (Eigen::Index j = 0; j < 16; ++j) psi[j] = Coeff(gauss(rng), gauss(rng));
    psi /= psi.norm();
    auto exact = [&](const std::vector<double>& th) {   // independent path: bind + act, no circuit
        const Eigen::VectorXcd phi = cobble::act(cobble::bind(U, th), psi);
        return phi.dot(cobble::act(O, phi)).real();
    };
    CHECK(cobble::expectation(c, O, psi, theta) == doctest::Approx(exact(theta)).epsilon(1e-12));
    const std::vector<double> grad = cobble::gradient(c, O, psi, theta);
    for (std::size_t j = 0; j < theta.size(); ++j) {
        std::vector<double> p = theta, m = theta;
        p[j] += 1e-5;
        m[j] -= 1e-5;
        CHECK(grad[j] == doctest::Approx((exact(p) - exact(m)) / 2e-5).epsilon(1e-6));
    }
    // the adjoint of the ansatz: inverted rotations, gradient still right
    const Term Ud = cobble::adjoint(U);
    const Circuit cd = check(Ud, theta);
    auto exact_d = [&](const std::vector<double>& th) {
        const Eigen::VectorXcd phi = cobble::act(cobble::bind(Ud, th), psi);
        return phi.dot(cobble::act(O, phi)).real();
    };
    const std::vector<double> gd = cobble::gradient(cd, O, psi, theta);
    for (std::size_t j = 0; j < theta.size(); ++j) {
        std::vector<double> p = theta, m = theta;
        p[j] += 1e-5;
        m[j] -= 1e-5;
        CHECK(gd[j] == doctest::Approx((exact_d(p) - exact_d(m)) / 2e-5).epsilon(1e-6));
    }
    // a block encoding has no expectation value on its data alone
    const Circuit lcu = cobble::compile(O);
    CHECK_THROWS(cobble::expectation(lcu, O, psi, {}));
}

TEST_CASE("rotations inside block encodings: a parameterised linear combination") {
    const Term R0 = cobble::rotation(PauliString::from_string("XY"), 0), R1 = cobble::rotation(PauliString::from_string("ZI"), 1);
    const Term S = cobble::sum({0.6, Coeff(0, -0.4)}, {R0, cobble::product({R1, R0})});
    check(S, {0.4, -2.2});
}
