// Reference checks for Basics.h (written by Claude, 4 Nov 2026).
// Each check compares the simulator with something built independently of it: the full 2^n x 2^n
// matrix of an operation written entry by entry, the DFT formula, an explicit partial trace, or
// exact outcome statistics. They are NOT the known-answer tests: those come from Charbel's own
// hand calculations and go in their own file (tests/kat_*.cpp, added to CMakeLists.txt).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Basics.h"

#include <algorithm>

namespace {

std::mt19937_64 gen(2026);

MatrixXcd random_unitary(size_t d) {
    std::normal_distribution<double> n;
    MatrixXcd A(d, d);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) A(i, j) = Complex(n(gen), n(gen));
    return HouseholderQR<MatrixXcd>(A).householderQ();
}

VectorXcd random_state(size_t D) {
    std::normal_distribution<double> n;
    VectorXcd v(D);
    for (size_t i = 0; i < D; ++i) v(i) = Complex(n(gen), n(gen));
    return v.normalized();
}

// Full matrix of U acting on `targets` (bit j of U's index = qubit targets[j]), applied only where
// every qubit in `controls` is 1. Built entry by entry, independently of Qubits::apply.
MatrixXcd full_matrix(size_t n, const MatrixXcd& U, const std::vector<size_t>& targets,
                      const std::vector<size_t>& controls) {
    const size_t D = size_t(1) << n;
    size_t tmask = 0;
    for (size_t q : targets) tmask |= size_t(1) << q;
    auto local = [&](size_t i) {
        size_t l = 0;
        for (size_t j = 0; j < targets.size(); ++j)
            if ((i >> targets[j]) & 1) l |= size_t(1) << j;
        return l;
    };
    MatrixXcd F = MatrixXcd::Zero(D, D);
    for (size_t i = 0; i < D; ++i)
        for (size_t j = 0; j < D; ++j) {
            if ((i & ~tmask) != (j & ~tmask)) continue;
            bool on = true;
            for (size_t c : controls) on = on && ((i >> c) & 1);
            F(i, j) = on ? U(local(i), local(j)) : Complex(i == j ? 1.0 : 0.0, 0.0);
        }
    return F;
}

std::vector<size_t> random_subset(size_t n, size_t k, const std::vector<size_t>& exclude = {}) {
    std::vector<size_t> all;
    for (size_t q = 0; q < n; ++q)
        if (std::find(exclude.begin(), exclude.end(), q) == exclude.end()) all.push_back(q);
    std::shuffle(all.begin(), all.end(), gen);
    all.resize(k);
    return all;
}

VectorXcd basis(size_t D, size_t k) {
    VectorXcd v = VectorXcd::Zero(D);
    v(k) = 1;
    return v;
}

constexpr double TOL = 1e-12;

}  // namespace

TEST_CASE("applyOn equals the full matrix, any target list") {
    const size_t n = 5, D = size_t(1) << n;
    for (int t = 0; t < 200; ++t) {
        const size_t k = 1 + gen() % 4;
        auto targets = random_subset(n, k);
        auto U = random_unitary(size_t(1) << k);
        auto v = random_state(D);
        Qubits q(v);
        q.applyOn(U, targets);
        CHECK((q.state_vector() - full_matrix(n, U, targets, {}) * v).norm() < TOL);
    }
}

TEST_CASE("applyControlledOn equals the full matrix, one to three controls") {
    const size_t n = 5, D = size_t(1) << n;
    for (int t = 0; t < 200; ++t) {
        const size_t k = 1 + gen() % 2, c = 1 + gen() % 3;
        auto targets = random_subset(n, k);
        auto controls = random_subset(n, c, targets);
        auto U = random_unitary(size_t(1) << k);
        auto v = random_state(D);
        Qubits q(v);
        q.applyControlledOn(U, controls, targets);
        CHECK((q.state_vector() - full_matrix(n, U, targets, controls) * v).norm() < TOL);
    }
}

TEST_CASE("applyGate and applyCGate (ranges and single qubits) equal the full matrix") {
    const size_t n = 5, D = size_t(1) << n;
    for (size_t s = 0; s < n; ++s)
        for (size_t e = s; e < n; ++e) {
            auto U = random_unitary(size_t(1) << (e - s + 1));
            std::vector<size_t> range;
            for (size_t x = s; x <= e; ++x) range.push_back(x);
            auto v = random_state(D);
            Qubits q(v);
            q.applyGate(U, s, e);
            CHECK((q.state_vector() - full_matrix(n, U, range, {}) * v).norm() < TOL);
            for (size_t c = 0; c < n; ++c) {
                if (c >= s && c <= e) continue;
                auto w = random_state(D);
                Qubits r(w);
                r.applyCGate(U, c, s, e);
                CHECK((r.state_vector() - full_matrix(n, U, range, {c}) * w).norm() < TOL);
            }
        }
    Matrix2cd U = random_unitary(2);
    for (size_t c = 0; c < n; ++c)
        for (size_t t = 0; t < n; ++t) {
            if (c == t) continue;
            auto v = random_state(D);
            Qubits q(v);
            q.applyCGate(U, c, t);
            CHECK((q.state_vector() - full_matrix(n, U, {t}, {c}) * v).norm() < TOL);
        }
}

TEST_CASE("swapQubits exchanges the bits of every basis state") {
    const size_t n = 4, D = size_t(1) << n;
    for (size_t a = 0; a < n; ++a)
        for (size_t b = 0; b < n; ++b)
            for (size_t x = 0; x < D; ++x) {
                Qubits q(basis(D, x));
                q.swapQubits(a, b);
                size_t y = x;
                if (((x >> a) & 1) != ((x >> b) & 1)) y = x ^ (size_t(1) << a) ^ (size_t(1) << b);
                CHECK(std::abs(q.state_vector()(y) - 1.0) < TOL);
            }
}

TEST_CASE("QFT equals the DFT on the register index; IQFT inverts it") {
    for (size_t n = 1; n <= 5; ++n) {
        const size_t D = size_t(1) << n;
        for (size_t j = 0; j < D; ++j) {
            Qubits q(basis(D, j));
            q.applyQFT();
            VectorXcd ref(D);
            for (size_t k = 0; k < D; ++k)
                ref(k) = std::exp(Complex(0, 2 * PI * double(j) * double(k) / double(D))) / std::sqrt(double(D));
            CHECK((q.state_vector() - ref).norm() < 1e-10);
        }
        for (int t = 0; t < 10; ++t) {
            auto v = random_state(D);
            Qubits q(v);
            q.applyQFT();
            q.applyIQFT();
            CHECK((q.state_vector() - v).norm() < 1e-10);
        }
    }
}

TEST_CASE("operator[] is the partial trace") {
    const size_t n = 4, D = size_t(1) << n;
    for (int t = 0; t < 20; ++t) {
        auto v = random_state(D);
        Qubits q(v);
        for (size_t i = 0; i < n; ++i) {
            Matrix2cd ref = Matrix2cd::Zero();
            for (size_t a = 0; a < 2; ++a)
                for (size_t b = 0; b < 2; ++b)
                    for (size_t r = 0; r < D; ++r) {
                        if ((r >> i) & 1) continue;
                        ref(a, b) += v(r | (a << i)) * std::conj(v(r | (b << i)));
                    }
            CHECK((q[i].density_matrix() - ref).norm() < TOL);
        }
    }
}

TEST_CASE("BasisChange: coordinates inside, restoration after, rejection of non-unitaries") {
    const size_t n = 5, D = size_t(1) << n;
    auto v = random_state(D);
    Qubits q(v);
    auto U = random_unitary(4);
    {
        BasisChange guard(q, U, {3, 1});
        CHECK((q.state_vector() - full_matrix(n, U.adjoint(), {3, 1}, {}) * v).norm() < TOL);
    }
    CHECK((q.state_vector() - v).norm() < TOL);

    {
        BasisChange a(q, random_unitary(2), {0});
        BasisChange b(q, random_unitary(8), {4, 0, 2});
    }
    CHECK((q.state_vector() - v).norm() < TOL);

    try {
        BasisChange a(q, random_unitary(2), {2});
        throw std::runtime_error("leave the block");
    } catch (const std::runtime_error&) {
    }
    CHECK((q.state_vector() - v).norm() < TOL);

    MatrixXcd bad = MatrixXcd::Identity(2, 2);
    bad(0, 1) = 0.5;
    CHECK_THROWS_AS(BasisChange(q, bad, {1}), QuantumError);
    CHECK((q.state_vector() - v).norm() < TOL);

    CHECK_FALSE(std::is_copy_constructible_v<BasisChange>);
}

TEST_CASE("measurement: GHZ outcomes agree; Bell-basis measurement is deterministic") {
    std::mt19937_64 rng(5);
    int ones = 0, disagree = 0;
    const int T = 4000;
    for (int t = 0; t < T; ++t) {
        Qubits q(3);
        q.applyGate(Q_Gates::H, 0);
        q.applyCGate(Q_Gates::X, 0, 1);
        q.applyCGate(Q_Gates::X, 1, 2);
        const size_t a = q.measureQubit(1, rng), b = q.measureQubit(0, rng), c = q.measureQubit(2, rng);
        ones += int(a);
        disagree += !(a == b && b == c);
    }
    CHECK(disagree == 0);
    CHECK(std::abs(ones / double(T) - 0.5) < 0.05);   // 0.05 is more than 6 standard deviations

    const double s = 1 / std::sqrt(2.0);
    VectorXcd bell[4];
    for (auto& b : bell) b = VectorXcd::Zero(4);
    bell[0](0) = s; bell[0](3) = s;
    bell[1](0) = s; bell[1](3) = -s;
    bell[2](1) = s; bell[2](2) = s;
    bell[3](1) = s; bell[3](2) = -s;
    std::vector<MatrixXcd> P;
    for (auto& b : bell) P.push_back(b * b.adjoint());
    for (int t = 0; t < 200; ++t) {
        Qubits q(3);
        q.applyGate(Q_Gates::H, 0);
        q.applyCGate(Q_Gates::X, 0, 2);
        q.applyGate(Q_Gates::X, 1);
        CHECK(q.measureOn(P, {0, 2}, rng) == 0);
    }
}

TEST_CASE("Qubit::measure with a general measurement: trine POVM statistics") {
    std::mt19937_64 rng(11);
    Matrix2cd M[3];
    for (int k = 0; k < 3; ++k) {
        const double th = 2 * PI * k / 3;
        Vector2cd phi(std::cos(th / 2), std::sin(th / 2));
        M[k] = std::sqrt(2.0 / 3) * phi * phi.adjoint();
    }
    int count[3] = {0, 0, 0};
    const int T = 30000;
    for (int t = 0; t < T; ++t) {
        Qubit q(Vector2cd(1, 0));
        ++count[q.measure({M[0], M[1], M[2]}, rng)];
    }
    CHECK(std::abs(count[0] / double(T) - 2.0 / 3) < 0.02);
    CHECK(std::abs(count[1] / double(T) - 1.0 / 6) < 0.02);
    CHECK(std::abs(count[2] / double(T) - 1.0 / 6) < 0.02);
}

TEST_CASE("operator^ appends the new qubit as the highest one") {
    Qubits a(2);
    a.applyGate(Q_Gates::X, 0);
    auto b = a ^ q_utils::create_one_state();
    CHECK(b.num_qubits() == 3);
    CHECK(std::abs(b.state_vector()(5) - 1.0) < TOL);
}
