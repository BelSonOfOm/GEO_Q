// Checks for polymax.h (written by Claude, 5 Nov 2026): known maxima, the Chebyshev table, basis
// round trips, and random polynomials against dense sampling.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "polymax.h"
#include <complex>
#include <numbers>
#include <random>
using namespace polymax;

// Independent references: dense sampling (a lower bound that is tight up to the grid spacing).
static double sample_interval(const Poly& q, int N = 400001) {
    double best = 0;
    for (int i = 0; i < N; ++i) { double x = -1.0 + 2.0 * i / (N - 1); best = std::max(best, std::abs(eval(q, x))); }
    return best;
}
static double sample_circle(const Poly& c, int N = 400001) {
    double best = 0;
    for (int i = 0; i < N; ++i) {
        std::complex<double> z = std::polar(1.0, 2 * std::numbers::pi * i / N), w = 1, s = 0;
        for (double cn : c) { s += cn * w; w *= z; }
        best = std::max(best, std::abs(s));
    }
    return best;
}
// T_n(x) evaluated as cos(n arccos x), independent of the table.
static double cheb_direct(std::size_t n, double x) { return std::cos(double(n) * std::acos(x)); }

TEST_CASE("known maxima on [-1, 1]") {
    CHECK(max_abs_interval({0, 0, 4, 0, -4}) == doctest::Approx(1.0));   // 4x^2 - 4x^4, max 1 at x^2 = 1/2
    CHECK(max_abs_interval({3}) == doctest::Approx(3.0));
    CHECK(max_abs_interval({0, -2}) == doctest::Approx(2.0));
    CHECK(max_abs_interval({0.25, 0, -1}) == doctest::Approx(0.75));      // 1/4 - x^2: |.| largest at x = +-1
}

TEST_CASE("Chebyshev table matches cos(n arccos x), and T_n has max 1") {
    auto T = chebyshev_table(20);
    for (std::size_t n = 0; n <= 20; ++n) {
        for (double x : {-0.93, -0.4, 0.0, 0.31, 0.77})
            CHECK(eval(T[n], x) == doctest::Approx(cheb_direct(n, x)).epsilon(1e-9));
        CHECK(max_abs_interval(T[n]) == doctest::Approx(1.0).epsilon(1e-9));
    }
}

TEST_CASE("basis change round trips") {
    Poly c = to_chebyshev({0, 0, 1});                                      // x^2 = (T_0 + T_2)/2
    REQUIRE(c.size() == 3);
    CHECK(c[0] == doctest::Approx(0.5)); CHECK(c[1] == doctest::Approx(0.0)); CHECK(c[2] == doctest::Approx(0.5));
    std::mt19937 rng(7); std::uniform_real_distribution<double> U(-1, 1);
    for (int t = 0; t < 50; ++t) {
        Poly q(1 + t % 20); for (double& v : q) v = U(rng);
        Poly back = from_chebyshev(to_chebyshev(q));
        for (std::size_t k = 0; k < q.size(); ++k) CHECK(back[k] == doctest::Approx(q[k]).epsilon(1e-9));
    }
}

TEST_CASE("known maxima on the unit circle") {
    CHECK(max_abs_circle({1, 1}) == doctest::Approx(2.0));                // |1 + z|, max at z = 1
    CHECK(max_abs_circle({0, 0, 0, 1}) == doctest::Approx(1.0));          // |z^3| = 1
    CHECK(max_abs_circle({0.5, 0, 0.5}) == doctest::Approx(1.0));
    CHECK(max_abs_circle({1, -1}) == doctest::Approx(2.0));               // |1 - z|, max at z = -1
}

TEST_CASE("random polynomials against dense sampling") {
    std::mt19937 rng(11); std::uniform_real_distribution<double> U(-1, 1);
    for (int t = 0; t < 60; ++t) {
        Poly q(2 + t % 20); for (double& v : q) v = U(rng);
        double a = max_abs_interval(q), s = sample_interval(q);
        CHECK(a >= s * (1 - 1e-12));                      // never below a value actually reached
        CHECK(a <= s * (1 + 1e-6));                       // and not above the true maximum
        double b = max_abs_circle(q), sc = sample_circle(q);
        CHECK(b >= sc * (1 - 1e-9));
        CHECK(b <= sc * (1 + 1e-6));
    }
}

TEST_CASE("Cobble's spectral-thresholding polynomial (degree 19)") {
    Poly c(20, 0.0);
    for (int i = 0; i < 10; ++i) c[2 * i + 1] = (4 / std::numbers::pi) / (2 * i + 1);   // (4/pi) sum T_{2i+1}/(2i+1)
    Poly q = from_chebyshev(c);
    double a = max_abs_interval(q), s = sample_interval(q);
    CHECK(a == doctest::Approx(s).epsilon(1e-6));
    double b = max_abs_circle(c), sc = sample_circle(c);
    CHECK(b == doctest::Approx(sc).epsilon(1e-6));
    MESSAGE("spectral thresholding: interval max = " << a << ", circle max = " << b);
}
