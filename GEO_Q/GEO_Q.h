// P.H. <3
#pragma once
#include "pauli.h"
#include <complex>
#include <memory>
#include <variant>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <array>
#include <optional>
#include <tuple>
#include <numbers>
#include <random>
#include "polymax.h"

namespace cobble {

using Coeff = std::complex<double>;

// A Cobble term is a tree. Term is a shared, read-only handle to one node of it.
struct Node;
using Term = std::shared_ptr<const Node>;

// The cost of a term (Cobble, Section 5.1, Tables 1 and 2): k queries (uses of a leaf),
// subnormalisation alpha (the circuit encodes the matrix divided by alpha), m workspace qubits.
// rescaled counts the direct-sum blocks in the tree whose alpha had to be raised to match the
// others (decision of 5 Nov: a direct sum is never rejected for alpha, the rescaling is recorded).
struct Cost {
    std::uint64_t k = 0;
    double alpha = 0.0;
    std::size_t m = 0;
    std::size_t rescaled = 0;
    double eps = 0.0;          // bound on ||implemented - intended|| in operator norm (decision of 5 Nov;
                               // 0 = exact, as everywhere in Cobble, Sec. 8)
};

// The five ways to compile a Poly node: the four of Cobble's Table 2, and Ko (arXiv:2608.15161).
// Truncated: QSVT on the Chebyshev expansion of the polynomial cut at the lowest degree whose tail fits
// the node's error budget (added 5 Nov; Apers & Sarlette 2019, Sachdeva & Vishnoi 2014).
enum class PolyMethod { LCU = 0, Horner = 1, QSVT = 2, GQET = 3, Ko = 4, Truncated = 5 };

// Everything worked out about a node, computed on first request and then kept in the node.
struct Info {
    Cost cost;                                   // the cost used by the parents
    bool hermitian = false;                      // Cobble's Hermiticity judgment (Fig. 9), see info()
    PolyMethod method = PolyMethod::LCU;         // Poly nodes only: the method chosen
    std::array<std::optional<Cost>, 6> rows{};   // Poly nodes only: the cost of every valid method
    std::vector<double> realised_cheb;           // Truncated only: Chebyshev coefficients of the polynomial
                                                 // actually built, in the variable x = M / alpha_M
};

// The seven kinds of node (Cobble, section 4.1; leaves are Pauli strings, decision of 4 Nov).
struct Leaf      { PauliString p; };                               // a Pauli string
struct Adjoint   { Term a; };                                      // a^dagger
struct Sum       { std::vector<Coeff> c; std::vector<Term> t; };   // c[0] t[0] + c[1] t[1] + ...
struct Product   { std::vector<Term> t; };                         // t[0] t[1] ... (matrix product)
struct DirectSum { std::vector<Term> t; };                         // block-diagonal, 2, 4, 8, ... blocks
struct Tensor    { std::vector<Term> t; };                         // t[0] on the lowest qubits, then t[1], ...
struct Poly      { Term a; std::vector<double> coeff; double budget = 0.0; };   // coeff[0] + ... + coeff[d] a^d,
                                                                               // within error budget (0 = exact)
struct Unitary   { std::string name; Eigen::MatrixXcd U; };         // any unitary, encoding itself (added 5 Nov)
struct Rotation  { PauliString p; std::size_t param; };            // exp(-i theta[param] P / 2), P a Hermitian
                                                                   // Pauli string; theta is given later (added 6 Nov)

using Kind = std::variant<Leaf, Adjoint, Sum, Product, DirectSum, Tensor, Poly, Unitary, Rotation>;

struct Node {
    Kind kind;
    std::size_t qubits;    // number of data qubits
    std::uint64_t id;      // same id    <=> identical tree (as written)
    std::uint64_t canon;   // same canon <=> equal up to flattening nested sums, products and tensor
                           //                products, and reordering the terms of sums
    mutable std::optional<Info> info = std::nullopt;   // empty until info(term) is first called
};

// ---- The table: every distinct node is stored once (hash-consing) ------------------------------
namespace detail {

struct Table {
    std::unordered_map<std::string, std::weak_ptr<const Node>> nodes;   // exact key -> node
    std::unordered_map<std::string, std::uint64_t> canon;               // canonical key -> label
    std::uint64_t next_id = 1;
    std::uint64_t next_canon = 1;
};

inline Table& table() {
    static Table t;
    return t;
}

// Keys are byte strings. Numbers are appended by their exact bits (exact equality, decision of 4 Nov);
// -0.0 is turned into 0.0 first, so that the two zeros count as equal.
inline void put(std::string& key, std::uint64_t v) {
    key.append(reinterpret_cast<const char*>(&v), sizeof v);
}
inline void put(std::string& key, double v) {
    if (!std::isfinite(v)) throw std::invalid_argument("cobble: coefficients must be finite numbers.");
    if (v == 0.0) v = 0.0;
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    put(key, bits);
}
inline void put(std::string& key, const Coeff& c) {
    put(key, c.real());
    put(key, c.imag());
}

// Return the stored node with this exact key, or store a new one.
inline Term make(Kind kind, std::size_t qubits, const std::string& key, const std::string& canon_key) {
    Table& T = table();
    if (auto it = T.nodes.find(key); it != T.nodes.end())
        if (Term existing = it->second.lock()) return existing;
    std::uint64_t canon;
    if (auto it = T.canon.find(canon_key); it != T.canon.end()) {
        canon = it->second;
    } else {
        canon = T.next_canon++;
        T.canon.emplace(canon_key, canon);
    }
    Term node = std::make_shared<const Node>(Node{std::move(kind), qubits, T.next_id++, canon});
    T.nodes[key] = node;
    return node;
}

inline void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

inline bool is_power_of_two(std::size_t k) { return k >= 2 && (k & (k - 1)) == 0; }

// The smallest e with 2^e >= n: the number of qubits that can hold an index 0, ..., n-1.
inline std::size_t ceil_log2(std::size_t n) {
    std::size_t e = 0;
    while ((std::size_t(1) << e) < n) ++e;
    return e;
}

// The degree of a coefficient list as written: the index of its last nonzero coefficient, or -1.
inline long true_degree(const std::vector<double>& a) {
    for (std::size_t j = a.size(); j-- > 0;)
        if (a[j] != 0.0) return long(j);
    return -1;
}

// Flattened view of a sum: the (coefficient, canonical label) pairs of all its non-sum terms,
// with the coefficients of nested sums multiplied through.
inline void flatten_sum(const Term& t, const Coeff& factor, std::vector<std::pair<Coeff, std::uint64_t>>& out) {
    if (const Sum* s = std::get_if<Sum>(&t->kind)) {
        for (std::size_t j = 0; j < s->t.size(); ++j) flatten_sum(s->t[j], factor * s->c[j], out);
    } else {
        out.emplace_back(factor, t->canon);
    }
}

// Flattened view of a product or tensor product: the canonical labels of its factors, in order,
// with nested products of the same kind opened up.
template <class K>
inline void flatten_chain(const Term& t, std::vector<std::uint64_t>& out) {
    if (const K* p = std::get_if<K>(&t->kind)) {
        for (const Term& f : p->t) flatten_chain<K>(f, out);
    } else {
        out.push_back(t->canon);
    }
}

}  // namespace detail

// ---- Building terms ---------------------------------------------------------------------------

inline Term leaf(const PauliString& p) {
    detail::require(p.n >= 1, "cobble: a leaf needs at least one qubit.");
    std::string key = "L";
    detail::put(key, std::uint64_t(p.n));
    detail::put(key, p.x);
    detail::put(key, p.z);
    detail::put(key, std::uint64_t(p.k % 4));
    return detail::make(Leaf{p}, p.n, key, key);
}

inline Term adjoint(const Term& a) {
    std::string key = "A", ckey = "A";
    detail::put(key, a->id);
    detail::put(ckey, a->canon);
    return detail::make(Adjoint{a}, a->qubits, key, ckey);
}

// c[0] t[0] + c[1] t[1] + ...   (Cobble requires the coefficients not to be all zero)
inline Term sum(const std::vector<Coeff>& c, const std::vector<Term>& t) {
    detail::require(!t.empty() && c.size() == t.size(), "cobble: a sum needs one coefficient per term.");
    bool nonzero = false;
    std::string key = "S";
    for (std::size_t j = 0; j < t.size(); ++j) {
        detail::require(t[j]->qubits == t[0]->qubits, "cobble: the terms of a sum act on different numbers of qubits.");
        detail::put(key, c[j]);
        detail::put(key, t[j]->id);
        nonzero = nonzero || c[j] != Coeff(0.0);
    }
    detail::require(nonzero, "cobble: the coefficients of a sum are all zero.");

    std::vector<std::pair<Coeff, std::uint64_t>> flat;
    for (std::size_t j = 0; j < t.size(); ++j) detail::flatten_sum(t[j], c[j], flat);
    std::sort(flat.begin(), flat.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;
        if (a.first.real() != b.first.real()) return a.first.real() < b.first.real();
        return a.first.imag() < b.first.imag();
    });
    std::string ckey = "S";
    for (const auto& [coef, label] : flat) {
        detail::put(ckey, coef);
        detail::put(ckey, label);
    }
    return detail::make(Sum{c, t}, t[0]->qubits, key, ckey);
}

// t[0] t[1] ... t[last]: the matrix product, so t[last] acts first on a state.
inline Term product(const std::vector<Term>& t) {
    detail::require(t.size() >= 2, "cobble: a product needs at least two factors.");
    std::string key = "P", ckey = "P";
    for (const Term& f : t) {
        detail::require(f->qubits == t[0]->qubits, "cobble: the factors of a product act on different numbers of qubits.");
        detail::put(key, f->id);
    }
    std::vector<std::uint64_t> flat;
    for (const Term& f : t) detail::flatten_chain<Product>(f, flat);
    for (std::uint64_t label : flat) detail::put(ckey, label);
    return detail::make(Product{t}, t[0]->qubits, key, ckey);
}

// Block-diagonal with blocks t[0], t[1], ...: 2, 4, 8, ... blocks of equal size (decision of 4 Nov).
// Block j sits where the extra log2(k) qubits, above the data qubits, hold the number j.
inline Term direct_sum(const std::vector<Term>& t) {
    detail::require(detail::is_power_of_two(t.size()), "cobble: a direct sum needs 2, 4, 8, ... blocks.");
    std::string key = "D", ckey = "D";
    for (const Term& b : t) {
        detail::require(b->qubits == t[0]->qubits, "cobble: the blocks of a direct sum act on different numbers of qubits.");
        detail::put(key, b->id);
        detail::put(ckey, b->canon);
    }
    return detail::make(DirectSum{t}, t[0]->qubits + detail::ceil_log2(t.size()), key, ckey);
}

// t[0] (x) t[1] (x) ...: t[0] acts on the lowest-numbered qubits, t[1] on the next ones, and so on.
inline Term tensor(const std::vector<Term>& t) {
    detail::require(t.size() >= 2, "cobble: a tensor product needs at least two factors.");
    std::string key = "T", ckey = "T";
    std::size_t qubits = 0;
    for (const Term& f : t) {
        detail::put(key, f->id);
        qubits += f->qubits;
    }
    std::vector<std::uint64_t> flat;
    for (const Term& f : t) detail::flatten_chain<Tensor>(f, flat);
    for (std::uint64_t label : flat) detail::put(ckey, label);
    return detail::make(Tensor{t}, qubits, key, ckey);
}

// A unitary leaf: any unitary U on n qubits encodes itself with alpha = 1 and no workspace (Cobble,
// Example 2.3), like Cobble's Basic black boxes. Added 5 Nov for shift operators. The matrix is kept
// for the checks; two leaves with the same matrix are the same node, whatever their names.
inline Term unitary_leaf(const std::string& name, const Eigen::MatrixXcd& U) {
    detail::require(U.rows() == U.cols() && U.rows() >= 2 && (U.rows() & (U.rows() - 1)) == 0,
                    "cobble: a unitary leaf needs a 2^n x 2^n matrix.");
    const Eigen::MatrixXcd E = U.adjoint() * U - Eigen::MatrixXcd::Identity(U.rows(), U.cols());
    detail::require(E.norm() <= 1e-10 * double(U.rows()), "cobble: the matrix of a unitary leaf is not unitary.");
    std::size_t n = 0;
    while ((Eigen::Index(1) << n) < U.rows()) ++n;
    std::string key = "U";
    detail::put(key, std::uint64_t(n));
    for (Eigen::Index j = 0; j < U.cols(); ++j)
        for (Eigen::Index i = 0; i < U.rows(); ++i) detail::put(key, Coeff(U(i, j)));
    return detail::make(Unitary{name, U}, n, key, key);
}

// exp(-i theta_param P / 2): a rotation whose angle is the parameter number `param`, given only when the
// term is run (compile.h) or bound (bind below). P must be a Hermitian Pauli string, so P^2 = I and
// exp(-i theta P / 2) = cos(theta / 2) I - i sin(theta / 2) P (Nielsen & Chuang, Exercise 4.2). Parameters may
// be shared between rotations (e.g. one angle for a whole symmetry-averaged generator). Added 6 Nov.
inline Term rotation(const PauliString& p, std::size_t param) {
    detail::require(p.n >= 1, "cobble: a rotation needs at least one qubit.");
    detail::require(is_hermitian(p), "cobble: a rotation needs a Hermitian Pauli string.");
    std::string key = "R";
    detail::put(key, std::uint64_t(p.n));
    detail::put(key, p.x);
    detail::put(key, p.z);
    detail::put(key, std::uint64_t(p.k % 4));
    detail::put(key, std::uint64_t(param));
    return detail::make(Rotation{p, param}, p.n, key, key);
}

// coeff[0] + coeff[1] a + ... + coeff[d] a^d, real coefficients as in Cobble (section 4.3).
inline Term poly(const Term& a, const std::vector<double>& coeff, double budget = 0.0) {
    detail::require(std::isfinite(budget) && budget >= 0.0, "cobble: the error budget must be a finite number >= 0.");
    // The list is kept as written; its degree is the last nonzero coefficient (decision of 4-5 Nov).
    detail::require(detail::true_degree(coeff) >= 1, "cobble: a polynomial needs degree at least 1.");
    std::string key = "Y", ckey = "Y";
    detail::put(key, a->id);
    detail::put(ckey, a->canon);
    for (double v : coeff) {
        detail::put(key, v);
        detail::put(ckey, v);
    }
    if (budget > 0.0) {
        key += "B";
        ckey += "B";
        detail::put(key, budget);
        detail::put(ckey, budget);
    }
    return detail::make(Poly{a, coeff, budget}, a->qubits, key, ckey);
}

// ---- Costs and Hermiticity, computed on demand and kept in the node ---------------------------

const Info& info(const Term& t);

namespace detail {

// Two alphas of a direct sum count as equal within this relative tolerance (decision of 5 Nov).
constexpr double alpha_tolerance = 1e-12;

// Which blocks of a direct sum are rescaled: those whose alpha is below the largest one by more
// than the tolerance. Rescaling block j puts a one-qubit rotation with top-left entry
// alpha_j / alpha_max next to it (arXiv:2403.19512, App. B), so that every block shares alpha_max.
// The cost now and the compiler later both call this function, so they cannot disagree.
inline std::vector<bool> rescaled_blocks(const DirectSum& d) {
    double top = 0.0;
    for (const Term& b : d.t) top = std::max(top, info(b).cost.alpha);
    std::vector<bool> r;
    for (const Term& b : d.t) r.push_back(top - info(b).cost.alpha > alpha_tolerance * top);
    return r;
}

// Hermiticity of a product, by (M_1 ... M_n)^dagger = M_n^dagger ... M_1^dagger: the product is
// Hermitian when factor n+1-j is the adjoint of factor j, and a middle factor is Hermitian.
// A sufficient condition only; Cobble instead asks the user to declare commuting factors (Fig. 9).
inline bool palindromic(const Product& p) {
    const std::size_t n = p.t.size();
    for (std::size_t j = 0; j < (n + 1) / 2; ++j) {
        const Term& a = p.t[j];
        const Term& b = p.t[n - 1 - j];
        if (j == n - 1 - j) {
            if (!info(a).hermitian) return false;
        } else if (!(b->canon == adjoint(a)->canon || (info(a).hermitian && b->canon == a->canon))) {
            return false;
        }
    }
    return true;
}

// Hermiticity of a sum, by (sum_j c_j T_j)^dagger = sum_j conj(c_j) T_j^dagger: the sum is Hermitian
// when its terms pair up with their own adjoints. mirror(T) names the node that means T^dagger.
inline std::uint64_t mirror_canon(const Term& t) {
    if (const Unitary* u = std::get_if<Unitary>(&t->kind)) return unitary_leaf(u->name, u->U.adjoint())->canon;
    if (const Adjoint* a = std::get_if<Adjoint>(&t->kind)) return a->a->canon;
    if (info(t).hermitian) return t->canon;
    return adjoint(t)->canon;
}

inline bool adjoint_symmetric(const Sum& s) {
    std::vector<std::tuple<double, double, std::uint64_t>> own, mir;
    for (std::size_t j = 0; j < s.t.size(); ++j) {
        own.emplace_back(s.c[j].real(), s.c[j].imag(), s.t[j]->canon);
        mir.emplace_back(s.c[j].real(), -s.c[j].imag() + 0.0, mirror_canon(s.t[j]));
    }
    std::sort(own.begin(), own.end());
    std::sort(mir.begin(), mir.end());
    return own == mir;
}

// Up to this degree, maxima are found exactly from the roots of the derivative (polymax); above it,
// the monomial basis is too ill-conditioned, so Chebyshev coefficients are sampled and the maximum is
// bounded by sum |c_k| (|T_k| <= 1 on [-1, 1]), an upper bound and therefore still a valid alpha.
constexpr long exact_degree_limit = 30;

// Drop Chebyshev coefficients at rounding level, so that parity and degree are read correctly.
inline std::vector<double> clean(std::vector<double> c) {
    double top = 0.0;
    for (double v : c) top = std::max(top, std::abs(v));
    for (double& v : c)
        if (std::abs(v) <= 1e-14 * top) v = 0.0;
    return c;
}

// max over [-1, 1] of |sum_k c_k T_k| (exact up to the degree limit, an upper bound above it).
inline double cheb_max(const std::vector<double>& c) {
    const long d = true_degree(c);
    if (d < 0) return 0.0;
    if (d <= exact_degree_limit) return polymax::max_abs_interval(polymax::from_chebyshev(std::vector<double>(c.begin(), c.begin() + d + 1)));
    double s = 0.0;
    for (double v : c) s += std::abs(v);
    return s;
}

// QSVT on a polynomial given by its Chebyshev coefficients: each parity part costs its degree in
// queries; two parts are joined by one more sum (one more ancilla).
inline Cost qsvt_on_cheb(const std::vector<double>& c, const Cost& M, double eps) {
    std::vector<double> ce(c.size(), 0.0), co(c.size(), 0.0);
    for (std::size_t j = 0; j < c.size(); ++j) (j % 2 ? co : ce)[j] = c[j];
    const long de = true_degree(ce), dq = true_degree(co);
    if (de >= 0 && dq >= 0)
        return Cost{M.k * std::uint64_t(de + dq), cheb_max(ce) + cheb_max(co), 2 + M.m, M.rescaled, eps};
    return Cost{M.k * std::uint64_t(std::max(de, dq)), cheb_max(c), 1 + M.m, M.rescaled, eps};
}

// The cost of every way to compile Poly(M, a) (Cobble Table 2 and Sec. 5.2; Ko, Corollary 1; and the
// truncated Chebyshev method). q(x) = p(alpha_M x) is the polynomial the methods actually see.
inline void poly_rows(const Poly& P, Info& out) {
    const Info& M = info(P.a);
    const long d = true_degree(P.coeff);
    std::vector<double> q(std::size_t(d) + 1);
    for (long j = 0; j <= d; ++j) q[std::size_t(j)] = P.coeff[std::size_t(j)] * std::pow(M.cost.alpha, double(j));
    double l1 = 0.0;
    for (double v : q) l1 += std::abs(v);
    const std::uint64_t D = std::uint64_t(d);

    // Error carried in from an approximate M: ||M~^j - M^j|| <= (alpha + eps)^j - alpha^j, term by term.
    double carried = 0.0;
    if (M.cost.eps > 0.0)
        for (long j = 1; j <= d; ++j)
            carried += std::abs(P.coeff[std::size_t(j)]) *
                       (std::pow(M.cost.alpha + M.cost.eps, double(j)) - std::pow(M.cost.alpha, double(j)));

    // LCU: j queries for each nonzero monomial a_j M^j, one sum at the end.
    std::uint64_t lcu_queries = 0;
    for (long j = 1; j <= d; ++j)
        if (P.coeff[std::size_t(j)] != 0.0) lcu_queries += std::uint64_t(j);
    out.rows[0] = Cost{M.cost.k * lcu_queries, l1, ceil_log2(std::size_t(d)) + std::size_t(d) + M.cost.m, M.cost.rescaled, carried};
    // Horner: one query per iteration, two ancillas per iteration.
    out.rows[1] = Cost{M.cost.k * D, l1, 2 * std::size_t(d) + M.cost.m, M.cost.rescaled, carried};

    if (!M.hermitian) return;   // QSVT, GQET, Ko and Truncated need a Hermitian M

    const bool high = d > exact_degree_limit;
    const std::vector<double> c = high ? polymax::chebyshev_by_sampling(q) : polymax::to_chebyshev(q);

    // QSVT on the actual even and odd parts (decision of 4 Nov). Parity and degrees are read from the
    // monomial coefficients, whose zeros are exact.
    std::vector<double> qe(q.size(), 0.0), qo(q.size(), 0.0);
    for (std::size_t j = 0; j < q.size(); ++j) (j % 2 ? qo : qe)[j] = q[j];
    const long de = true_degree(qe), dq = true_degree(qo);
    auto part_bound = [&](int parity) {                   // sum |c_k| over k of one parity (high degree)
        double s = 0.0;
        for (std::size_t k = std::size_t(parity); k < c.size(); k += 2) s += std::abs(c[k]);
        return s;
    };
    if (de >= 0 && dq >= 0) {
        const double a = high ? part_bound(0) + part_bound(1) : polymax::max_abs_interval(qe) + polymax::max_abs_interval(qo);
        out.rows[2] = Cost{M.cost.k * std::uint64_t(de + dq), a, 2 + M.cost.m, M.cost.rescaled, carried};
    } else {
        const double a = high ? part_bound(de >= 0 ? 0 : 1) : polymax::max_abs_interval(q);
        out.rows[2] = Cost{M.cost.k * D, a, 1 + M.cost.m, M.cost.rescaled, carried};
    }
    // GQET (Sunderhauf, Eqs. 7-8): Chebyshev coefficients of q, maximum on the unit circle (bounded by
    // sum |c_n| at high degree).
    double circle = 0.0;
    if (high) for (double v : c) circle += std::abs(v);
    else circle = polymax::max_abs_circle(c);
    out.rows[3] = Cost{M.cost.k * D, circle, 1 + M.cost.m, M.cost.rescaled, carried};

    // Ko (arXiv:2608.15161, Corollary 1): degree padded to a power of two Dp; at most 4Dp - 1 queries
    // each to controlled-U and controlled-U^dagger; subnormalisation sqrt(2) times the largest |q| over
    // the 4Dp points cos(2 pi r / 4Dp) (my reading of its rescaling); log2(Dp) + 4 extra ancillas.
    std::uint64_t Dp = 1;
    std::size_t log_Dp = 0;
    while (Dp < D) { Dp *= 2; ++log_Dp; }
    double top = 0.0;
    for (std::uint64_t r = 0; r < 4 * Dp; ++r) {
        const double x = std::cos(2.0 * std::numbers::pi * double(r) / double(4 * Dp));
        top = std::max(top, std::abs(high ? polymax::clenshaw(c, x) : polymax::eval(q, x)));
    }
    out.rows[4] = Cost{M.cost.k * 2 * (4 * Dp - 1), std::sqrt(2.0) * top, log_Dp + 4 + M.cost.m, M.cost.rescaled, carried};

    // Truncated Chebyshev: keep T_0 ... T_K, with K the lowest degree (at least 1) whose tail
    // sum_{k > K} |c_k| fits the budget; the tail bounds the error on [-1, 1], hence in operator norm
    // for Hermitian M (Mason & Handscomb 2003; Higham 2008, Def. 1.2).
    if (P.budget > 0.0) {
        std::vector<double> suffix(c.size() + 1, 0.0);
        for (std::size_t k = c.size(); k-- > 0;) suffix[k] = suffix[k + 1] + std::abs(c[k]);
        std::size_t K = 1;
        while (K < c.size() && suffix[K + 1] > P.budget) ++K;
        K = std::min(K, c.size() - 1);
        const std::vector<double> raw(c.begin(), c.begin() + long(K) + 1);
        const std::vector<double> cK = clean(raw);        // rounding-level coefficients dropped...
        double dropped = 0.0;                             // ...and charged to the error bound
        for (std::size_t k = 0; k < raw.size(); ++k) dropped += std::abs(raw[k] - cK[k]);
        out.rows[5] = qsvt_on_cheb(cK, M.cost, carried + suffix[K + 1] + dropped);
        out.realised_cheb = cK;
    }
}

inline Info compute_info(const Term& t) {
    Info out;
    if (const Leaf* l = std::get_if<Leaf>(&t->kind)) {
        out.cost = Cost{1, 1.0, 0, 0};                 // a unitary encodes itself (Example 2.3)
        if (l->p.x == 0 && l->p.z == 0) out.cost.k = 0;   // the identity (times a phase) is Cobble's Const: no
                                                          // query (expr.py, Const.queries() = 0); decision of 6 Nov
        out.hermitian = is_hermitian(l->p);            // known exactly for a Pauli string
    } else if (const Adjoint* a = std::get_if<Adjoint>(&t->kind)) {
        out.cost = info(a->a).cost;
        out.hermitian = info(a->a).hermitian;
    } else if (const Sum* s = std::get_if<Sum>(&t->kind)) {
        std::size_t m = 0;
        out.hermitian = true;
        for (std::size_t j = 0; j < s->t.size(); ++j) {
            const Info& c = info(s->t[j]);
            out.cost.k += c.cost.k;
            out.cost.alpha += std::abs(s->c[j]) * c.cost.alpha;       // l1 norm (Def. 2.4)
            out.cost.eps += std::abs(s->c[j]) * c.cost.eps;
            out.cost.rescaled += c.cost.rescaled;
            m = std::max(m, c.cost.m);
            out.hermitian = out.hermitian && c.hermitian && s->c[j].imag() == 0.0;
        }
        out.cost.m = ceil_log2(s->t.size()) + m;
        if (!out.hermitian) out.hermitian = adjoint_symmetric(*s);
    } else if (const Product* p = std::get_if<Product>(&t->kind)) {
        std::size_t m = 0;
        out.cost.alpha = 1.0;
        double with_eps = 1.0;                                 // prod (alpha_j + eps_j)
        for (const Term& f : p->t) {
            const Info& c = info(f);
            out.cost.k += c.cost.k;
            out.cost.alpha *= c.cost.alpha;
            with_eps *= c.cost.alpha + c.cost.eps;
            out.cost.rescaled += c.cost.rescaled;
            m = std::max(m, c.cost.m);
        }
        out.cost.eps = with_eps - out.cost.alpha;              // telescoping bound
        out.cost.m = ceil_log2(p->t.size()) + m;
        out.hermitian = palindromic(*p);
    } else if (const DirectSum* d = std::get_if<DirectSum>(&t->kind)) {
        const std::vector<bool> r = rescaled_blocks(*d);
        out.hermitian = true;
        for (std::size_t j = 0; j < d->t.size(); ++j) {
            const Info& c = info(d->t[j]);
            out.cost.k += c.cost.k;
            out.cost.alpha = std::max(out.cost.alpha, c.cost.alpha);
            out.cost.m = std::max(out.cost.m, c.cost.m + (r[j] ? 1 : 0));
            out.cost.rescaled += c.cost.rescaled + (r[j] ? 1 : 0);
            out.hermitian = out.hermitian && c.hermitian;      // scaling by a positive number keeps it
        }
        // A block kept within the tolerance instead of rescaled is off by at most alpha_max - alpha_j.
        for (std::size_t j = 0; j < d->t.size(); ++j) {
            const Cost& c = info(d->t[j]).cost;
            out.cost.eps = std::max(out.cost.eps, c.eps + (r[j] ? 0.0 : out.cost.alpha - c.alpha));
        }
    } else if (const Tensor* x = std::get_if<Tensor>(&t->kind)) {
        out.cost.alpha = 1.0;
        out.hermitian = true;
        double with_eps = 1.0;
        for (const Term& f : x->t) {
            const Info& c = info(f);
            out.cost.k += c.cost.k;
            out.cost.alpha *= c.cost.alpha;
            with_eps *= c.cost.alpha + c.cost.eps;
            out.cost.m += c.cost.m;                            // each factor has its own workspace
            out.cost.rescaled += c.cost.rescaled;
            out.hermitian = out.hermitian && c.hermitian;
        }
        out.cost.eps = with_eps - out.cost.alpha;
    } else if (const Unitary* u = std::get_if<Unitary>(&t->kind)) {
        out.cost = Cost{1, 1.0, 0, 0};                 // a unitary encodes itself (Example 2.3)
        out.hermitian = (u->U - u->U.adjoint()).norm() <= 1e-12 * u->U.norm();
    } else if (std::holds_alternative<Rotation>(t->kind)) {
        out.cost = Cost{1, 1.0, 0, 0};                 // a unitary for every angle
        out.hermitian = false;                         // not for a general angle
    } else {
        const Poly& P = std::get<Poly>(t->kind);
        poly_rows(P, out);
        // The cheapest valid method by alpha * k (Cobble's total cost); ties (equal up to rounding) go to
        // fewer workspace qubits, then to the order LCU, Horner, QSVT, GQET, Ko, Truncated.
        std::size_t best = 0;
        for (std::size_t j = 1; j < out.rows.size(); ++j) {
            if (!out.rows[j]) continue;
            const Cost& a = *out.rows[j];
            const Cost& b = *out.rows[best];
            const double ca = a.alpha * double(a.k), cb = b.alpha * double(b.k);
            const bool tie = std::abs(ca - cb) <= 1e-9 * std::max(ca, cb);   // equal up to rounding
            if ((!tie && ca < cb) || (tie && a.m < b.m)) best = j;
        }
        out.method = PolyMethod(best);
        out.cost = *out.rows[best];
        out.hermitian = info(P.a).hermitian;                   // real coefficients keep it
    }
    return out;
}

}  // namespace detail

// Cost and Hermiticity of a term: computed the first time it is asked for, then kept in the node.
// Terms are built and read on one thread; two threads asking at once would race to fill it.
inline const Info& info(const Term& t) {
    if (!t->info) t->info = detail::compute_info(t);
    return *t->info;
}

inline const Cost& cost(const Term& t) { return info(t).cost; }

// ---- The meaning of a term, applied to a vector -----------------------------------------------

// [[t]] x, or [[t]]^dagger x when dagger is true, without building any matrix.
// (Not called "apply": std::apply would be found instead through argument-dependent lookup.)
inline Eigen::VectorXcd act(const Term& t, const Eigen::VectorXcd& x, bool dagger = false) {
    detail::require(x.size() == Eigen::Index(std::size_t(1) << t->qubits), "cobble: vector of the wrong size.");
    if (const Leaf* l = std::get_if<Leaf>(&t->kind)) {
        // P |c> = i^k (-1)^{|z & c|} |c ^ x>, so (P x)[c ^ x] = ph(c) x[c] and (P^dagger x)[c] = conj(ph(c)) x[c ^ x].
        static const Coeff ik[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        const PauliString& P = l->p;
        Eigen::VectorXcd y(x.size());
        for (std::uint64_t c = 0; c < std::uint64_t(x.size()); ++c) {
            const Coeff ph = ik[P.k % 4] * ((std::popcount(P.z & c) % 2) ? -1.0 : 1.0);
            if (dagger) y[Eigen::Index(c)] = std::conj(ph) * x[Eigen::Index(c ^ P.x)];
            else        y[Eigen::Index(c ^ P.x)] = ph * x[Eigen::Index(c)];
        }
        return y;
    }
    if (const Adjoint* a = std::get_if<Adjoint>(&t->kind)) return act(a->a, x, !dagger);
    if (const Sum* s = std::get_if<Sum>(&t->kind)) {
        Eigen::VectorXcd y = Eigen::VectorXcd::Zero(x.size());
        for (std::size_t j = 0; j < s->t.size(); ++j)
            y += (dagger ? std::conj(s->c[j]) : s->c[j]) * act(s->t[j], x, dagger);
        return y;
    }
    if (const Product* p = std::get_if<Product>(&t->kind)) {
        // (t0 t1 ... t_last) x applies t_last first; the adjoint reverses the order.
        Eigen::VectorXcd y = x;
        const std::size_t n = p->t.size();
        for (std::size_t i = 0; i < n; ++i) y = act(p->t[dagger ? i : n - 1 - i], y, dagger);
        return y;
    }
    if (const DirectSum* d = std::get_if<DirectSum>(&t->kind)) {
        // Block j acts on the j-th segment: the extra qubits sit above the data qubits.
        const Eigen::Index B = Eigen::Index(std::size_t(1) << d->t[0]->qubits);
        Eigen::VectorXcd y = Eigen::VectorXcd::Zero(x.size());
        for (std::size_t j = 0; j < d->t.size(); ++j)
            y.segment(Eigen::Index(j) * B, B) = act(d->t[j], x.segment(Eigen::Index(j) * B, B), dagger);
        return y;
    }
    if (const Tensor* tt = std::get_if<Tensor>(&t->kind)) {
        // Factor j acts on its own qubits, above those of factors 0, ..., j-1; factors on different
        // qubits commute, so they are applied one at a time, each on every slice of its qubits.
        Eigen::VectorXcd y = x;
        std::size_t low = 0;
        for (const Term& f : tt->t) {
            const std::size_t s = std::size_t(1) << low, b = std::size_t(1) << f->qubits;
            const std::size_t outer = std::size_t(y.size()) / (s * b);
            Eigen::VectorXcd slice = Eigen::VectorXcd::Zero(Eigen::Index(b));
            Eigen::VectorXcd out;
            for (std::size_t o = 0; o < outer; ++o)
                for (std::size_t i = 0; i < s; ++i) {
                    for (std::size_t u = 0; u < b; ++u) slice[Eigen::Index(u)] = y[Eigen::Index(o * s * b + u * s + i)];
                    out = act(f, slice, dagger);
                    for (std::size_t u = 0; u < b; ++u) y[Eigen::Index(o * s * b + u * s + i)] = out[Eigen::Index(u)];
                }
            low += f->qubits;
        }
        return y;
    }
    if (const Unitary* u = std::get_if<Unitary>(&t->kind))
        return dagger ? Eigen::VectorXcd(u->U.adjoint() * x) : Eigen::VectorXcd(u->U * x);
    if (std::holds_alternative<Rotation>(t->kind))
        throw std::invalid_argument("cobble: a rotation has no matrix until its angle is given; use bind(term, theta).");
    // Poly: a_0 x + a_1 M x + ... + a_d M^d x by Horner's rule; real coefficients, so the
    // adjoint is the same polynomial of M^dagger.
    const Poly& P = std::get<Poly>(t->kind);
    const long d = detail::true_degree(P.coeff);
    Eigen::VectorXcd y = P.coeff[std::size_t(d)] * x;
    for (long j = d - 1; j >= 0; --j) y = act(P.a, y, dagger) + P.coeff[std::size_t(j)] * x;
    return y;
}

// Do a and b mean the same matrix? Checked on r random vectors (Freivalds-style, as in the check
// policy of 4 Nov: r = 2, relative tolerance 1e-10, at most 17 qubits).
inline bool same_matrix(const Term& a, const Term& b, int r = 2, double tol = 1e-10) {
    detail::require(a->qubits == b->qubits, "cobble: the two terms act on different numbers of qubits.");
    detail::require(a->qubits <= 17, "cobble: checks are limited to 17 qubits.");
    static std::mt19937_64 rng(20261105);
    std::normal_distribution<double> g(0.0, 1.0);
    const Eigen::Index N = Eigen::Index(std::size_t(1) << a->qubits);
    for (int i = 0; i < r; ++i) {
        Eigen::VectorXcd x(N);
        for (Eigen::Index j = 0; j < N; ++j) x[j] = Coeff(g(rng), g(rng));
        const Eigen::VectorXcd ya = act(a, x);
        const Eigen::VectorXcd yb = act(b, x);
        const double scale = std::max({ya.norm(), yb.norm(), 1e-300});
        if ((ya - yb).norm() > tol * scale) return false;
    }
    return true;
}


// ---- Parameters (added 6 Nov) -------------------------------------------------------------------

// One more than the largest parameter number used in t (0 if t has no rotation).
inline std::size_t parameter_count(const Term& t) {
    if (const Rotation* r = std::get_if<Rotation>(&t->kind)) return r->param + 1;
    if (const Adjoint* a = std::get_if<Adjoint>(&t->kind)) return parameter_count(a->a);
    if (const Poly* p = std::get_if<Poly>(&t->kind)) return parameter_count(p->a);
    std::size_t n = 0;
    auto over = [&](const std::vector<Term>& ts) { for (const Term& c : ts) n = std::max(n, parameter_count(c)); };
    if (const Sum* s = std::get_if<Sum>(&t->kind)) over(s->t);
    if (const Product* p = std::get_if<Product>(&t->kind)) over(p->t);
    if (const DirectSum* d = std::get_if<DirectSum>(&t->kind)) over(d->t);
    if (const Tensor* x = std::get_if<Tensor>(&t->kind)) over(x->t);
    return n;
}

// The same term with every rotation replaced by its matrix at the given angles (a unitary leaf), so that
// act and same_matrix can be used on it. Costs are unchanged: a rotation and a unitary leaf cost the same.
inline Term bind(const Term& t, const std::vector<double>& theta) {
    if (const Rotation* r = std::get_if<Rotation>(&t->kind)) {
        detail::require(r->param < theta.size(), "cobble: bind: not enough parameter values.");
        const double a = theta[r->param];
        const Eigen::MatrixXcd P = matrix(r->p);
        const Eigen::MatrixXcd U = std::cos(a / 2) * Eigen::MatrixXcd::Identity(P.rows(), P.cols()) - Coeff(0, std::sin(a / 2)) * P;
        return unitary_leaf("R_" + std::to_string(r->param), U);
    }
    auto all = [&](const std::vector<Term>& ts) { std::vector<Term> o; for (const Term& c : ts) o.push_back(bind(c, theta)); return o; };
    if (const Adjoint* a = std::get_if<Adjoint>(&t->kind)) return adjoint(bind(a->a, theta));
    if (const Sum* s = std::get_if<Sum>(&t->kind)) return sum(s->c, all(s->t));
    if (const Product* p = std::get_if<Product>(&t->kind)) return product(all(p->t));
    if (const DirectSum* d = std::get_if<DirectSum>(&t->kind)) return direct_sum(all(d->t));
    if (const Tensor* x = std::get_if<Tensor>(&t->kind)) return tensor(all(x->t));
    if (const Poly* p = std::get_if<Poly>(&t->kind)) return poly(bind(p->a, theta), p->coeff, p->budget);
    return t;   // Pauli and unitary leaves
}

}  // namespace cobble
