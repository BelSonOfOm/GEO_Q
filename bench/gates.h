#pragma once
// Toffoli estimate for one call of a block encoding (written by Claude, 5 Nov 2026). This is an
// estimate under a stated model, not a compiled circuit. Sources for each part of the model:
//  - SELECT over L unitaries by unary iteration: L - 1 temporary logical-ANDs, i.e. 4(L - 1) T gates
//    (Babbush et al., "Encoding electronic spectra in quantum circuits with linear T complexity",
//    PRX 8, 041015, 2018). Each unitary is then controlled by a single qubit.
//  - A Pauli string controlled by one qubit is a product of CNOT/CZ/CY gates: Clifford, 0 Toffolis.
//  - Increment on m qubits: m - 1 Toffolis with one ancilla (Kharazi et al., arXiv:2407.18347,
//    Sec. III.B, citing their Ref. [52]); an upper bound for small m.
//    Controlled increment on m qubits = increment on m + 1 qubits with the control as lowest bit,
//    then X on the control (our two-line derivation, checked in tests/test_literature.cpp): m Toffolis.
//  - S^a with |a| = sum of distinct 2^k: S^(2^k) is an increment of the top n - k qubits, one per
//    binary digit of |a| (our construction; S^-a is its inverse, same count).
//  - Reflection I - 2P on a set G of grid points: G is split into disjoint subcubes (some bits fixed,
//    the rest free) by Shannon expansion; I - 2P is the product of the subcube reflections because the
//    subcube projectors are orthogonal. A subcube reflection with f fixed bits, controlled by one more
//    qubit, is a C^f Z: f - 1 temporary ANDs (Gidney, "Halving the cost of quantum addition",
//    Quantum 2, 74, 2018).
//  - PREPARE is not counted: it is made of rotations, not Toffolis, and the Pauli form has more terms
//    than the shift form, so leaving it out favours the Pauli form.
#include "GEO_Q.h"
#include <bit>
#include <cmath>
#include <limits>
#include <map>

namespace bench::gates {
using namespace cobble;

// Toffolis of each unitary leaf when controlled by one qubit, keyed by the node's id.
inline std::map<std::uint64_t, double>& registry() {
    static std::map<std::uint64_t, double> r;
    return r;
}

inline double controlled_increment(std::size_t m) { return double(m); }

// S^a on n qubits (N = 2^n points), controlled by one qubit. a is taken as its representative in (-N/2, N/2].
inline double controlled_shift(std::size_t n, long a) {
    const long N = long(1) << n;
    long r = ((a % N) + N) % N;
    if (r > N / 2) r -= N;
    const unsigned long u = (unsigned long)std::labs(r);
    double c = 0.0;
    for (std::size_t k = 0; k < n; ++k)
        if ((u >> k) & 1UL) c += controlled_increment(n - k);
    return c;
}

// Disjoint subcubes covering a set of indices in [0, 2^bits): each is (mask of fixed bits, their values).
struct Cube { std::uint64_t mask = 0, value = 0; };

inline void cover(const std::vector<bool>& f, int bits, std::uint64_t mask, std::uint64_t value, std::vector<Cube>& out) {
    bool any = false, all = true;
    for (bool b : f) { any = any || b; all = all && b; }
    if (!any) return;
    if (all) { out.push_back({mask, value}); return; }
    const std::size_t half = f.size() / 2;
    const std::vector<bool> f0(f.begin(), f.begin() + std::ptrdiff_t(half)), f1(f.begin() + std::ptrdiff_t(half), f.end());
    const std::uint64_t bit = std::uint64_t(1) << (bits - 1);
    if (f0 == f1) { cover(f0, bits - 1, mask, value, out); return; }   // the top bit is free
    cover(f0, bits - 1, mask | bit, value, out);
    cover(f1, bits - 1, mask | bit, value | bit, out);
}

inline std::vector<Cube> cubes(const std::vector<bool>& in_set) {
    int bits = 0;
    while ((std::size_t(1) << bits) < in_set.size()) ++bits;
    std::vector<Cube> out;
    cover(in_set, bits, 0, 0, out);
    return out;
}

inline double controlled_reflection(const std::vector<bool>& in_set) {
    double c = 0.0;
    for (const Cube& q : cubes(in_set)) c += std::max(0, std::popcount(q.mask) - 1);
    return c;
}

constexpr double unknown = std::numeric_limits<double>::quiet_NaN();

// A term used as one of the unitaries of a SELECT, controlled by one qubit.
inline double controlled(const Term& t) {
    if (std::holds_alternative<Leaf>(t->kind)) return 0.0;
    if (std::holds_alternative<Rotation>(t->kind)) return 0.0;   // a controlled rotation: rotations, no Toffoli
    if (std::holds_alternative<Unitary>(t->kind)) {
        auto it = registry().find(t->id);
        return it == registry().end() ? unknown : it->second;
    }
    if (const Tensor* x = std::get_if<Tensor>(&t->kind)) {
        double c = 0.0;
        for (const Term& f : x->t) c += controlled(f);
        return c;
    }
    return unknown;
}

// One call of the block encoding of t: Toffolis of SELECT (unary iteration + controlled unitaries).
inline double per_call(const Term& t) {
    if (const Sum* s = std::get_if<Sum>(&t->kind)) {
        double c = double(s->t.size()) - 1.0;
        for (const Term& u : s->t) c += controlled(u);
        return c;
    }
    if (const Tensor* x = std::get_if<Tensor>(&t->kind)) {   // factors run side by side
        double c = 0.0;
        for (const Term& f : x->t) c += per_call(f);
        return c;
    }
    return controlled(t);   // a single unitary (counted controlled: an upper bound)
}

}  // namespace bench::gates
