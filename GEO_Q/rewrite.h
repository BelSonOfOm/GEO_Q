#pragma once
// Cobble's rewrite optimiser, ported to GEO_Q terms (written by Claude, 6 Nov 2026).
// Source: Cobble's artifact, cobble/src/cobble/optimize.py and the optimize() methods of expr.py
// (Zenodo 18898778), rule for rule, with TOL = 1e-12 as in its polynomial.py. Adaptations to our language:
//   - Cobble's Const(c) is c times the identity Pauli leaf here (it costs no query, as Const does);
//   - coefficients of sums may be complex; a scalar is moved into a polynomial only when it is real
//     (polynomial coefficients are real, Cobble Sec. 4.3);
//   - Cobble's If is our direct sum; its rule "if x then A else A => I (x) A" becomes: a direct sum of
//     identical blocks is A tensor the identity on the selector qubits;
//   - the adjoint of a sum conjugates the coefficients (Cobble's are real);
//   - Division and Cobble's other If rules have no counterpart in GEO_Q and are not ported.
// rewrite() applies the rules; optimize() keeps the result only if it is cheaper by alpha x queries and is
// the same matrix (never worse, as for the routes in bench/symbol_route.h).
#include "GEO_Q.h"
#include <map>
#include <optional>

namespace cobble {

namespace rw {

constexpr double TOL = 1e-12;   // Cobble, polynomial.py

using P = std::vector<double>;

inline P trim(P a) {
    std::size_t i = a.size();
    while (i > 1 && std::abs(a[i - 1]) < TOL) --i;
    a.resize(std::max<std::size_t>(i, 1));
    return a;
}
inline P add(const P& a, const P& b) {
    P o(std::max(a.size(), b.size()), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) o[i] += a[i];
    for (std::size_t i = 0; i < b.size(); ++i) o[i] += b[i];
    return o;
}
inline P scale(const P& a, double c) { P o = a; for (double& v : o) v *= c; return o; }
inline P mul(const P& a, const P& b) {
    P o(a.size() + b.size() - 1, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i]) < TOL) continue;
        for (std::size_t j = 0; j < b.size(); ++j)
            if (std::abs(b[j]) >= TOL) o[i + j] += a[i] * b[j];
    }
    return trim(o);
}
inline P compose(const P& outer, const P& inner) {   // outer(inner(x)), Horner
    P o{0.0};
    for (std::size_t k = outer.size(); k-- > 0;) o = add(mul(o, inner), P{outer[k]});
    return trim(o);
}

inline bool is_identity_leaf(const Term& t) {
    const Leaf* l = std::get_if<Leaf>(&t->kind);
    return l && l->p.x == 0 && l->p.z == 0;
}
// The value c if t means c times the identity (Cobble's Const).
inline std::optional<Coeff> scalar_value(const Term& t) {
    static const Coeff ik[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    if (const Leaf* l = std::get_if<Leaf>(&t->kind))
        if (l->p.x == 0 && l->p.z == 0) return ik[l->p.k % 4];
    if (const Sum* s = std::get_if<Sum>(&t->kind)) {
        Coeff v = 0.0;
        for (std::size_t j = 0; j < s->t.size(); ++j) {
            const auto w = scalar_value(s->t[j]);
            if (!w) return std::nullopt;
            v += s->c[j] * *w;
        }
        return v;
    }
    return std::nullopt;
}
inline Term identity(std::size_t n) { return leaf(PauliString::from_string(std::string(n, 'I'))); }
inline Term scalar(Coeff c, std::size_t n) { return std::abs(c - 1.0) < TOL ? identity(n) : sum({c}, {identity(n)}); }

}  // namespace rw

inline Term rewrite(const Term& t);

namespace rw {

// Poly(base, p) for an already rewritten base, simplified as in Cobble's Poly.optimize: a polynomial
// base is composed into, a scalar base is evaluated, a constant p gives c I (nullptr for 0), p = x gives base.
inline Term make_poly(const Term& base, P p, double budget = 0.0) {
    p = trim(p);
    if (const Poly* q = std::get_if<Poly>(&base->kind)) return make_poly(q->a, compose(p, q->coeff), budget);
    if (p.size() <= 1) return std::abs(p[0]) < TOL ? nullptr : scalar(p[0], base->qubits);
    if (const auto v = scalar_value(base); v && std::abs(v->imag()) < TOL) {
        const double val = polymax::eval(p, v->real());
        return std::abs(val) < TOL ? nullptr : scalar(val, base->qubits);
    }
    if (p.size() == 2 && std::abs(p[0]) < TOL && std::abs(p[1] - 1.0) < TOL) return base;
    return poly(base, p, budget);
}

using Terms = std::vector<std::pair<Coeff, Term>>;

inline Term build_sum(const Terms& ts) {
    std::vector<Coeff> c;
    std::vector<Term> t;
    for (const auto& [a, b] : ts) { c.push_back(a); t.push_back(b); }
    return sum(c, t);
}

// Flatten nested sums (tensor products kept whole, so that they can be factored), combine like terms
// (same canonical label), drop zero coefficients except on polynomials (optimize.py, _flatten_and_combine_terms).
inline void flatten(const Terms& in, Terms& out) {
    for (const auto& [c, t] : in) {
        if (std::holds_alternative<Tensor>(t->kind)) out.emplace_back(c, t);
        else if (const Sum* s = std::get_if<Sum>(&t->kind)) {
            Terms inner;
            for (std::size_t j = 0; j < s->t.size(); ++j) inner.emplace_back(s->c[j], s->t[j]);
            Terms sub;
            flatten(inner, sub);
            for (const auto& [c2, t2] : sub) out.emplace_back(c * c2, t2);
        } else out.emplace_back(c, rewrite(t));
    }
}
inline Terms combine(const Terms& flat) {
    Terms out;
    std::map<std::uint64_t, std::size_t> at;
    for (const auto& [c, t] : flat) {
        if (auto it = at.find(t->canon); it != at.end()) out[it->second].first += c;
        else { at[t->canon] = out.size(); out.emplace_back(c, t); }
    }
    Terms kept;
    for (const auto& [c, t] : out)
        if (std::abs(c) > TOL || std::holds_alternative<Poly>(t->kind)) kept.emplace_back(c, t);
    return kept;
}

// Two sums that are scalar multiples of each other: same terms, proportional coefficients (_are_scalar_multiples).
inline std::optional<Coeff> multiple(const Term& a, const Term& b) {
    const Sum* sa = std::get_if<Sum>(&a->kind);
    const Sum* sb = std::get_if<Sum>(&b->kind);
    if (!sa || !sb || sa->t.size() != sb->t.size()) return std::nullopt;
    std::map<std::uint64_t, Coeff> ma, mb;
    for (std::size_t j = 0; j < sa->t.size(); ++j) ma[sa->t[j]->canon] += sa->c[j];
    for (std::size_t j = 0; j < sb->t.size(); ++j) mb[sb->t[j]->canon] += sb->c[j];
    if (ma.size() != mb.size()) return std::nullopt;
    for (const auto& [k, v] : ma) if (!mb.count(k)) return std::nullopt;
    const auto& [k0, v0] = *ma.begin();
    if (std::abs(v0) < TOL) return std::nullopt;
    const Coeff s = mb[k0] / v0;
    for (const auto& [k, v] : ma) if (std::abs(mb[k] - v * s) > TOL) return std::nullopt;
    return s;
}

// Group terms by a common factor and pull it out (_try_factor_by_common). extract returns (factor, rest).
template <class Extract, class Build>
inline bool factor_by(Terms& terms, Extract extract, Build build, bool check_scalars) {
    struct Group { Term factor; Terms rest; Terms originals; bool factorable; };
    std::vector<Group> groups;
    for (const auto& [c, t] : terms) {
        const auto fr = extract(t);
        if (!fr) { groups.push_back({nullptr, {}, {{c, t}}, false}); continue; }
        const auto& [f, rest] = *fr;
        bool placed = false;
        for (Group& g : groups) {
            if (!g.factorable) continue;
            if (g.factor->canon == f->canon) { g.rest.emplace_back(c, rest); g.originals.emplace_back(c, t); placed = true; break; }
            if (check_scalars)
                if (const auto s = multiple(g.factor, f)) { g.rest.emplace_back(c * *s, rest); g.originals.emplace_back(c, t); placed = true; break; }
        }
        if (!placed) groups.push_back({f, {{c, rest}}, {{c, t}}, true});
    }
    Terms out;
    bool did = false;
    for (const Group& g : groups) {
        bool ok = g.factorable && g.rest.size() > 1;
        for (const auto& [c, r] : g.rest) ok = ok && !std::holds_alternative<Sum>(r->kind);
        if (ok) { out.emplace_back(1.0, build(g.factor, build_sum(g.rest))); did = true; }
        else out.insert(out.end(), g.originals.begin(), g.originals.end());
    }
    if (did) terms = out;
    return did;
}

// Factor a sum by common left/right product factors and left/right tensor factors, until nothing changes
// (_factor_sum). Returns the single term if everything was factored into one.
inline Term factor_sum(Terms terms) {
    using FR = std::optional<std::pair<Term, Term>>;
    auto prod_left = [](const Term& t) -> FR {
        if (const Product* p = std::get_if<Product>(&t->kind))
            return std::make_pair(p->t.front(), p->t.size() > 2 ? product(std::vector<Term>(p->t.begin() + 1, p->t.end())) : p->t[1]);
        return std::nullopt;
    };
    auto prod_right = [](const Term& t) -> FR {
        if (const Product* p = std::get_if<Product>(&t->kind))
            return std::make_pair(p->t.back(), p->t.size() > 2 ? product(std::vector<Term>(p->t.begin(), p->t.end() - 1)) : p->t[0]);
        return std::nullopt;
    };
    auto tensor_left = [](const Term& t) -> FR {
        if (const Tensor* x = std::get_if<Tensor>(&t->kind); x && x->t.size() == 2) return std::make_pair(x->t[0], x->t[1]);
        return std::nullopt;
    };
    auto tensor_right = [](const Term& t) -> FR {
        if (const Tensor* x = std::get_if<Tensor>(&t->kind); x && x->t.size() == 2) return std::make_pair(x->t[1], x->t[0]);
        return std::nullopt;
    };
    for (int round = 0; round < 64; ++round) {
        if (terms.size() == 1 && std::abs(terms[0].first - 1.0) < TOL) break;
        if (factor_by(terms, prod_left, [](const Term& f, const Term& s) { return product({f, s}); }, false)) continue;
        if (factor_by(terms, prod_right, [](const Term& f, const Term& s) { return product({s, f}); }, false)) continue;
        if (factor_by(terms, tensor_left, [](const Term& f, const Term& s) { return tensor({f, s}); }, true)) continue;
        if (factor_by(terms, tensor_right, [](const Term& f, const Term& s) { return tensor({s, f}); }, true)) continue;
        break;
    }
    if (terms.size() == 1 && std::abs(terms[0].first - 1.0) < TOL) return terms[0].second;
    return build_sum(terms);
}

inline bool real(Coeff c) { return std::abs(c.imag()) < TOL; }

inline Term rewrite_sum(const Sum& s, std::size_t n) {
    Terms in;
    for (std::size_t j = 0; j < s.t.size(); ++j) in.emplace_back(s.c[j], s.t[j]);
    Terms flat;
    flatten(in, flat);
    Terms terms = combine(flat);
    if (terms.empty()) return nullptr;
    if (terms.size() == 1 && std::abs(terms[0].first - 1.0) < TOL) return terms[0].second;

    const Term factored = factor_sum(terms);
    if (!std::holds_alternative<Sum>(factored->kind)) return rewrite(factored);
    terms.clear();
    {
        const Sum& f = std::get<Sum>(factored->kind);
        for (std::size_t j = 0; j < f.t.size(); ++j) {
            Term tj = f.t[j];
            // tensors were kept whole for factoring, and factoring builds products around unrewritten sums
            if (std::holds_alternative<Tensor>(tj->kind) || std::holds_alternative<Product>(tj->kind)) tj = rewrite(tj);
            if (!tj) continue;
            terms.emplace_back(f.c[j], tj);
        }
    }
    // c poly(A, f) -> poly(A, c f) for real c (_merge_scalars_into_expressions)
    for (auto& [c, t] : terms)
        if (const Poly* p = std::get_if<Poly>(&t->kind); p && real(c) && std::abs(c - 1.0) > TOL) {
            Term m = make_poly(p->a, scale(p->coeff, c.real()), p->budget);
            if (m) { t = m; c = 1.0; }
        }
    // poly(A, f) + poly(A, g) -> poly(A, f + g) (_combine_polys_with_same_base)
    {
        Terms others;
        std::vector<std::pair<Term, P>> groups;
        for (const auto& [c, t] : terms) {
            const Poly* p = std::get_if<Poly>(&t->kind);
            if (p && std::abs(c - 1.0) < TOL) {
                bool placed = false;
                for (auto& [b, acc] : groups)
                    if (b->canon == p->a->canon) { acc = add(acc, p->coeff); placed = true; break; }
                if (!placed) groups.emplace_back(p->a, p->coeff);
            } else others.emplace_back(c, t);
        }
        terms.clear();
        for (const auto& [b, acc] : groups)
            if (Term m = make_poly(b, acc)) terms.emplace_back(1.0, m);
        terms.insert(terms.end(), others.begin(), others.end());
    }
    // c A + poly(A, f) -> poly(A, c x + f); or (sum of the non-polynomial terms) + poly(that sum, f)
    // -> poly(that sum, x + f) (_combine_base_expr_with_poly)
    {
        std::vector<std::size_t> polys, nonp;
        for (std::size_t j = 0; j < terms.size(); ++j)
            (std::holds_alternative<Poly>(terms[j].second->kind) ? polys : nonp).push_back(j);
        bool done = false;
        if (!polys.empty() && !nonp.empty()) {
            for (std::size_t i : nonp) {
                for (std::size_t j : polys) {
                    const Poly& pj = std::get<Poly>(terms[j].second->kind);
                    if (std::abs(terms[j].first - 1.0) < TOL && real(terms[i].first) && terms[i].second->canon == pj.a->canon) {
                        Term m = make_poly(pj.a, add(P{0.0, terms[i].first.real()}, pj.coeff));
                        Terms r;
                        if (m) r.emplace_back(1.0, m);
                        for (std::size_t k = 0; k < terms.size(); ++k) if (k != i && k != j) r.push_back(terms[k]);
                        terms = r;
                        done = true;
                        break;
                    }
                }
                if (done) break;
            }
            if (!done) {
                Terms np;
                for (std::size_t i : nonp) np.push_back(terms[i]);
                Term ns = (np.size() == 1 && std::abs(np[0].first - 1.0) < TOL) ? np[0].second : rewrite(build_sum(np));
                if (ns) {
                    for (std::size_t j : polys) {
                        const Poly& pj = std::get<Poly>(terms[j].second->kind);
                        if (std::abs(terms[j].first - 1.0) < TOL && ns->canon == pj.a->canon) {
                            Term m = make_poly(pj.a, add(P{0.0, 1.0}, pj.coeff));
                            Terms r;
                            if (m) r.emplace_back(1.0, m);
                            for (std::size_t k : polys) if (k != j) r.push_back(terms[k]);
                            terms = r;
                            break;
                        }
                    }
                }
            }
        }
    }
    // c I + poly(A, f) -> poly(A, f + c) when the polynomial is the only other term (_merge_const_with_poly)
    {
        Coeff cval = 0.0;
        Terms others;
        for (const auto& [c, t] : terms) {
            if (is_identity_leaf(t)) cval += c * *scalar_value(t);
            else others.emplace_back(c, t);
        }
        if (std::abs(cval) > TOL && real(cval)) {
            if (others.size() == 1 && std::abs(others[0].first - 1.0) < TOL)
                if (const Poly* p = std::get_if<Poly>(&others[0].second->kind)) {
                    P q = p->coeff;
                    q[0] += cval.real();
                    if (Term m = make_poly(p->a, q, p->budget)) return m;
                }
        }
        if (std::abs(cval) > TOL) others.insert(others.begin(), {cval, identity(n)});
        if (others.empty()) return nullptr;
        if (others.size() == 1 && std::abs(others[0].first - 1.0) < TOL) return others[0].second;
        return build_sum(others);
    }
}

// Product fusion (_try_poly_product_fusion): scalars, a single expression E, c E, and poly(E, .) all with
// the same base multiply into one poly(E, .).
inline Term fuse_product(const std::vector<Term>& fs, std::size_t n) {
    Term base;
    P acc;
    double sc = 1.0;
    for (const Term& f : fs) {
        if (const auto v = scalar_value(f)) {
            if (!real(*v)) return nullptr;
            sc *= v->real();
            continue;
        }
        Term b;
        P q;
        if (const Poly* p = std::get_if<Poly>(&f->kind)) { b = p->a; q = p->coeff; }
        else if (const Sum* s = std::get_if<Sum>(&f->kind); s && s->t.size() == 1 && real(s->c[0])) { b = s->t[0]; q = {0.0, s->c[0].real()}; }
        else { b = f; q = {0.0, 1.0}; }
        if (!base) { base = b; acc = q; }
        else if (b->canon == base->canon) acc = mul(acc, q);
        else return nullptr;
    }
    if (!base) return scalar(sc, n);
    if (std::abs(sc - 1.0) > TOL) acc = scale(acc, sc);
    return make_poly(base, acc);
}

}  // namespace rw

// Apply Cobble's rewrite rules (bottom-up). Returns nullptr only if the whole term is the zero matrix.
inline Term rewrite(const Term& t) {
    const std::size_t n = t->qubits;
    if (std::holds_alternative<Leaf>(t->kind) || std::holds_alternative<Unitary>(t->kind) || std::holds_alternative<Rotation>(t->kind))
        return t;
    if (const Sum* s = std::get_if<Sum>(&t->kind)) return rw::rewrite_sum(*s, n);
    if (const Product* p = std::get_if<Product>(&t->kind)) {
        std::vector<Term> fs;
        for (const Term& f : p->t) {
            Term r = rewrite(f);
            if (!r) return nullptr;                                  // a zero factor
            if (const Product* q = std::get_if<Product>(&r->kind)) fs.insert(fs.end(), q->t.begin(), q->t.end());
            else fs.push_back(r);
        }
        if (Term fused = rw::fuse_product(fs, n)) return fused;
        // scalars come out in front; identity factors are dropped
        Coeff sc = 1.0;
        std::vector<Term> rest;
        for (const Term& f : fs) {
            if (const auto v = rw::scalar_value(f)) sc *= *v;
            else rest.push_back(f);
        }
        if (std::abs(sc) < rw::TOL) return nullptr;
        Term core = rest.empty() ? rw::identity(n) : (rest.size() == 1 ? rest[0] : product(rest));
        return std::abs(sc - 1.0) < rw::TOL ? core : sum({sc}, {core});
    }
    if (const Tensor* x = std::get_if<Tensor>(&t->kind)) {
        std::vector<Term> fs;
        for (const Term& f : x->t) {
            Term r = rewrite(f);
            if (!r) return nullptr;
            if (const Tensor* y = std::get_if<Tensor>(&r->kind)) fs.insert(fs.end(), y->t.begin(), y->t.end());
            else fs.push_back(r);
        }
        return tensor(fs);
    }
    if (const Poly* p = std::get_if<Poly>(&t->kind)) {
        // poly(poly(A, f), g) -> poly(A, g o f); poly(c I, f) -> f(c) I; poly(A, x) -> A (Cobble's Poly.optimize)
        Term e = rewrite(p->a);
        if (!e) return std::abs(p->coeff[0]) < rw::TOL ? nullptr : rw::scalar(p->coeff[0], n);   // p(0) I
        return rw::make_poly(e, p->coeff, p->budget);
    }
    if (const DirectSum* d = std::get_if<DirectSum>(&t->kind)) {
        std::vector<Term> bs;
        for (const Term& b : d->t) {
            Term r = rewrite(b);
            if (!r) return t;                                       // a zero block: keep as written
            bs.push_back(r);
        }
        bool same = true;
        for (const Term& b : bs) same = same && b->canon == bs[0]->canon;
        if (same) return tensor({bs[0], rw::identity(n - bs[0]->qubits)});   // Cobble: if x then A else A => I (x) A
        return direct_sum(bs);
    }
    // Adjoint: pushed down (Cobble's Dagger.optimize)
    const Adjoint& a = std::get<Adjoint>(t->kind);
    Term e = rewrite(a.a);
    if (!e) return nullptr;
    if (const Adjoint* b = std::get_if<Adjoint>(&e->kind)) return b->a;
    if (const Leaf* l = std::get_if<Leaf>(&e->kind)) {
        if (is_hermitian(l->p)) return e;
        PauliString q = l->p;
        q.k = (4 - q.k % 4 + 2 * (std::popcount(q.x & q.z) % 2)) % 4;   // (i^k X^x Z^z)^dagger = i^-k (-1)^|x&z| X^x Z^z
        return leaf(q);
    }
    if (const Unitary* u = std::get_if<Unitary>(&e->kind)) return unitary_leaf(u->name + "^dagger", u->U.adjoint());
    if (const Sum* s = std::get_if<Sum>(&e->kind)) {
        std::vector<Coeff> c;
        std::vector<Term> ts;
        for (std::size_t j = 0; j < s->t.size(); ++j) { c.push_back(std::conj(s->c[j])); ts.push_back(adjoint(s->t[j])); }
        return rewrite(sum(c, ts));
    }
    if (const Product* p = std::get_if<Product>(&e->kind)) {
        std::vector<Term> fs;
        for (auto it = p->t.rbegin(); it != p->t.rend(); ++it) fs.push_back(adjoint(*it));
        return rewrite(product(fs));
    }
    if (const Tensor* x = std::get_if<Tensor>(&e->kind)) {
        std::vector<Term> fs;
        for (const Term& f : x->t) fs.push_back(adjoint(f));
        return rewrite(tensor(fs));
    }
    if (const DirectSum* d = std::get_if<DirectSum>(&e->kind)) {
        std::vector<Term> bs;
        for (const Term& b : d->t) bs.push_back(adjoint(b));
        return rewrite(direct_sum(bs));
    }
    if (const Poly* p = std::get_if<Poly>(&e->kind)) {           // real coefficients: p(A)^dagger = p(A^dagger)
        if (info(p->a).hermitian) return e;
        return rewrite(poly(adjoint(p->a), p->coeff, p->budget));
    }
    return adjoint(e);   // a rotation
}

// The rewritten term if it is cheaper (alpha x queries) and the same matrix; otherwise t itself.
inline Term optimize(const Term& t) {
    const Term r = rewrite(t);
    if (!r || r.get() == t.get()) return t;
    const double a = cost(r).alpha * double(cost(r).k), b = cost(t).alpha * double(cost(t).k);
    if (!(a < b || (a == b && cost(r).m < cost(t).m))) return t;
    if (t->qubits <= 17) {
        std::vector<double> theta(parameter_count(t));
        for (std::size_t j = 0; j < theta.size(); ++j) theta[j] = 0.37 + 0.91 * double(j);
        const Term tb = theta.empty() ? t : cobble::bind(t, theta), rb = theta.empty() ? r : cobble::bind(r, theta);
        detail::require(same_matrix(tb, rb, 2, 1e-9), "optimize: a rewrite changed the matrix (a bug).");
    }
    return r;
}

}  // namespace cobble
