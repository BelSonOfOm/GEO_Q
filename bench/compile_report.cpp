// What the compiler produces (Claude, 6 Nov 2026). For each term: the circuit's qubits, gates, alpha and
// error bound, against the cost model; and the largest difference between alpha x (the circuit's block,
// read off the simulator) and the term's matrix. Then a variational circuit: expectation and gradient.
#include "compile.h"
#include "../bench/heat.h"
#include <cstdio>

using cobble::Term;
using cobble::Coeff;

namespace {
void line(const char* name, const Term& t) {
    const cobble::Circuit c = cobble::compile(t);
    const Eigen::MatrixXcd B = cobble::block(c), T = bench::dense_of(t);
    std::size_t controls = 0;
    for (const auto& g : c.gates) controls = std::max(controls, g.on1.size() + g.on0.size());
    const cobble::Cost& k = cobble::cost(t);
    std::printf("%-40s | data %2zu + work %2zu (model %2zu) | %5zu gates, <= %2zu controls | alpha %10.4f (model %10.4f) |"
                " calls %4llu (model %4llu) | eps %.1e | max |diff| %.1e %s\n",
                name, c.data, c.workspace, k.m, c.gates.size(), controls, c.alpha, k.alpha,
                (unsigned long long)c.leaf_calls, (unsigned long long)k.k, c.eps, (B - T).cwiseAbs().maxCoeff(),
                c.notes.empty() ? "" : ("| " + c.notes).c_str());
}
}  // namespace

int main() {
    std::printf("Every circuit below was run on the simulator (Basics.h); 'max |diff|' compares alpha x its block with the term's matrix.\n\n");
    const Term lap = bench::pauli_sum(bench::second_difference(4), 4);
    line("1D Laplacian, 16 pts, Pauli form", lap);
    line("1D Laplacian, 16 pts, after optimise()", bench::optimise(lap, 4, 0));
    const Term dir = bench::pauli_sum_sparse(bench::second_difference(4, false), 4);
    line("1D Dirichlet Laplacian, after optimise()", bench::optimise(dir, 4, 0));
    const Term Hx = bench::pauli_sum(bench::second_difference(2), 2);
    const bench::Pair LF = bench::laplacian_filter(Hx, Hx);
    line("laplacian-filter 4x4, Cobble's input", LF.prog);
    line("laplacian-filter 4x4, Cobble's target", LF.targ);
    line("laplacian-filter 4x4, after optimise()", bench::optimise(LF.targ, 2, 2));
    const Term H = cobble::sum({0.5, -0.3, 0.2}, {cobble::leaf(PauliString::from_string("XZ")), cobble::leaf(PauliString::from_string("ZI")),
                                                  cobble::leaf(PauliString::from_string("YY"))});
    line("spectral thresholding (deg 19), 2-qubit H", bench::spectral_thresholding(H, true));
    line("Hamiltonian simulation (deg 15), 2-qubit H", bench::hamiltonian_simulation(H));
    line("heat, 40 steps, 4x4 grid, eps budget 1e-6", bench::heat_steps(bench::random_walk(2), 40, 1e-6));

    // A variational circuit: 4-qubit periodic chain, shared angles per generator (TFIM shape).
    const std::size_t n = 4;
    std::vector<PauliString> zz, xs;
    for (std::size_t i = 0; i < n; ++i) {
        std::string s(n, 'I'), x(n, 'I');
        s[i] = 'Z'; s[(i + 1) % n] = 'Z'; x[i] = 'X';
        zz.push_back(PauliString::from_string(s));
        xs.push_back(PauliString::from_string(x));
    }
    const Term U = cobble::product({cobble::pauli_evolution(xs, 1), cobble::pauli_evolution(zz, 0)});
    const cobble::Circuit c = cobble::compile(U);
    std::vector<Coeff> ones(n, 1.0);
    std::vector<Term> zs;
    for (std::size_t i = 0; i < n; ++i) { std::string s(n, 'I'); s[i] = 'Z'; zs.push_back(cobble::leaf(PauliString::from_string(s))); }
    const Term O = cobble::sum(ones, zs);
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(16);
    psi[0] = 1.0;
    const std::vector<double> theta{0.4, 1.3};
    const double E = cobble::expectation(c, O, psi, theta);
    const std::vector<double> g = cobble::gradient(c, O, psi, theta);
    std::printf("\nvariational: exp(-i t1 sum X_i / 2) exp(-i t0 sum Z_i Z_(i+1) / 2) on 4 qubits, %zu gates, %zu parameters\n",
                c.gates.size(), cobble::parameter_count(U));
    std::printf("  <0000| U^dagger (sum Z_i) U |0000> at (t0, t1) = (0.4, 1.3): %.12f\n", E);
    std::printf("  gradient by the parameter-shift rule: (%.12f, %.12f)\n", g[0], g[1]);
    std::printf("  (sum Z_i on |0000> after exp(-i t1 sum X/2) is 4 cos t1 = %.12f; t0 only adds a phase on |0000>, so dE/dt0 = 0)\n",
                4 * std::cos(1.3));
}
