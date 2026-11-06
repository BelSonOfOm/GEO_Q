# GEO_Q — a C++ compiler for block encodings and quantum circuits

GEO_Q is a rebuild and extension of **Cobble** (Yuan et al., *Cobble: Compiling Block Encodings for Quantum Computational Linear Algebra*, arXiv:2511.01736) in C++20. A program is a term: sums, products, tensor products, direct sums, adjoints and polynomials of Pauli strings, unitaries and parameterised rotations. GEO_Q computes what the term costs, rewrites it into something cheaper with the same matrix, and generates a gate circuit, which it runs on its own state-vector simulator to check that the circuit implements the term.

## What it does

**Cost model.** Every term gets Cobble's cost: queries *k*, subnormalisation *α* (the circuit encodes the matrix divided by α) and workspace qubits *m*, following Cobble's Tables 1 and 2, plus an error bound ε in operator norm. Polynomials are costed under six methods (LCU, Horner, QSVT, GQET, Ko's method, and QSVT on a Chebyshev series truncated to a given error budget) and the cheapest valid one is chosen.

**Optimisation passes.**
- *Cobble's rewrite rules*: flattening and fusion of sums and polynomials, factoring of common product and tensor factors, polynomial composition, adjoints pushed to the leaves.
- *Symbol route*: detects when a term's matrix is invariant under the translations of a periodic grid and rewrites it as a linear combination of grid shifts, certifying from the coefficients alone when the resulting α equals the operator norm (no block encoding can do better).
- *Boundary route*: handles stencils with Dirichlet, Neumann, Robin or mixed boundaries by writing each diagonal of the matrix with reflections, following the explicit constructions of Kharazi et al. (arXiv:2407.18347).

The optimiser keeps a rewrite only when it is cheaper and is the same matrix, so it is never worse than its input.

**Code generation.** Pauli strings become one-qubit gates. Shifts become increment circuits; reflections become multi-controlled Z gates. Sums become PREPARE–SELECT–PREPARE†. Products are sequenced on shared workspace with a success counter. Polynomials become QSVT circuits, with phases computed in C++ (Dong, Meng, Whaley & Lin's method with continuation, plus a fallback). Every generated circuit is run on the simulator, and α times its top-left block is compared with the term's matrix.

**Variational circuits.** Rotations exp(−iθP/2) with shared, symbolic angles; expectation values; gradients by the parameter-shift rule.

**Resource estimates.** Toffoli counts per call (unary-iteration SELECT, controlled increments and reflections) as a second cost alongside Cobble's queries × α.

## Results (selected; full output in `bench/*_output.txt`)

| Benchmark | Baseline | GEO_Q | Note |
|---|---|---|---|
| Cobble Table 3, four benchmarks | Cobble's unoptimised costs | Cobble's optimised costs, reproduced | e.g. laplacian-filter 8 × 59.3 → 2 × 27.8 |
| laplacian-filter, concrete 8×8 periodic stencil | Cobble's optimised form: 1112.8 | 357.0 (α at the floor ‖T‖) | queries × α |
| 1D upwind advection, 64 points | Pauli decomposition | 47× fewer estimated Toffolis (× α) | |
| 1D Laplacian, Dirichlet / Neumann | Kharazi et al.'s hand-built encodings | the same encodings, found from the matrix | same α = 4 and Toffoli count |
| Spectral thresholding, Hamiltonian simulation | — | compiled to QSVT circuits, error ≤ 1e-13 | checked on the simulator |

One more finding: the spectral-thresholding benchmark in Cobble's artifact uses the Chebyshev series of sign(x) without its alternating signs. The corrected series has α = 1.180 instead of 2.716.

## Scope and limits

The structure passes apply to constant-coefficient stencils on grids, detected densely up to 12 qubits. The gate-level costs are estimates under a stated model (PREPARE not counted, which favours the baseline); on small grids the Pauli decomposition can be cheaper in Toffolis, and the advantage appears as grids grow. Multi-controlled gates are not decomposed further. When the cost model prefers GQET, Ko or Horner, the code generator builds QSVT or the sum of powers instead, and says so.

## Build and run

Requires a C++20 compiler, CMake ≥ 3.16 and Eigen 3.

```
mkdir build && cd build && cmake .. && make -j4
ctest                          # 12 test programs
./geoq_compile_report          # what the compiler produces, checked on the simulator
./geoq_literature_report       # structure passes against the Pauli form and hand-built encodings
./geoq_suite_report            # structured-operator suite
```

## Layout

`GEO_Q/` holds the core: `GEO_Q.h` (terms, costs, meaning), `rewrite.h` (Cobble's rewrite rules), `compile.h` (code generation, simulation, gradients), `polymax.h` (polynomial maxima), `pauli.h`, and `Basics.h` (the state-vector simulator). `bench/` holds the benchmarks, structure passes and reports; `tests/` holds the test programs.

## References

Yuan et al., arXiv:2511.01736 (Cobble) · Gilyén, Su, Low & Wiebe, STOC 2019 (QSVT) · Dong, Meng, Whaley & Lin, PRA 103, 042419 (2021) · Dong, Lin, Ni & Wang (2022) · Kharazi et al., arXiv:2407.18347 · Mahmud & Herrman, arXiv:2607.11596 · Camps, Lin, Van Beeumen & Yang (banded circulant encodings) · Babbush et al., PRX 8, 041015 (2018) · Gidney, Quantum 2, 74 (2018) · Sünderhauf (GQET) · Ko, arXiv:2608.15161 · Sachdeva & Vishnoi (2014) · Mitarai et al. (2018) and Schuld et al. (2019) (parameter shift).
