#include <iostream>
#include <array>
#include <complex>
#include <Eigen/Dense>

#include<omp.h>

#include <stdexcept>
#include <random>
#include <vector>
#include <functional>

using namespace Eigen;
using Complex = std::complex<double>;

constexpr auto PI = 3.141592653589793238462643383279;

class QuantumError : public std::runtime_error {
public:
    explicit QuantumError(const std::string& message) : std::runtime_error(message) {}
};

class DimensionMismatchError : public QuantumError {
public:
    DimensionMismatchError() : QuantumError("Dimension mismatch error") {}
};

class InvalidDensityMatrixError : public QuantumError {
public:
    InvalidDensityMatrixError() : QuantumError("Invalid density matrix error") {}
};

namespace q_utils {

    inline bool is_PureState(const Matrix<Complex, 2, 2>& rho) {
        double trace_rho_squared = (rho * rho).trace().real();
        return abs(trace_rho_squared - 1.0) < 1e-10;
    }

    inline bool isBasisStateWithQubit0(int i, size_t qubit_index) {
        return ((i >> qubit_index) & 1) == 0;
    }

    inline bool isBasisStateWithQubit1(int i, size_t qubit_index) {
        return ((i >> qubit_index) & 1) == 1;
    }

    // True when U is square and U^dagger U = I up to tol (entrywise).
    inline bool is_unitary(const MatrixXcd& U, double tol = 1e-10) {
        if (U.rows() != U.cols()) return false;
        return (U.adjoint() * U - MatrixXcd::Identity(U.rows(), U.cols())).cwiseAbs().maxCoeff() <= tol;
    }
}

// One qubit. The state is always stored in the computational basis {|0>, |1>};
// to work in another basis temporarily, use BasisChange (below).
class Qubit {
public:
    Qubit() : state(Vector<Complex, 2>::Zero()) {}

    explicit Qubit(const Vector<Complex, 2>& val) {
        Complex norm = val.norm();
        if (norm != Complex(0, 0)) {
            state = val / norm;
        }
        else {
            state = Vector<Complex, 2>::Zero();
        }
    }

    explicit Qubit(const Matrix<Complex, 2, 2>& rho) {
        if (q_utils::is_PureState(rho)) {
            SelfAdjointEigenSolver<Matrix<Complex, 2, 2>> solver(rho);
            auto eigenvalues = solver.eigenvalues();
            auto eigenvectors = solver.eigenvectors();

            for (int i = 0; i < 2; ++i) {
                if (abs(eigenvalues(i) - 1.0) < 1e-10) {
                    state = eigenvectors.col(i);
                    break;
                }
            }
        }
        else {
            throw InvalidDensityMatrixError();
        }
    }

    Qubit(const Qubit& other) = default;
    Qubit& operator=(const Qubit& other) = default;

    inline Vector<Complex, 2> get_state() const { return state; }

    inline Complex operator[](const size_t& i) const { return state[i]; }

    inline Qubit operator+(const Qubit& other) const { return Qubit(Vector<Complex, 2>(state + other.state)); }

    inline Qubit operator-(const Qubit& other) const { return Qubit(Vector<Complex, 2>(state - other.state)); }

    // Inner product <this|other>.
    inline Complex operator*(const Qubit& other) const { return state.dot(other.state); }

    // Apply a 2x2 matrix to the state (no normalisation).
    inline void operator*(const Matrix<Complex, 2, 2>& U) { state = U * state; }

    // Equal as vectors, up to tol (a global phase counts as a difference).
    inline bool operator==(const Qubit& other) const { return (state - other.state).norm() <= 1e-10; }

    inline Matrix<Complex, 2, 2> density_matrix() const {
        return state * state.adjoint();
    }

    // General measurement on one qubit (Nielsen & Chuang, section 2.2.3).
    // M = {M_0, ..., M_{k-1}}: measurement operators with sum_m M_m^dagger M_m = I.
    // Outcome m occurs with probability p(m) = <psi| M_m^dagger M_m |psi>,
    // and the state becomes M_m |psi> / sqrt(p(m)). Returns m.
    // Any POVM {E_m} is covered, since E_m = M_m^dagger M_m (section 2.2.6).
    size_t measure(const std::vector<Matrix<Complex, 2, 2>>& M, std::mt19937_64& rng, double tol = 1e-10) {
        if (M.empty()) {
            throw QuantumError("measure: no measurement operators given.");
        }
        Matrix<Complex, 2, 2> completeness = Matrix<Complex, 2, 2>::Zero();
        for (const auto& Mm : M) {
            completeness += Mm.adjoint() * Mm;
        }
        if ((completeness - Matrix<Complex, 2, 2>::Identity()).cwiseAbs().maxCoeff() > tol) {
            throw QuantumError("measure: operators violate sum_m M_m^dagger M_m = I.");
        }

        std::vector<double> p(M.size());
        for (size_t m = 0; m < M.size(); ++m) {
            p[m] = (M[m] * state).squaredNorm();   // <psi| M_m^dagger M_m |psi>
        }

        std::discrete_distribution<size_t> pick(p.begin(), p.end());   // P(m) = p[m] / sum p
        const size_t outcome = pick(rng);

        state = (M[outcome] * state) / std::sqrt(p[outcome]);
        return outcome;
    }

    // Measurement in the computational basis: the special case M = {|0><0|, |1><1|}.
    size_t measure_qubit(std::mt19937_64& rng) {
        Matrix<Complex, 2, 2> P0 = Matrix<Complex, 2, 2>::Zero(), P1 = Matrix<Complex, 2, 2>::Zero();
        P0(0, 0) = 1;
        P1(1, 1) = 1;
        return measure({ P0, P1 }, rng);
    }

private:
    Vector<Complex, 2> state;
};

namespace q_utils {
    inline Qubit create_zero_state() {
        return Qubit(Vector<Complex, 2>(Complex(1, 0), Complex(0, 0)));
    }

    inline Qubit create_one_state() {
        return Qubit(Vector<Complex, 2>(Complex(0, 0), Complex(1, 0)));
    }
}

class Mixed_Qubit : public Qubit {
public:
    Mixed_Qubit(const Qubit& q) : rho{ q.density_matrix() } {}

    template<std::size_t N>

    explicit Mixed_Qubit(const std::array<Qubit, N>& q, const std::array<double, N>& p) : rho{ Matrix<Complex, 2, 2>::Zero() } {
        for (std::size_t i = 0; i < N; i++) {
            rho += p[i] * (q[i].get_state() * q[i].get_state().adjoint());
        }
    }

    inline Mixed_Qubit(const Matrix<Complex, 2, 2>& rhoo) : rho{rhoo}{}


    inline void operator*(const Matrix<Complex, 2, 2>& U) { rho = U * rho * U.adjoint(); }

    inline Matrix<Complex, 2, 2> density_matrix() { return rho; }

   
private:
    Matrix<Complex, 2, 2> rho;
};

namespace Q_Gates {
    static const Matrix2cd X = (Matrix2cd() << Complex(0, 0), Complex(1, 0), Complex(1, 0), Complex(0, 0)).finished();
    static const Matrix2cd Z = (Matrix2cd() << Complex(1, 0), Complex(0, 0), Complex(0, 0), Complex(-1, 0)).finished();
    static const Matrix2cd Y = (Matrix2cd() << Complex(0, 0), Complex(0, -1), Complex(0, 1), Complex(0, 0)).finished();
    static const Matrix2cd H = (Matrix2cd() << Complex(1 / sqrt(2), 0), Complex(1 / sqrt(2), 0), Complex(1 / sqrt(2), 0), Complex(-1 / sqrt(2), 0)).finished();
    static const Matrix2cd I = Matrix2cd::Identity();

    inline Matrix2cd Rot_X(const double& theta) { return cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * X; }
    inline Matrix2cd Rot_Y(const double& theta) { return cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * Y; }
    inline Matrix2cd Rot_Z(const double& theta) { return cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * Z; }

    inline Matrix2cd R_k(const size_t& k) {
        return (Matrix2cd() << Complex(1, 0), Complex(0, 0), Complex(0, 0), std::exp(Complex(0, (2 * PI) / (1 << k)))).finished();
        
    }
    inline Matrix2cd Rot_N(const Vector3d& n, const double& theta) {
        return cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * (n[0] * X + n[1] * Y + n[2] * Z);
    }

    inline Matrix2cd Gate(const Vector3d& n, const double& theta, const double& alpha) {
        return std::exp(Complex(0, alpha)) * (cos(theta) * Matrix2cd::Identity() + (sin(theta) * Complex(0, -1)) * (n[0] * X + n[1] * Y + n[2] * Z));
    }

    inline void cnot(Qubit& control, Qubit& target) {
        bool pure = (true);
        if (control[1] != Complex(0, 0)) {
            target = Qubit(Vector<Complex, 2>(target[1], target[0]));
        }
    }

    static const Matrix4cd SWAP = (Matrix4cd() <<
        1, 0, 0, 0,
        0, 0, 1, 0,
        0, 1, 0, 0,
        0, 0, 0, 1).finished();

    static const Matrix4cd CNOT = (Matrix4cd() <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 0).finished();

}

// A register of n qubits; n is chosen at run time. The state is stored in the
// computational basis: amplitude i belongs to the basis state whose bit j is qubit j.
// Memory is 2^n complex numbers (16 bytes each).
class Qubits {
public:
    // |0...0> on n qubits.
    explicit Qubits(const size_t& n) : n_(checked_size(n)), state(VectorXcd::Zero(size_t(1) << n_)) {
        state(0) = 1;
    }

    // From an amplitude vector of length 2^n (normalised here).
    explicit Qubits(const VectorXcd& v) : n_(log2_exact(size_t(v.size()))), state(v) {
        normalize();
    }

    inline size_t num_qubits() const { return n_; }
    inline size_t dimension() const { return size_t(state.size()); }

    inline void normalize() {
        Complex norm = state.norm();
        if (norm != Complex(0, 0)) {
            state = state / norm;
        }
        else {
            state = VectorXcd::Zero(state.size());
        }
    }

    // Reduced density matrix of qubit i: rho_ab = sum over the other qubits of psi_(i=a) * conj(psi_(i=b)).
    inline Mixed_Qubit operator[](const size_t& i) const {
        check_qubit(i);
        Matrix<Complex, 2, 2> rho_i = Matrix<Complex, 2, 2>::Zero();
        const size_t bit = size_t(1) << i;
        for (size_t b0 = 0; b0 < dimension(); ++b0) {
            if (b0 & bit) continue;            // b0: qubit i is 0
            const size_t b1 = b0 | bit;        // b1: same index with qubit i set to 1
            rho_i(0, 0) += state(b0) * std::conj(state(b0));
            rho_i(0, 1) += state(b0) * std::conj(state(b1));
            rho_i(1, 0) += state(b1) * std::conj(state(b0));
            rho_i(1, 1) += state(b1) * std::conj(state(b1));
        }
        return Mixed_Qubit(rho_i);
    }

    inline Qubits operator+(const Qubits& other) const {
        same_size(other);
        return Qubits(VectorXcd(state + other.state));
    }

    inline Qubits operator-(const Qubits& other) const {
        same_size(other);
        return Qubits(VectorXcd(state - other.state));
    }

    // Tensor product with one more qubit. Convention: the new qubit becomes qubit n (the highest bit),
    // so the indices of the existing qubits do not change. Index i + 2^n * b holds state(i) * q[b].
    inline Qubits operator^(const Qubit& q) const {
        const size_t D = dimension();
        VectorXcd v(2 * D);
        v.head(D) = q[0] * state;
        v.tail(D) = q[1] * state;
        return Qubits(v);
    }

    // Inner product <this|other>.
    inline Complex operator*(const Qubits& other) const {
        same_size(other);
        return state.dot(other.state);
    }

    // ---- Gates -------------------------------------------------------------------------------

    // Apply a 2^k x 2^k matrix U to the k qubits listed in targets (any order, need not be adjacent).
    // Bit j of U's row/column index refers to qubit targets[j].
    void applyOn(const MatrixXcd& U, const std::vector<size_t>& targets) {
        apply(U, targets, {});
    }

    // Same, but only on the part of the state where every qubit in controls is 1.
    void applyControlledOn(const MatrixXcd& U, const std::vector<size_t>& controls, const std::vector<size_t>& targets) {
        apply(U, targets, controls);
    }

    // One-qubit gate on qubit qubit_index.
    void applyGate(const Eigen::Matrix<std::complex<double>, 2, 2>& gate, const size_t& qubit_index) {
        apply(gate, { qubit_index }, {});
    }

    // Gate on the consecutive qubits start..end; bit j of the gate's index is qubit start + j.
    void applyGate(const MatrixXcd& gate, const size_t& start_index, const size_t& end_index) {
        apply(gate, range(start_index, end_index), {});
    }

    // One-qubit gate U on qubit t_index, applied only where qubit c_index is 1.
    void applyCGate(const Matrix<Complex, 2, 2>& U, const size_t& c_index, const size_t& t_index) {
        apply(U, { t_index }, { c_index });
    }

    // Gate on qubits start..end, applied only where qubit c_index is 1.
    void applyCGate(const MatrixXcd& gate, const size_t& c_index, const size_t& start_index, const size_t& end_index) {
        apply(gate, range(start_index, end_index), { c_index });
    }

    // Exchange qubits a and b (they need not be neighbours).
    void swapQubits(const size_t& a, const size_t& b) {
        check_qubit(a);
        check_qubit(b);
        if (a == b) return;
        const size_t ma = size_t(1) << a, mb = size_t(1) << b;
        for (size_t i = 0; i < dimension(); ++i) {
            if ((i & ma) && !(i & mb)) {
                std::swap(state(i), state(i ^ ma ^ mb));
            }
        }
    }

    // Quantum Fourier transform on the register index (qubit 0 = least significant bit):
    //   |j>  ->  2^(-n/2) sum_k exp(2 pi i j k / 2^n) |k>      (Nielsen & Chuang, eq. 5.2).
    // Circuit of Nielsen & Chuang Fig. 5.1, with qubit n-1 in the role of their top qubit j_1
    // (the most significant bit): Hadamard, then controlled R_k from each less significant qubit,
    // then the qubit order is reversed by swaps.
    void applyQFT() {
        const size_t N = n_;
        for (size_t i = N; i-- > 0;) {
            applyGate(Q_Gates::H, i);
            for (size_t j = i; j-- > 0;) {
                size_t k = i - j + 1;
                applyCGate(Q_Gates::R_k(k), j, i);
            }
        }

        for (size_t i = 0; i < N / 2; ++i) {
            swapQubits(i, N - i - 1);
        }
    }

    // Exact inverse of applyQFT: the same gates in reverse order, each replaced by its adjoint
    // (H is its own adjoint).
    void applyIQFT() {
        const size_t N = n_;
        for (size_t i = 0; i < N / 2; ++i) {
            swapQubits(i, N - i - 1);
        }

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < i; ++j) {
                size_t k = i - j + 1;
                applyCGate(Q_Gates::R_k(k).adjoint(), j, i);
            }
            applyGate(Q_Gates::H, i);
        }
    }

    // Apply a full 2^n x 2^n matrix to the whole register.
    inline void operator* (const MatrixXcd& U) {
        if (size_t(U.rows()) != dimension() || size_t(U.cols()) != dimension()) {
            throw DimensionMismatchError();
        }
        state = U * state;
    }

    inline VectorXcd state_vector() const { return state; }

    // Full 2^n x 2^n density matrix |psi><psi| (memory 4^n complex numbers: small registers only).
    inline MatrixXcd density_matrix() const {
        return state * state.adjoint();
    }

    // ---- Measurement -------------------------------------------------------------------------

    // General measurement (Nielsen & Chuang, section 2.2.3) on the qubits listed in targets.
    // M = {M_m}: 2^k x 2^k operators (bit j of the index is qubit targets[j]) with sum_m M_m^dagger M_m = I;
    // the register sees M_m tensor I on the other qubits.
    // Outcome m occurs with probability p(m) = || (M_m tensor I) psi ||^2,
    // and the state becomes (M_m tensor I) psi / sqrt(p(m)). Returns m.
    size_t measureOn(const std::vector<MatrixXcd>& M, const std::vector<size_t>& targets, std::mt19937_64& rng,
                     double tol = 1e-10) {
        if (M.empty()) {
            throw QuantumError("measure: no measurement operators given.");
        }
        const size_t d = size_t(1) << targets.size();
        MatrixXcd completeness = MatrixXcd::Zero(d, d);
        for (const auto& Mm : M) {
            if (size_t(Mm.rows()) != d || size_t(Mm.cols()) != d) {
                throw DimensionMismatchError();
            }
            completeness += Mm.adjoint() * Mm;
        }
        if ((completeness - MatrixXcd::Identity(d, d)).cwiseAbs().maxCoeff() > tol) {
            throw QuantumError("measure: operators violate sum_m M_m^dagger M_m = I.");
        }

        const VectorXcd psi = state;
        std::vector<VectorXcd> branch(M.size());
        std::vector<double> p(M.size());
        for (size_t m = 0; m < M.size(); ++m) {
            state = psi;
            apply(M[m], targets, {});   // state = (M_m tensor I) psi
            branch[m] = state;
            p[m] = state.squaredNorm();
        }

        std::discrete_distribution<size_t> pick(p.begin(), p.end());   // P(m) = p[m] / sum p
        const size_t outcome = pick(rng);
        state = branch[outcome] / std::sqrt(p[outcome]);
        return outcome;
    }

    // One qubit, 2x2 operators.
    size_t measure(const std::vector<Matrix<Complex, 2, 2>>& M, const size_t& qubit_index, std::mt19937_64& rng,
                   double tol = 1e-10) {
        return measureOn(std::vector<MatrixXcd>(M.begin(), M.end()), { qubit_index }, rng, tol);
    }

    // Measurement of one qubit in the computational basis: M = {|0><0|, |1><1|}. Returns 0 or 1.
    size_t measureQubit(const size_t& qubit_index, std::mt19937_64& rng) {
        Matrix<Complex, 2, 2> P0 = Matrix<Complex, 2, 2>::Zero(), P1 = Matrix<Complex, 2, 2>::Zero();
        P0(0, 0) = 1;
        P1(1, 1) = 1;
        return measure({ P0, P1 }, qubit_index, rng);
    }

private:
    size_t n_;
    VectorXcd state;

    static size_t checked_size(const size_t& n) {
        if (n == 0 || n >= 8 * sizeof(size_t) - 1) {
            throw std::invalid_argument("Qubits: number of qubits must be between 1 and 62.");
        }
        return n;
    }

    static size_t log2_exact(const size_t& D) {
        size_t n = 0;
        while ((size_t(1) << n) < D) ++n;
        if (D < 2 || (size_t(1) << n) != D) {
            throw std::invalid_argument("Qubits: the amplitude vector's length must be a power of 2 (at least 2).");
        }
        return n;
    }

    void check_qubit(const size_t& i) const {
        if (i >= n_) {
            throw std::invalid_argument("Qubits: qubit index out of range.");
        }
    }

    void same_size(const Qubits& other) const {
        if (other.n_ != n_) {
            throw DimensionMismatchError();
        }
    }

    static std::vector<size_t> range(const size_t& start, const size_t& end) {
        if (end < start) {
            throw std::invalid_argument("Qubits: empty qubit range.");
        }
        std::vector<size_t> r;
        for (size_t q = start; q <= end; ++q) r.push_back(q);
        return r;
    }

    // Core of every gate: apply the 2^k x 2^k matrix U to the k qubits in targets, on the part of the
    // state where every qubit in controls is 1. For each index whose target bits are all 0 (and whose
    // control bits are all 1), gather the 2^k amplitudes of its block, multiply by U, write them back.
    void apply(const MatrixXcd& U, const std::vector<size_t>& targets, const std::vector<size_t>& controls) {
        const size_t k = targets.size();
        if (k == 0) {
            throw std::invalid_argument("Qubits: a gate needs at least one target qubit.");
        }
        size_t target_mask = 0, control_mask = 0;
        for (size_t q : targets) {
            check_qubit(q);
            if (target_mask & (size_t(1) << q)) throw std::invalid_argument("Qubits: repeated target qubit.");
            target_mask |= size_t(1) << q;
        }
        for (size_t q : controls) {
            check_qubit(q);
            if (target_mask & (size_t(1) << q)) throw std::invalid_argument("Qubits: a control qubit is also a target.");
            control_mask |= size_t(1) << q;
        }
        const size_t d = size_t(1) << k;
        if (size_t(U.rows()) != d || size_t(U.cols()) != d) {
            throw DimensionMismatchError();
        }

        // offset[l]: the index bits that local index l sets (bit j of l goes to qubit targets[j]).
        std::vector<size_t> offset(d, 0);
        for (size_t l = 0; l < d; ++l) {
            for (size_t j = 0; j < k; ++j) {
                if ((l >> j) & 1) offset[l] |= size_t(1) << targets[j];
            }
        }

        VectorXcd local(d), image(d);                                  // buffers reused for every block
        for (size_t base = 0; base < dimension(); ++base) {
            if (base & target_mask) continue;                          // visit each block once
            if ((base & control_mask) != control_mask) continue;       // every control qubit must be 1
            for (size_t l = 0; l < d; ++l) local(l) = state(base | offset[l]);
            image.noalias() = U * local;
            for (size_t l = 0; l < d; ++l) state(base | offset[l]) = image(l);
        }
    }
};

// Temporary change of basis, undone automatically (RAII).
// Creating a BasisChange checks that U is unitary and rewrites the listed qubits in the basis given by
// U's columns: the coordinates become U^dagger psi. When the BasisChange goes out of scope (the end of
// the enclosing { } block, or an exception leaving it), U is applied again and the original coordinates
// come back. Nested changes are undone in reverse order of creation. The register must outlive the guard.
class BasisChange {
public:
    BasisChange(Qubits& reg, const MatrixXcd& U, const std::vector<size_t>& targets, double tol = 1e-10) {
        if (!q_utils::is_unitary(U, tol)) {
            throw QuantumError("BasisChange: the matrix is not unitary.");
        }
        reg.applyOn(U.adjoint(), targets);
        Qubits* r = &reg;
        MatrixXcd Ucopy = U;
        std::vector<size_t> t = targets;
        undo_ = [r, Ucopy, t]() { r->applyOn(Ucopy, t); };
    }

    BasisChange(Qubit& q, const Matrix<Complex, 2, 2>& U, double tol = 1e-10) {
        if (!q_utils::is_unitary(U, tol)) {
            throw QuantumError("BasisChange: the matrix is not unitary.");
        }
        q * Matrix<Complex, 2, 2>(U.adjoint());
        Qubit* p = &q;
        Matrix<Complex, 2, 2> Ucopy = U;
        undo_ = [p, Ucopy]() { (*p) * Ucopy; };
    }

    ~BasisChange() { undo_(); }

    BasisChange(const BasisChange&) = delete;
    BasisChange& operator=(const BasisChange&) = delete;
    BasisChange(BasisChange&&) = delete;
    BasisChange& operator=(BasisChange&&) = delete;

private:
    std::function<void()> undo_;
};

inline void test_1() {

    // Define the CNOT gate as a 4x4 matrix
    Eigen::Matrix<Complex, 4, 4> CNOT = Eigen::Matrix<Complex, 4, 4>::Zero();
    CNOT(0, 0) = 1;
    CNOT(1, 1) = 1;
    CNOT(2, 3) = 1;
    CNOT(3, 2) = 1;

    // Initial state |01> (2-qubit system, so state vector has size 4)
    Eigen::Vector4cd initial_state;
    initial_state << 0, 0, 0, 1; // |11> state

    // Create Qubits object with initial state
    Qubits qubits(initial_state);

    // Define a 2x2 identity gate (no-op)
    Matrix2cd identity_gate = Matrix2cd::Identity();

    // Apply the CNOT gate using applyCGate
    qubits.applyCGate(identity_gate, 1, 0 );

    // Output the resulting state
    std::cout << "State after applying controlled gate:\n" << qubits.state_vector() << std::endl; //should be |10>

}



