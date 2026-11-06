#pragma once
#include <cstdint>
#include <bit>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>

struct PauliString{
    std::size_t n=0;
    std::uint64_t x=0;
    std::uint64_t z=0;
    unsigned k =0;

    static PauliString from_string(const std::string& s){
        if (s.size() > 64) throw std::invalid_argument("PauliString: at most 64 qubits.");
        PauliString p;
        p.n = s.size();
        for(std::size_t j=0; j< s.size(); ++j){
            const std::uint64_t bit = std::uint64_t(1) << j;
            switch (s[j]) {
                case 'I': break;
                case 'X': p.x |= bit; break;
                case 'Z': p.z |= bit; break;
                case 'Y': p.x |= bit; p.z |= bit; p.k += 1; break;   // Y = i X Z
                default: throw std::invalid_argument("PauliString: use only I, X, Y, Z.");
            }
        }
        p.k %= 4;
        return p;
    }
};
    // Product  P Q.  From  Z X = -X Z  on each qubit:
    //   (X^{x1} Z^{z1}) (X^{x2} Z^{z2}) = (-1)^{|z1 & x2|} X^{x1 ^ x2} Z^{z1 ^ z2},
    // where |z1 & x2| counts the qubits on which P's Z must pass Q's X.
    // So the phases add, plus 2 (that is, a factor -1 = i^2) for each such crossing.
    inline PauliString operator*(const PauliString& P, const PauliString& Q) {
        if (P.n != Q.n) throw std::invalid_argument("PauliString: different numbers of qubits.");
        PauliString R;
        R.n = P.n;
        R.x = P.x ^ Q.x;
        R.z = P.z ^ Q.z;
        R.k = (P.k + Q.k + 2u * unsigned(std::popcount(P.z & Q.x))) % 4u;
    return R;
    }
    
    // P and Q commute exactly when the symplectic product  |x_P & z_Q| + |z_P & x_Q|  is even;
    // otherwise they anticommute (P Q = -Q P).
    inline bool commute(const PauliString& P, const PauliString& Q) {
        return (std::popcount(P.x & Q.z) + std::popcount(P.z & Q.x)) % 2 == 0;
    }
    
    // P is Hermitian exactly when k has the parity of |x & z|  (from (X^x Z^z)^dagger = (-1)^{|x & z|} X^x Z^z).
    inline bool is_hermitian(const PauliString& P) {
     return P.k % 2 == unsigned(std::popcount(P.x & P.z)) % 2;
    }

    // Dense 2^n x 2^n matrix, for checks only: entry (row, col) of i^k X^x Z^z.
    // Z^z multiplies basis state |col> by (-1)^{|z & col|}; X^x then sends it to |col ^ x>.
    inline Eigen::MatrixXcd matrix(const PauliString& P) {
        const std::size_t D = std::size_t(1) << P.n;
        const std::complex<double> ik[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
     Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(D, D);
      for (std::size_t col = 0; col < D; ++col) {
            const double sign = (std::popcount(P.z & col) % 2) ? -1.0 : 1.0;
            M(col ^ P.x, col) = ik[P.k] * sign;
        }
        return M;
    }

