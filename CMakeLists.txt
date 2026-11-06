cmake_minimum_required(VERSION 3.16)
project(GEO_Q LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimised build unless another type is asked for (cmake -DCMAKE_BUILD_TYPE=Debug ..).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Eigen is header-only. It must be installed on the machine (it is not downloaded).
find_package(Eigen3 3.3 NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR
    "Eigen 3 was not found. Install it (Ubuntu: libeigen3-dev; Nix: the 'eigen' package), "
    "or point CMake at it with -DEigen3_DIR=<folder containing Eigen3Config.cmake>.")
endif()

# Basics.h includes <omp.h>; link OpenMP when the compiler provides it.
find_package(OpenMP)

# The simulator and the Cobble code are headers in GEO_Q/; this target carries their settings.
add_library(geoq INTERFACE)
target_include_directories(geoq INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/GEO_Q)
target_link_libraries(geoq INTERFACE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(geoq INTERFACE OpenMP::OpenMP_CXX)
endif()

# The existing program (main calls test_1()).
add_executable(geoq_main GEO_Q/ConsoleApplication1.cpp)
target_link_libraries(geoq_main PRIVATE geoq)

# Tests (doctest, single header in third_party/doctest). Run with: ctest --output-on-failure
enable_testing()
add_executable(geoq_tests tests/test_basics.cpp)
target_include_directories(geoq_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_tests PRIVATE geoq)
add_test(NAME geoq_tests COMMAND geoq_tests)

# Known-answer tests for pauli.h (textbook results, Nielsen & Chuang).
add_executable(geoq_kat tests/kat_pauli.cpp)
target_include_directories(geoq_kat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_kat PRIVATE geoq)
add_test(NAME geoq_kat COMMAND geoq_kat)

# Checks for the Cobble term builders (GEO_Q.h).
add_executable(geoq_term tests/test_term.cpp)
target_include_directories(geoq_term PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_term PRIVATE geoq)
add_test(NAME geoq_term COMMAND geoq_term)

# Maxima of polynomials on [-1, 1] and on the unit circle (alpha of QSVT and GQET).
add_executable(geoq_polymax tests/test_polymax.cpp)
target_include_directories(geoq_polymax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_polymax PRIVATE geoq)
add_test(NAME geoq_polymax COMMAND geoq_polymax)

# Costs and Hermiticity, and the meaning of terms on vectors (GEO_Q.h).
foreach(t cost meaning)
  add_executable(geoq_${t} tests/test_${t}.cpp)
  target_include_directories(geoq_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
  target_link_libraries(geoq_${t} PRIVATE geoq)
  add_test(NAME geoq_${t} COMMAND geoq_${t})
endforeach()

# Cobble's benchmarks rebuilt: known answers from Cobble's tables, and a cost report.
add_executable(geoq_benchmarks tests/test_benchmarks.cpp)
target_include_directories(geoq_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_benchmarks PRIVATE geoq)
add_test(NAME geoq_benchmarks COMMAND geoq_benchmarks)
add_executable(geoq_report bench/report.cpp)
target_link_libraries(geoq_report PRIVATE geoq)

# The symbol route (first numerical version of the hunch) and its report.
add_executable(geoq_symbol tests/test_symbol.cpp)
target_include_directories(geoq_symbol PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_symbol PRIVATE geoq)
add_test(NAME geoq_symbol COMMAND geoq_symbol)
add_executable(geoq_symbol_report bench/symbol_report.cpp)
target_link_libraries(geoq_symbol_report PRIVATE geoq)

# Error column, truncated Chebyshev method, heat-diffusion benchmark and its report.
add_executable(geoq_heat tests/test_heat.cpp)
target_include_directories(geoq_heat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_heat PRIVATE geoq)
add_test(NAME geoq_heat COMMAND geoq_heat)
add_executable(geoq_heat_report bench/heat_report.cpp)
target_link_libraries(geoq_heat_report PRIVATE geoq)

# Structured-operator suite: Pauli form against the symbol route.
add_executable(geoq_suite_report bench/suite_report.cpp)
target_link_libraries(geoq_suite_report PRIVATE geoq)

# From the literature (5 Nov, evening): certificate, moment order, boundary route, Toffoli estimate.
add_executable(geoq_literature tests/test_literature.cpp)
target_include_directories(geoq_literature PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_literature PRIVATE geoq)
add_test(NAME geoq_literature COMMAND geoq_literature)
add_executable(geoq_literature_report bench/literature_report.cpp)
target_link_libraries(geoq_literature_report PRIVATE geoq)

# Code generation (6 Nov): terms -> gate circuits run on the simulator; rotations, expectations, gradients.
add_executable(geoq_compile tests/test_compile.cpp)
target_include_directories(geoq_compile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_compile PRIVATE geoq)
add_test(NAME geoq_compile COMMAND geoq_compile)
add_executable(geoq_compile_report bench/compile_report.cpp)
target_link_libraries(geoq_compile_report PRIVATE geoq)

# Cobble's rewrite optimiser (6 Nov): known answers from Cobble's Table 3.
add_executable(geoq_rewrite tests/test_rewrite.cpp)
target_include_directories(geoq_rewrite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(geoq_rewrite PRIVATE geoq)
add_test(NAME geoq_rewrite COMMAND geoq_rewrite)
