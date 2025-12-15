cmake_minimum_required(VERSION 3.10)
project(ppn LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

# --- Options (profiling / blocked matmul) ---
option(ENABLE_PROFILE_MATMUL "Print timing for matmul (PROFILE_MATMUL)" OFF)
set(MATMUL_BLOCK_SIZE "" CACHE STRING "Block size for blocked matmul (e.g. 64,128). Empty = default in code")

if(ENABLE_PROFILE_MATMUL)
    add_compile_definitions(PROFILE_MATMUL)
endif()

if(NOT MATMUL_BLOCK_SIZE STREQUAL "")
    add_compile_definitions(BLOCK_SIZE=${MATMUL_BLOCK_SIZE})
endif()
# -------------------------------------------

find_path(OPENBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATH_SUFFIXES openblas
)
find_library(OPENBLAS_LIB
    NAMES openblaso openblas openblas64
)

if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS not found: set OPENBLAS_INCLUDE_DIR and OPENBLAS_LIB")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    link_libraries(OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found")
endif()

add_compile_definitions(USE_OPENBLAS)
include_directories(${OPENBLAS_INCLUDE_DIR})

# Inclure les headers
include_directories(include)
include_directories(mnist_reader)

# MNIST Reader Library
add_library(mnist_reader_lib STATIC mnist_reader/mnist_reader.c)

# Sources
file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "src/main.cpp$")

# Exécutables
add_executable(test_matrix tests/test_matrix.cpp ${SOURCES})
add_executable(test_activations tests/test_activations.cpp ${SOURCES})
add_executable(test_linear tests/test_linear.cpp ${SOURCES})
add_executable(test_mlp tests/test_mlp.cpp ${SOURCES})

enable_testing()

add_executable(test_autodiff tests/test_autodiff.cpp ${SOURCES})
add_executable(test_gradient_check tests/test_gradient_check.cpp ${SOURCES})

add_executable(test_dataloader tests/test_dataloader.cpp ${SOURCES})
add_executable(test_optimizer tests/test_optimizer.cpp ${SOURCES})

# Main Application
add_executable(ppn_train src/main.cpp ${SOURCES})
target_link_libraries(ppn_train PRIVATE ${OPENBLAS_LIB} mnist_reader_lib)

set(TEST_TARGETS
    test_matrix
    test_activations
    test_linear
    test_mlp
    test_autodiff
    test_gradient_check
    test_dataloader
    test_optimizer
)

foreach(target ${TEST_TARGETS})
    target_link_libraries(${target} PRIVATE ${OPENBLAS_LIB} mnist_reader_lib)
endforeach()

add_test(NAME MatrixTest COMMAND test_matrix)
add_test(NAME ActivationsTest COMMAND test_activations)
add_test(NAME LinearTest COMMAND test_linear)
add_test(NAME MLPTest COMMAND test_mlp)
add_test(NAME AutodiffTest COMMAND test_autodiff)
add_test(NAME GradientCheckTest COMMAND test_gradient_check)
add_test(NAME DataLoaderTest COMMAND test_dataloader)
add_test(NAME OptimizerTest COMMAND test_optimizer)

add_executable(test_benchmark_large tests/test_benchmark_large.cpp ${SOURCES})
target_link_libraries(test_benchmark_large PUBLIC ${OPENBLAS_LIB} mnist_reader_lib)

