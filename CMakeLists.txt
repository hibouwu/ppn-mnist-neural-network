cmake_minimum_required(VERSION 3.10)
project(ppn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_path(OPENBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATH_SUFFIXES openblas
)
find_library(OPENBLAS_LIB
    NAMES openblas openblas64
)

if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS not found: set OPENBLAS_INCLUDE_DIR and OPENBLAS_LIB")
endif()

add_compile_definitions(USE_OPENBLAS)
include_directories(${OPENBLAS_INCLUDE_DIR})

# Inclure les headers
include_directories(include)

# Sources
file(GLOB SOURCES "src/*.cpp")

# Exécutables
add_executable(test_matrix tests/test_matrix.cpp ${SOURCES})
add_executable(test_activations tests/test_activations.cpp ${SOURCES})
add_executable(test_linear tests/test_linear.cpp ${SOURCES})
add_executable(test_mlp tests/test_mlp.cpp ${SOURCES})

enable_testing()

add_executable(test_autodiff tests/test_autodiff.cpp ${SOURCES})
add_executable(test_gradient_check tests/test_gradient_check.cpp ${SOURCES})

set(TEST_TARGETS
    test_matrix
    test_activations
    test_linear
    test_mlp
    test_autodiff
    test_gradient_check
)

foreach(target ${TEST_TARGETS})
    target_link_libraries(${target} PRIVATE ${OPENBLAS_LIB})
endforeach()

add_test(NAME MatrixTest COMMAND test_matrix)
add_test(NAME ActivationsTest COMMAND test_activations)
add_test(NAME LinearTest COMMAND test_linear)
add_test(NAME MLPTest COMMAND test_mlp)
add_test(NAME AutodiffTest COMMAND test_autodiff)
add_test(NAME GradientCheckTest COMMAND test_gradient_check)
