cmake_minimum_required(VERSION 3.10)
project(ppn LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# --- Options (profiling / blocked matmul) ---
option(ENABLE_GPROF "Enable gprof instrumentation (-pg) for profiling builds" OFF)
option(ENABLE_PROFILE_MATMUL "Print timing for matmul (PROFILE_MATMUL)" OFF)
option(ENABLE_MPI "Enable MPI distributed data parallel" OFF)
set(MATMUL_BLOCK_SIZE "" CACHE STRING "Block size for blocked matmul (e.g. 64,128). Empty = default in code")

if(ENABLE_GPROF)
    message(STATUS "gprof instrumentation enabled (-pg)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

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
find_library(JPEG_LIB
    NAMES jpeg
)

if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS not found: set OPENBLAS_INCLUDE_DIR and OPENBLAS_LIB")
endif()
if(NOT JPEG_LIB)
    message(FATAL_ERROR "libjpeg not found")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    link_libraries(OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found")
endif()

if(ENABLE_MPI)
    find_package(MPI REQUIRED)
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
file(GLOB AUTOGRAD_SOURCES "src/autograd/*.cpp")
file(GLOB DISTRIBUTED_SOURCES "src/distributed/*.cpp")
list(APPEND SOURCES ${AUTOGRAD_SOURCES} ${DISTRIBUTED_SOURCES})
list(FILTER SOURCES EXCLUDE REGEX "src/main.cpp$")

# Exécutables
add_executable(test_matrix tests/test_matrix.cpp ${SOURCES})
add_executable(test_activations tests/test_activations.cpp ${SOURCES})
add_executable(test_linear tests/test_linear.cpp ${SOURCES})
add_executable(test_mlp tests/test_mlp.cpp ${SOURCES})

enable_testing()

add_executable(test_autodiff tests/test_autodiff.cpp ${SOURCES})
add_executable(test_gradient_check tests/test_gradient_check.cpp ${SOURCES})
add_executable(test_autograd_engine tests/test_autograd_engine.cpp ${SOURCES})
add_executable(test_cnn_autograd tests/test_cnn_autograd.cpp ${SOURCES})

add_executable(test_dataloader tests/test_dataloader.cpp ${SOURCES})
add_executable(test_optimizer tests/test_optimizer.cpp ${SOURCES})
add_executable(test_tiny_imagenet_dataset tests/test_tiny_imagenet_dataset.cpp ${SOURCES})

# Main Application
add_executable(ppn_train src/main.cpp ${SOURCES})
target_link_libraries(ppn_train PRIVATE ${OPENBLAS_LIB} mnist_reader_lib)
target_link_libraries(ppn_train PRIVATE ${JPEG_LIB})
if(ENABLE_MPI)
    target_compile_definitions(ppn_train PRIVATE USE_MPI)
    target_link_libraries(ppn_train PRIVATE MPI::MPI_CXX)
endif()

set(TEST_TARGETS
    test_matrix
    test_activations
    test_linear
    test_mlp
    test_autodiff
    test_gradient_check
    test_autograd_engine
    test_cnn_autograd
    test_dataloader
    test_optimizer
    test_tiny_imagenet_dataset
)

foreach(target ${TEST_TARGETS})
    target_link_libraries(${target} PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB})
endforeach()

add_test(NAME MatrixTest COMMAND test_matrix)
add_test(NAME ActivationsTest COMMAND test_activations)
add_test(NAME LinearTest COMMAND test_linear)
add_test(NAME MLPTest COMMAND test_mlp)
add_test(NAME AutodiffTest COMMAND test_autodiff)
add_test(NAME GradientCheckTest COMMAND test_gradient_check)
add_test(NAME AutogradEngineTest COMMAND test_autograd_engine)
add_test(NAME CNNAutogradTest COMMAND test_cnn_autograd)
add_test(NAME DataLoaderTest COMMAND test_dataloader WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME OptimizerTest COMMAND test_optimizer)
add_test(NAME TinyImageNetDatasetTest COMMAND test_tiny_imagenet_dataset)

add_executable(test_benchmark_large tests/test_benchmark_large.cpp ${SOURCES})
target_link_libraries(test_benchmark_large PUBLIC ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB})


add_executable(test_init tests/test_init.cpp ${SOURCES})
target_link_libraries(test_init PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB})
add_test(NAME InitTest COMMAND test_init)

add_executable(test_cnn_configurable tests/test_cnn_configurable.cpp ${SOURCES})
target_link_libraries(test_cnn_configurable PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB})
add_test(NAME CNNConfigurableTest COMMAND test_cnn_configurable)

# CLI regression tests for argument validation (must fail fast before data loading).
add_test(NAME CLIInvalidModelTest
    COMMAND ppn_train --model not_a_model --epochs 0)
set_tests_properties(CLIInvalidModelTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLIUnknownArgumentTest
    COMMAND ppn_train --cnn_typo 1)
set_tests_properties(CLIUnknownArgumentTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLICNNBoolParseTest
    COMMAND ppn_train --model cnn --cnn_conv_channels 8 --cnn_pool_after 2 --epochs 0)
set_tests_properties(CLICNNBoolParseTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLINonIntegerEpochTest
    COMMAND ppn_train --epochs abc)
set_tests_properties(CLINonIntegerEpochTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLIBadHiddenTokenTest
    COMMAND ppn_train --hidden_sizes 64,128x --epochs 0)
set_tests_properties(CLIBadHiddenTokenTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLINonFiniteLearningRateTest
    COMMAND ppn_train --learning_rate nan --epochs 0)
set_tests_properties(CLINonFiniteLearningRateTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLIBatchSizeZeroTest
    COMMAND ppn_train --batch_size 0 --epochs 0)
set_tests_properties(CLIBatchSizeZeroTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLIInvalidDatasetTest
    COMMAND ppn_train --dataset not_a_dataset --epochs 0)
set_tests_properties(CLIInvalidDatasetTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLISeedNegativeTest
    COMMAND ppn_train --seed -1 --epochs 0)
set_tests_properties(CLISeedNegativeTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLIInvalidOptimizerTest
    COMMAND ppn_train --optimizer not_a_optimizer --epochs 0)
set_tests_properties(CLIInvalidOptimizerTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLIBadMomentumRangeTest
    COMMAND ppn_train --optimizer momentum_sgd --momentum 1.0 --epochs 0)
set_tests_properties(CLIBadMomentumRangeTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLIBadBeta1RangeTest
    COMMAND ppn_train --optimizer adamw --beta1 1.0 --epochs 0)
set_tests_properties(CLIBadBeta1RangeTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLIBadBeta2RangeTest
    COMMAND ppn_train --optimizer adamw --beta2 1.0 --epochs 0)
set_tests_properties(CLIBadBeta2RangeTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLINonPositiveEpsTest
    COMMAND ppn_train --optimizer adamw --eps 0 --epochs 0)
set_tests_properties(CLINonPositiveEpsTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLINegativeWeightDecayTest
    COMMAND ppn_train --weight_decay -0.1 --epochs 0)
set_tests_properties(CLINegativeWeightDecayTest PROPERTIES
    WILL_FAIL TRUE)

# Positive CLI smoke test for custom CNN config.
# Only register when MNIST files are available in source tree.
if(EXISTS "${CMAKE_SOURCE_DIR}/mnist/train-images-idx3-ubyte" AND
   EXISTS "${CMAKE_SOURCE_DIR}/mnist/train-labels-idx1-ubyte" AND
   EXISTS "${CMAKE_SOURCE_DIR}/mnist/t10k-images-idx3-ubyte" AND
   EXISTS "${CMAKE_SOURCE_DIR}/mnist/t10k-labels-idx1-ubyte")
    add_test(NAME CLICNNSmokeTest
        COMMAND ppn_train
            --model cnn
            --cnn_conv_channels 6,16
            --cnn_pool_after 1,1
            --cnn_fc_hidden_sizes 64
            --epochs 0
            --batch_size 10000
            --data_dir mnist)
    set_tests_properties(CLICNNSmokeTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        PASS_REGULAR_EXPRESSION "Building custom CNN")
else()
    message(STATUS "Skipping CLICNNSmokeTest: MNIST files not found in ${CMAKE_SOURCE_DIR}/mnist")
endif()
