cmake_minimum_required(VERSION 3.10)
project(ppn LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# --- Options (profiling / blocked matmul) ---
option(ENABLE_GPROF "Enable gprof instrumentation (-pg) for profiling builds" OFF)
option(ENABLE_PROFILE_MATMUL "Print timing for matmul (PROFILE_MATMUL)" OFF)
option(ENABLE_PROFILE_OPS "Print timing for selected forward/backward ops (PROFILE_OPS)" OFF)
option(ENABLE_VTUNE_MARKERS "Enable Intel VTune ITT API markers when available" ON)
option(ENABLE_MPI "Enable MPI distributed data parallel" OFF)
option(ENABLE_ONEDNN_PARITY_PROBE "Enable isolated oneDNN conv parity probe target" ON)
option(ENABLE_ONEDNN_CONV_BACKEND "Enable optional oneDNN Conv2D backend in selected targets" OFF)
option(ENABLE_GPU_SMOKE_TESTS "Enable standalone CUDA runtime smoke test targets" OFF)
set(MATMUL_BLOCK_SIZE "" CACHE STRING "Block size for blocked matmul (e.g. 64,128). Empty = default in code")

if(ENABLE_GPROF)
    message(STATUS "gprof instrumentation enabled (-pg)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

if(ENABLE_PROFILE_MATMUL)
    add_compile_definitions(PROFILE_MATMUL)
endif()

if(ENABLE_PROFILE_OPS)
    add_compile_definitions(PROFILE_OPS)
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
    message(WARNING "libjpeg not found: Tiny ImageNet support disabled")
    set(JPEG_LIB "")
    set(PPN_HAVE_JPEG 0)
else()
    set(PPN_HAVE_JPEG 1)
endif()

set(PPN_HAVE_VTUNE 0)
if(ENABLE_VTUNE_MARKERS)
    set(VTUNE_SEARCH_HINTS
        ${VTUNE_PROFILER_DIR}
        $ENV{VTUNE_PROFILER_DIR}
        ${ONEAPI_ROOT}
        $ENV{ONEAPI_ROOT}
        /opt/intel/oneapi/vtune/latest
        /opt/intel/oneapi/vtune
        /opt/intel/oneapi
        /usr/local
        /usr
    )
    find_path(ITTNOTIFY_INCLUDE_DIR
        NAMES ittnotify.h
        HINTS ${VTUNE_SEARCH_HINTS}
        PATH_SUFFIXES
            include
            include/ittnotify
            sdk/include
    )
    find_library(ITTNOTIFY_LIB
        NAMES ittnotify libittnotify
        HINTS ${VTUNE_SEARCH_HINTS}
        PATH_SUFFIXES
            lib
            lib64
            sdk/lib
            sdk/lib64
    )

    if(ITTNOTIFY_INCLUDE_DIR AND ITTNOTIFY_LIB)
        set(PPN_HAVE_VTUNE 1)
        message(STATUS "Intel VTune ITT API found; task markers enabled")
    else()
        message(STATUS "Intel VTune ITT API not found; building without VTune markers")
    endif()
else()
    message(STATUS "Intel VTune markers explicitly disabled")
endif()

if(ENABLE_ONEDNN_PARITY_PROBE OR ENABLE_ONEDNN_CONV_BACKEND)
    set(DNNL_SEARCH_HINTS
        ${DNNL_ROOT}
        $ENV{DNNL_ROOT}
        ${CMAKE_PREFIX_PATH}
        /opt/onednn-cpu
        /opt/intel/oneapi/dnnl/2025.3
        /opt/intel/oneapi/dnnl/2025.2
        /opt/intel/oneapi/2025.3
        /opt/intel/oneapi/2025.2
    )
    find_path(DNNL_INCLUDE_DIR
        NAMES dnnl.hpp
        HINTS
            ${DNNL_SEARCH_HINTS}
        PATH_SUFFIXES
            include
    )
    find_library(DNNL_LIB
        NAMES dnnl
        HINTS
            ${DNNL_SEARCH_HINTS}
        PATH_SUFFIXES
            lib
            lib64
    )
    find_library(SYCL_LIB
        NAMES sycl
        HINTS
            /opt/intel/oneapi/compiler/2025.3/lib
            /opt/intel/oneapi/compiler/2025.2/lib
            /opt/intel/oneapi/2025.3/lib
            /opt/intel/oneapi/2025.2/lib
    )
    find_library(SVML_LIB
        NAMES svml
        HINTS
            /opt/intel/oneapi/compiler/2025.3/lib
            /opt/intel/oneapi/compiler/2025.2/lib
            /opt/intel/oneapi/2025.3/lib
            /opt/intel/oneapi/2025.2/lib
    )
    find_library(IMF_LIB
        NAMES imf
        HINTS
            /opt/intel/oneapi/compiler/2025.3/lib
            /opt/intel/oneapi/compiler/2025.2/lib
            /opt/intel/oneapi/2025.3/lib
            /opt/intel/oneapi/2025.2/lib
    )
    find_library(INTLC_LIB
        NAMES intlc intlc.so.5
        HINTS
            /opt/intel/oneapi/compiler/2025.3/lib
            /opt/intel/oneapi/compiler/2025.2/lib
            /opt/intel/oneapi/2025.3/lib
            /opt/intel/oneapi/2025.2/lib
    )
    find_library(TBB_LIB
        NAMES tbb
        HINTS
            /opt/intel/oneapi/tbb/2022.2/lib/intel64/gcc4.8
            /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8
            /opt/intel/oneapi/tbb/2022.2/lib
            /opt/intel/oneapi/tbb/latest/lib
    )
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
add_compile_definitions(PPN_HAVE_JPEG=${PPN_HAVE_JPEG})
include_directories(${OPENBLAS_INCLUDE_DIR})

# Inclure les headers
include_directories(include)
include_directories(mnist_reader)

# MNIST Reader Library
add_library(mnist_reader_lib STATIC mnist_reader/mnist_reader.c)
add_library(ppn_vtune INTERFACE)
target_compile_definitions(ppn_vtune INTERFACE PPN_HAVE_VTUNE=${PPN_HAVE_VTUNE})
if(ENABLE_VTUNE_MARKERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ppn_vtune INTERFACE -fno-omit-frame-pointer)
    target_link_options(ppn_vtune INTERFACE -fno-omit-frame-pointer)
endif()
if(PPN_HAVE_VTUNE)
    target_include_directories(ppn_vtune INTERFACE ${ITTNOTIFY_INCLUDE_DIR})
    target_link_libraries(ppn_vtune INTERFACE ${ITTNOTIFY_LIB})
endif()

# Sources
file(GLOB SOURCES "src/*.cpp")
file(GLOB GEMM_SOURCES "src/gemm/*.cpp")
file(GLOB AUTOGRAD_SOURCES "src/autograd/*.cpp")
file(GLOB DISTRIBUTED_SOURCES "src/distributed/*.cpp")
file(GLOB IO_SOURCES "src/io/*.cpp")
file(GLOB RUNTIME_SOURCES "src/runtime/*.cpp")
file(GLOB IST_SOURCES "src/ist/*.cpp")
list(APPEND SOURCES ${GEMM_SOURCES} ${AUTOGRAD_SOURCES} ${DISTRIBUTED_SOURCES} ${IO_SOURCES} ${RUNTIME_SOURCES} ${IST_SOURCES})
list(FILTER SOURCES EXCLUDE REGEX "src/main.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "src/gpu/cudnn_conv_forward.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "src/gpu/cudnn_conv_backward.cpp$")
if(NOT PPN_HAVE_JPEG)
    list(FILTER SOURCES EXCLUDE REGEX "src/tiny_imagenet_dataset.cpp$")
endif()

# Exécutables
add_executable(test_matrix tests/test_matrix.cpp ${SOURCES})
add_executable(test_gemm_microkernels tests/test_gemm_microkernels.cpp ${SOURCES})
add_executable(test_gemm_gotoblas_driver tests/test_gemm_gotoblas_driver.cpp ${SOURCES})
add_executable(test_activations tests/test_activations.cpp ${SOURCES})
add_executable(test_linear tests/test_linear.cpp ${SOURCES})
add_executable(test_mlp tests/test_mlp.cpp ${SOURCES})

enable_testing()

add_executable(test_autodiff tests/test_autodiff.cpp ${SOURCES})
add_executable(test_gradient_check tests/test_gradient_check.cpp ${SOURCES})
add_executable(test_autograd_engine tests/test_autograd_engine.cpp ${SOURCES})
add_executable(test_cnn_autograd tests/test_cnn_autograd.cpp ${SOURCES})
add_executable(test_resnet_network tests/test_resnet_network.cpp ${SOURCES})
add_executable(test_distributed_sync tests/test_distributed_sync.cpp ${SOURCES})
add_executable(test_distributed_sync_mpi_contracts tests/test_distributed_sync.cpp ${SOURCES})

add_executable(test_dataloader tests/test_dataloader.cpp ${SOURCES})
add_executable(test_optimizer tests/test_optimizer.cpp ${SOURCES})
add_executable(test_tiny_imagenet_dataset tests/test_tiny_imagenet_dataset.cpp ${SOURCES})
add_executable(test_ist_runtime tests/test_ist_runtime.cpp ${SOURCES})

# Main Application
add_executable(ppn_train src/main.cpp ${SOURCES})
target_link_libraries(ppn_train PRIVATE ${OPENBLAS_LIB} mnist_reader_lib)
target_link_libraries(ppn_train PRIVATE ${JPEG_LIB} ppn_vtune)
if(ENABLE_MPI)
    target_compile_definitions(ppn_train PRIVATE USE_MPI)
    target_link_libraries(ppn_train PRIVATE MPI::MPI_CXX)
endif()

set(TEST_TARGETS
    test_matrix
    test_gemm_microkernels
    test_gemm_gotoblas_driver
    test_activations
    test_linear
    test_mlp
    test_autodiff
    test_gradient_check
    test_autograd_engine
    test_cnn_autograd
    test_resnet_network
    test_distributed_sync
    test_distributed_sync_mpi_contracts
    test_dataloader
    test_optimizer
    test_tiny_imagenet_dataset
    test_ist_runtime
)

foreach(target ${TEST_TARGETS})
    target_link_libraries(${target} PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ppn_vtune)
endforeach()

if(ENABLE_MPI)
    target_compile_definitions(test_distributed_sync PRIVATE USE_MPI)
    target_link_libraries(test_distributed_sync PRIVATE MPI::MPI_CXX)
    target_compile_definitions(test_distributed_sync_mpi_contracts PRIVATE USE_MPI DISTRIBUTED_MPI_CONTRACTS_ONLY)
    target_link_libraries(test_distributed_sync_mpi_contracts PRIVATE MPI::MPI_CXX)
endif()

add_test(NAME MatrixTest COMMAND test_matrix)
add_test(NAME GemmMicrokernelsTest COMMAND test_gemm_microkernels)
add_test(NAME GemmGotoBlasDriverTest COMMAND test_gemm_gotoblas_driver)
add_test(NAME GemmGotoBlasAvx512DriverTest COMMAND test_gemm_gotoblas_driver)
set_tests_properties(GemmGotoBlasAvx512DriverTest
    PROPERTIES ENVIRONMENT "TEST_GEMM_GOTOBLAS_IMPL=avx512")
add_test(NAME ActivationsTest COMMAND test_activations)
add_test(NAME LinearTest COMMAND test_linear)
add_test(NAME MLPTest COMMAND test_mlp)
add_test(NAME AutodiffTest COMMAND test_autodiff)
add_test(NAME GradientCheckTest COMMAND test_gradient_check)
add_test(NAME AutogradEngineTest COMMAND test_autograd_engine)
add_test(NAME CNNAutogradTest COMMAND test_cnn_autograd)
add_test(NAME ResNetNetworkTest COMMAND test_resnet_network)
add_test(NAME DistributedSyncTest COMMAND test_distributed_sync)
if(ENABLE_MPI)
    add_test(NAME DistributedSyncMPIContractsWorldSize2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_distributed_sync_mpi_contracts>)
endif()
add_test(NAME DataLoaderTest COMMAND test_dataloader WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME OptimizerTest COMMAND test_optimizer)
add_test(NAME TinyImageNetDatasetTest COMMAND test_tiny_imagenet_dataset)
add_test(NAME IstRuntimeTest COMMAND test_ist_runtime)

add_executable(test_benchmark_large tests/test_benchmark_large.cpp ${SOURCES})
target_link_libraries(test_benchmark_large PUBLIC ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ppn_vtune)

add_executable(test_gemm_microkernel_benchmark tests/test_gemm_microkernel_benchmark.cpp ${SOURCES})
target_link_libraries(test_gemm_microkernel_benchmark PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ppn_vtune)

add_executable(test_autograd_runtime_benchmark tests/test_autograd_runtime_benchmark.cpp ${SOURCES})
target_link_libraries(test_autograd_runtime_benchmark PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ppn_vtune)

if(ENABLE_GPU_SMOKE_TESTS)
    find_package(CUDAToolkit REQUIRED)
    add_executable(cuda_runtime_smoke_test tests/gpu/cuda_runtime_smoke_test.cpp)
    target_link_libraries(cuda_runtime_smoke_test PRIVATE CUDA::cudart)

    find_path(CUDNN_INCLUDE_DIR
        NAMES cudnn.h
        HINTS
            /usr/local/cuda-13.0/targets/x86_64-linux/include
            /usr/local/cuda/include
            /usr/include
            /usr/local/include
    )
    find_library(CUDNN_LIBRARY
        NAMES cudnn
        HINTS
            /usr/local/cuda-13.0/targets/x86_64-linux/lib
            /usr/local/cuda/lib64
            /usr/lib64
            /usr/local/lib64
            /usr/local/lib
    )

    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        add_executable(cudnn_smoke_test tests/gpu/cudnn_smoke_test.cpp)
        target_include_directories(cudnn_smoke_test PRIVATE ${CUDNN_INCLUDE_DIR})
        target_link_libraries(cudnn_smoke_test PRIVATE ${CUDNN_LIBRARY} CUDA::cudart)

        add_executable(cudnn_direct_conv_forward_probe tests/gpu/cudnn_direct_conv_forward_probe.cpp)
        target_include_directories(cudnn_direct_conv_forward_probe PRIVATE ${CUDNN_INCLUDE_DIR})
        target_link_libraries(cudnn_direct_conv_forward_probe PRIVATE ${CUDNN_LIBRARY} CUDA::cudart)

        add_executable(cudnn_conv_forward_probe
            tests/gpu/cudnn_conv_forward_probe.cpp
            src/gpu/cudnn_conv_forward.cpp)
        target_include_directories(cudnn_conv_forward_probe PRIVATE ${CUDNN_INCLUDE_DIR})
        target_link_libraries(cudnn_conv_forward_probe PRIVATE ${CUDNN_LIBRARY} CUDA::cudart)

        add_executable(cudnn_conv_backward_probe
            tests/gpu/cudnn_conv_backward_probe.cpp
            ${SOURCES}
            src/gpu/cudnn_conv_backward.cpp)
        target_include_directories(cudnn_conv_backward_probe PRIVATE ${CUDNN_INCLUDE_DIR})
        target_link_libraries(cudnn_conv_backward_probe PRIVATE
            ${OPENBLAS_LIB}
            mnist_reader_lib
            ${JPEG_LIB}
            ${CUDNN_LIBRARY}
            CUDA::cudart)

        add_executable(test_conv2d_cudnn_forward_experimental
            tests/gpu/test_conv2d_cudnn_forward_experimental.cpp
            ${SOURCES}
            src/gpu/cudnn_conv_forward.cpp)
        target_include_directories(test_conv2d_cudnn_forward_experimental PRIVATE ${CUDNN_INCLUDE_DIR})
        target_compile_definitions(test_conv2d_cudnn_forward_experimental PRIVATE
            PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD=1)
        target_link_libraries(test_conv2d_cudnn_forward_experimental PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ${CUDNN_LIBRARY} CUDA::cudart)

        add_executable(test_conv2d_cudnn_backward_experimental
            tests/gpu/test_conv2d_cudnn_backward_experimental.cpp
            ${SOURCES}
            src/gpu/cudnn_conv_forward.cpp
            src/gpu/cudnn_conv_backward.cpp)
        target_include_directories(test_conv2d_cudnn_backward_experimental PRIVATE ${CUDNN_INCLUDE_DIR})
        target_compile_definitions(test_conv2d_cudnn_backward_experimental PRIVATE
            PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD=1
            PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_BACKWARD=1)
        target_link_libraries(test_conv2d_cudnn_backward_experimental PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ${CUDNN_LIBRARY} CUDA::cudart)
    else()
        message(STATUS "cuDNN not found; skipping cudnn_smoke_test target")
    endif()
endif()

add_executable(test_init tests/test_init.cpp ${SOURCES})
target_link_libraries(test_init PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ppn_vtune)
add_test(NAME InitTest COMMAND test_init)

add_executable(test_cnn_configurable tests/test_cnn_configurable.cpp ${SOURCES})
target_link_libraries(test_cnn_configurable PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ppn_vtune)
add_test(NAME CNNConfigurableTest COMMAND test_cnn_configurable)

add_executable(test_conv2d_onednn_backend tests/test_conv2d_onednn_backend.cpp ${SOURCES})
target_link_libraries(test_conv2d_onednn_backend PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ppn_vtune)
add_test(NAME Conv2DOneDnnBackendTest COMMAND test_conv2d_onednn_backend)

add_executable(test_conv2d_backend_benchmark tests/test_conv2d_backend_benchmark.cpp ${SOURCES})
target_link_libraries(test_conv2d_backend_benchmark PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ppn_vtune)

function(configure_onednn_conv_backend_target target_name)
    if(ENABLE_ONEDNN_CONV_BACKEND AND DNNL_INCLUDE_DIR AND DNNL_LIB)
        target_compile_definitions(${target_name} PRIVATE PPN_HAVE_ONEDNN_CONV_BACKEND=1)
        target_include_directories(${target_name} PRIVATE ${DNNL_INCLUDE_DIR})
        target_link_libraries(${target_name} PRIVATE ${DNNL_LIB})
        get_filename_component(DNNL_LIB_DIR ${DNNL_LIB} DIRECTORY)
        set(ONEDNN_TARGET_RPATHS ${DNNL_LIB_DIR})
        foreach(optional_lib SYCL_LIB SVML_LIB IMF_LIB INTLC_LIB TBB_LIB)
            if(${optional_lib})
                target_link_libraries(${target_name} PRIVATE ${${optional_lib}})
                get_filename_component(optional_lib_dir ${${optional_lib}} DIRECTORY)
                list(APPEND ONEDNN_TARGET_RPATHS ${optional_lib_dir})
            endif()
        endforeach()
        list(REMOVE_DUPLICATES ONEDNN_TARGET_RPATHS)
        set_target_properties(${target_name} PROPERTIES BUILD_RPATH "${ONEDNN_TARGET_RPATHS}")
    else()
        target_compile_definitions(${target_name} PRIVATE PPN_HAVE_ONEDNN_CONV_BACKEND=0)
    endif()
endfunction()

configure_onednn_conv_backend_target(ppn_train)
configure_onednn_conv_backend_target(test_cnn_autograd)
configure_onednn_conv_backend_target(test_cnn_configurable)
configure_onednn_conv_backend_target(test_conv2d_onednn_backend)
configure_onednn_conv_backend_target(test_conv2d_backend_benchmark)

if(ENABLE_ONEDNN_PARITY_PROBE)
    add_executable(test_onednn_conv_parity_probe tests/test_onednn_conv_parity_probe.cpp ${SOURCES})
    target_link_libraries(test_onednn_conv_parity_probe PRIVATE ${OPENBLAS_LIB} mnist_reader_lib ${JPEG_LIB} ppn_vtune)

    if(DNNL_INCLUDE_DIR AND DNNL_LIB)
        target_compile_definitions(test_onednn_conv_parity_probe PRIVATE ONEDNN_PARITY_PROBE_HAVE_DNNL=1)
        target_include_directories(test_onednn_conv_parity_probe PRIVATE ${DNNL_INCLUDE_DIR})
        target_link_libraries(test_onednn_conv_parity_probe PRIVATE ${DNNL_LIB})
        get_filename_component(DNNL_LIB_DIR ${DNNL_LIB} DIRECTORY)
        set(ONEDNN_PROBE_RPATHS ${DNNL_LIB_DIR})
        set(ONEDNN_PROBE_OPTIONAL_RUNTIME_LIBS "")
        foreach(optional_lib SYCL_LIB SVML_LIB IMF_LIB INTLC_LIB TBB_LIB)
            if(${optional_lib})
                list(APPEND ONEDNN_PROBE_OPTIONAL_RUNTIME_LIBS ${${optional_lib}})
                get_filename_component(optional_lib_dir ${${optional_lib}} DIRECTORY)
                list(APPEND ONEDNN_PROBE_RPATHS ${optional_lib_dir})
            endif()
        endforeach()
        if(ONEDNN_PROBE_OPTIONAL_RUNTIME_LIBS)
            target_link_libraries(test_onednn_conv_parity_probe PRIVATE ${ONEDNN_PROBE_OPTIONAL_RUNTIME_LIBS})
        endif()
        list(REMOVE_DUPLICATES ONEDNN_PROBE_RPATHS)
        set_target_properties(test_onednn_conv_parity_probe PROPERTIES
            BUILD_RPATH "${ONEDNN_PROBE_RPATHS}"
        )
        message(STATUS "oneDNN parity probe enabled with dnnl.hpp + libdnnl. Optional oneAPI runtime libs are linked only when found.")
    else()
        target_compile_definitions(test_onednn_conv_parity_probe PRIVATE ONEDNN_PARITY_PROBE_HAVE_DNNL=0)
        message(STATUS "oneDNN parity probe target built without oneDNN linkage; runtime will report SKIP-COMPILE-NO-ONEDNN.")
    endif()
endif()

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

add_test(NAME CLIInvalidConvBackendTest
    COMMAND ppn_train --model cnn --conv_backend not_a_backend --epochs 0)
set_tests_properties(CLIInvalidConvBackendTest PROPERTIES
    WILL_FAIL TRUE)

add_test(NAME CLIOneDnnBackendUnavailableTest
    COMMAND ppn_train --model cnn --conv_backend onednn --epochs 0)
set_tests_properties(CLIOneDnnBackendUnavailableTest PROPERTIES
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
