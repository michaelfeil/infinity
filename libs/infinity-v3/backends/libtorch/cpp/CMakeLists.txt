cmake_minimum_required(VERSION 3.25)
project(tei_torch LANGUAGES CXX)
if(NOT DEFINED ENV_VAR_BASE)
  if(DEFINED ENV{ENV_VAR_BASE})
    set(ENV_VAR_BASE "$ENV{ENV_VAR_BASE}")
  else()
    set(ENV_VAR_BASE "INFINITY")
  endif()
endif()
if(NOT ENV_VAR_BASE MATCHES "^[A-Z_][A-Z0-9_]*$")
  message(FATAL_ERROR "ENV_VAR_BASE must match [A-Z_][A-Z0-9_]*")
endif()
configure_file(env_config.h.in env_config.h @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100;120" CACHE STRING
      "CUDA architectures for portable LibTorch kernel distributions")
endif()
# Torch probes architectures before our override below. Use the same targets
# here so a narrowed build also works with older CUDA toolkits, independently
# of the build host's GPU or an inherited TORCH_CUDA_ARCH_LIST.
set(TORCH_CUDA_ARCH_LIST "")
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(arch MATCHES "^([0-9]+)([0-9])([a-z]?)(-real|-virtual)?$")
    set(torch_arch "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${CMAKE_MATCH_3}")
    if(CMAKE_MATCH_4 STREQUAL "-virtual")
      string(APPEND torch_arch "+PTX")
    endif()
    list(APPEND TORCH_CUDA_ARCH_LIST "${torch_arch}")
  else()
    # CMake resolves special values such as all/native itself. Torch only
    # needs a baseline probe; its generated flags are removed below.
    set(TORCH_CUDA_ARCH_LIST "8.0")
    break()
  endif()
endforeach()
list(REMOVE_DUPLICATES TORCH_CUDA_ARCH_LIST)
set(TEI_REQUESTED_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
find_package(Torch 2.14.1 EXACT REQUIRED)
# Torch appends its own gencode flags and disables CMake architecture selection.
# Honor the caller's architectures for our kernels, retaining other Torch flags.
if(TARGET torch_cuda AND TEI_REQUESTED_CUDA_ARCHITECTURES)
  torch_cuda_get_nvcc_gencode_flag(TEI_TORCH_ARCH_FLAGS)
  string(JOIN " " TEI_TORCH_ARCH_FLAGS_JOINED ${TEI_TORCH_ARCH_FLAGS})
  string(REPLACE " ${TEI_TORCH_ARCH_FLAGS_JOINED}" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CUDA_ARCHITECTURES "${TEI_REQUESTED_CUDA_ARCHITECTURES}")
endif()
add_library(tei_torch SHARED bridge.cpp batching/radix.cpp helpers/fp8_linear.cpp helpers/modern_cpu_math.cpp models/decoder_models.cpp models/decision_models.cpp models/dense_models.cpp models/encoder_models.cpp models/gemma_models.cpp models/qwen3_vl_models.cpp models/mistral3_models.cpp models/qwen35_models.cpp batching/cuda_graph.cpp kernels/moe_kernels.cpp)
# Keep separately rounded adapted CPU products and sums.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(helpers/modern_cpu_math.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_compile_features(tei_torch PRIVATE cxx_std_20)
target_sources(tei_torch PRIVATE batching/prefix_cache.cpp)
target_sources(tei_torch PRIVATE batching/prefix_lookup.cpp)
target_sources(tei_torch PRIVATE helpers/prefix_linear.cpp)
target_include_directories(tei_torch PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(tei_torch PRIVATE "${TORCH_LIBRARIES}")
set_target_properties(tei_torch PROPERTIES INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;$ORIGIN")
install(TARGETS tei_torch LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
# Fork license adoption by Infinity Core Contributors; retain native component notices.
install(FILES ../../../LICENSE ../../../NOTICE ../../../LICENSING.md
  ../../../THIRD_PARTY_NOTICES.md DESTINATION share/licenses/tei_torch)
install(DIRECTORY ../../../LICENSES/ DESTINATION share/licenses/tei_torch/LICENSES)
install(FILES vendor/xxhash/LICENSE vendor/xxhash/provenance.json
  DESTINATION share/licenses/tei_torch/xxhash)

install(DIRECTORY vendor/transformers_pixtral DESTINATION share/licenses/tei_torch)

# Custom packed operators use the same current CUDA stream as ATen.
if(TARGET torch_cuda)
  enable_language(CUDA)
  target_sources(tei_torch PRIVATE kernels/laya_features.cu kernels/qwen35_chunk.cu)
  install(DIRECTORY vendor/transformers_gdn DESTINATION share/licenses/tei_torch)
  set_source_files_properties(kernels/laya_features.cu PROPERTIES COMPILE_OPTIONS "--fmad=false;--prec-div=true;--prec-sqrt=true")
  # The exact router keeps F32 subnormals; RMS kernels retain their old flags.
  target_sources(tei_torch PRIVATE kernels/decoder_half_moe.cu)
  set_source_files_properties(kernels/decoder_half_moe.cu PROPERTIES
    COMPILE_OPTIONS "--ftz=false;--prec-div=true;--prec-sqrt=true;--expt-relaxed-constexpr")
  target_sources(tei_torch PRIVATE kernels/gemma_score.cu)
  target_sources(tei_torch PRIVATE kernels/prefix_state_copy.cu)
  target_sources(tei_torch PRIVATE kernels/prefix_input.cu)
  set_source_files_properties(kernels/gemma_score.cu PROPERTIES
    COMPILE_OPTIONS "--fmad=false;--prec-div=true;--prec-sqrt=true;--expt-relaxed-constexpr")
  target_sources(tei_torch PRIVATE kernels/radix_qkv.cu kernels/qwen35_vision.cu kernels/qwen35_reference.cu kernels/qwen35_recurrent.cu kernels/qwen35_kernels.cu kernels/fast_kernels.cu kernels/fp8_quant.cu kernels/encoder_activation.cu kernels/encoder_bias.cu kernels/encoder_norm.cu kernels/distil_embedding.cu kernels/decoder_norm.cu kernels/decoder_packed_readout.cu kernels/packed_pool.cu
    kernels/torch_flash_alibi64.cpp kernels/torch_flash_alibi64.cu
    kernels/torch_flash_fma.cpp kernels/torch_flash_fma_32.cu kernels/torch_flash_fma_64.cu kernels/torch_flash_fma_72.cu kernels/torch_flash_fma_128.cu kernels/torch_flash_fma_256.cu kernels/torch_flash_fma_512.cu kernels/torch_flash_unfused.cu)
  # Match independently verified core LayerNorm/softmax: default FMA and
  # precise exp/division, without --use_fast_math.
  set_source_files_properties(kernels/qwen35_vision.cu PROPERTIES COMPILE_OPTIONS "--fmad=true;--prec-div=true;--prec-sqrt=true")
  install(FILES vendor/qwen35_vision/LICENSE-APACHE vendor/qwen35_vision/LICENSE-MIT
    vendor/qwen35_vision/PROVENANCE.md DESTINATION share/licenses/tei_torch/qwen35_vision)
  set_source_files_properties(kernels/qwen35_reference.cu PROPERTIES COMPILE_OPTIONS "--fmad=false;--expt-relaxed-constexpr")
  target_sources(tei_torch PRIVATE kernels/torch_flash_exact32.cpp kernels/torch_flash_exact32.cu)
  set_source_files_properties(kernels/torch_flash_exact32.cu PROPERTIES
    COMPILE_OPTIONS "--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda"
    COMPILE_DEFINITIONS "FLASHATTENTION_DISABLE_DROPOUT;FLASHATTENTION_DISABLE_BACKWARD;FLASHATTENTION_DISABLE_ALIBI;FLASHATTENTION_DISABLE_SOFTCAP;FLASHATTENTION_DISABLE_UNEVEN_K")
  set_source_files_properties(kernels/encoder_norm.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math;--expt-relaxed-constexpr")
  set_source_files_properties(kernels/distil_embedding.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math;--default-stream=per-thread;--expt-relaxed-constexpr;--expt-extended-lambda")
  set_source_files_properties(kernels/encoder_activation.cu PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr")
  set_source_files_properties(kernels/decoder_norm.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math;--expt-relaxed-constexpr")
  # Exact Torch-pinned Flash forward source, with private kernel symbols and
  # fused softmax arithmetic. This operates on actual packed tokens.
  set_source_files_properties(kernels/torch_flash_fma_64.cu kernels/torch_flash_fma_128.cu kernels/torch_flash_fma_256.cu kernels/torch_flash_fma_512.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda"
    COMPILE_DEFINITIONS "FLASHATTENTION_DISABLE_DROPOUT;FLASHATTENTION_DISABLE_BACKWARD;FLASHATTENTION_DISABLE_ALIBI;FLASHATTENTION_DISABLE_SOFTCAP;FLASHATTENTION_DISABLE_UNEVEN_K")
  # Keep Torch's default UNFUSE_FMA softmax in a distinct namespace for head64/128
  # prefix suffixes. The existing fused specializations keep their arithmetic.
  set_source_files_properties(kernels/torch_flash_unfused.cu PROPERTIES
    COMPILE_OPTIONS "--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda"
    COMPILE_DEFINITIONS "UNFUSE_FMA;FLASHATTENTION_DISABLE_DROPOUT;FLASHATTENTION_DISABLE_BACKWARD;FLASHATTENTION_DISABLE_ALIBI;FLASHATTENTION_DISABLE_SOFTCAP;FLASHATTENTION_DISABLE_UNEVEN_K")
  # Head72 uses internal traits96 with true-dimension bounds; uneven-K masking
  # must remain enabled rather than exposing synthetic head columns.
  set_source_files_properties(kernels/torch_flash_fma_72.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda"
    COMPILE_DEFINITIONS "FLASHATTENTION_DISABLE_DROPOUT;FLASHATTENTION_DISABLE_BACKWARD;FLASHATTENTION_DISABLE_ALIBI;FLASHATTENTION_DISABLE_SOFTCAP")
  # Scope ALiBi to existing head32 and private JinaCode head64 specializations.
  set_source_files_properties(kernels/torch_flash_fma_32.cu kernels/torch_flash_alibi64.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda"
    COMPILE_DEFINITIONS "FLASHATTENTION_DISABLE_DROPOUT;FLASHATTENTION_DISABLE_BACKWARD;FLASHATTENTION_DISABLE_SOFTCAP;FLASHATTENTION_DISABLE_UNEVEN_K")
  install(FILES vendor/pytorch_cublaslt/LICENSE vendor/pytorch_cublaslt/PROVENANCE.md
    DESTINATION share/licenses/tei_torch/pytorch_cublaslt)
  install(FILES vendor/torch_flash_attention/LICENSE vendor/torch_flash_attention/AUTHORS
    vendor/torch_flash_attention/provenance.json DESTINATION share/licenses/tei_torch/torch_flash_attention)
  install(FILES ../../candle/extensions/candle-layer-norm/LICENSE
    DESTINATION share/licenses/tei_torch/candle-layer-norm)
  install(FILES ../../candle/src/kernels/LICENSE.candle-MIT
    DESTINATION share/licenses/tei_torch/candle-kernels)
  install(FILES ../../candle/extensions/candle-cublaslt/LICENSE-MIT
    DESTINATION share/licenses/tei_torch/candle-cublaslt)
  target_include_directories(tei_torch PRIVATE ../../candle/extensions/candle-layer-norm/kernels)
  find_package(CUDAToolkit REQUIRED)
  target_link_directories(tei_torch PRIVATE "${CUDAToolkit_LIBRARY_DIR}")
  target_compile_definitions(tei_torch PRIVATE TEI_TORCH_CUDA_KERNELS)
  set_target_properties(tei_torch PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_link_libraries(tei_torch PRIVATE torch_cuda c10_cuda CUDA::cublasLt)
  # Same pinned CUTLASS revision as Candle. Reuse its local checkout when
  # available; CUTLASS_ROOT supports offline native builds elsewhere.
  set(CUTLASS_ROOT "$ENV{CUTLASS_ROOT}" CACHE PATH "Pinned CUTLASS source root")
  if(NOT CUTLASS_ROOT AND EXISTS "$ENV{HOME}/.cudaforge/git/checkouts/cutlass-e406c186f510a150/include/cutlass/cutlass.h")
    set(CUTLASS_ROOT "$ENV{HOME}/.cudaforge/git/checkouts/cutlass-e406c186f510a150")
  endif()
  if(NOT CUTLASS_ROOT)
    include(FetchContent)
    FetchContent_Declare(tei_cutlass URL https://github.com/NVIDIA/cutlass/archive/e406c186f510a15091cce01f782020ceb7ba8eb5.tar.gz)
    FetchContent_GetProperties(tei_cutlass)
    if(NOT tei_cutlass_POPULATED)
      FetchContent_Populate(tei_cutlass)
    endif()
    set(CUTLASS_ROOT "${tei_cutlass_SOURCE_DIR}")
  endif()
  target_sources(tei_torch PRIVATE kernels/moe_portable.cu)
  target_sources(tei_torch PRIVATE kernels/nomic_moe.cpp kernels/nomic_moe.cu)
  set_source_files_properties(kernels/nomic_moe.cu PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr")
  target_include_directories(tei_torch PRIVATE "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}/tools/util/include")
  target_compile_definitions(tei_torch PRIVATE TEI_TORCH_MOE_KERNELS)
  install(FILES ../../candle/extensions/candle-gemma4-moe/LICENSE.cutlass
    DESTINATION share/licenses/tei_torch)
  set_source_files_properties(kernels/moe_portable.cu PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr")
  if("90" IN_LIST TEI_REQUESTED_CUDA_ARCHITECTURES OR "90a" IN_LIST TEI_REQUESTED_CUDA_ARCHITECTURES OR CMAKE_CUDA_FLAGS MATCHES "compute_90[, ]")
    add_library(tei_moe_hopper OBJECT kernels/moe_hopper.cu)
    set_target_properties(tei_moe_hopper PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20 CUDA_ARCHITECTURES 90a)
    target_include_directories(tei_moe_hopper PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}/tools/util/include")
    target_compile_options(tei_moe_hopper PRIVATE --expt-relaxed-constexpr -DNDEBUG)
    target_sources(tei_torch PRIVATE $<TARGET_OBJECTS:tei_moe_hopper>)
    target_compile_definitions(tei_torch PRIVATE TEI_TORCH_MOE_HOPPER)
  endif()
endif()

option(INFINITY_BUILD_NATIVE_TESTS "Build native packed model regression tests" OFF)
option(INFINITY_RUN_CUDA_TESTS "Register native tests requiring a CUDA GPU" OFF)
if(INFINITY_BUILD_NATIVE_TESTS)
  enable_testing()
  foreach(test radix prefix_lookup decoder_models decision_models dense_models encoder_models gemma_models qwen35_models qwen3_vl_models)
    add_executable(${test}_test tests/${test}_test.cpp)
    target_compile_features(${test}_test PRIVATE cxx_std_20)
    target_link_libraries(${test}_test PRIVATE tei_torch "${TORCH_LIBRARIES}")
    add_test(NAME ${test}_cpu COMMAND ${test}_test)
  endforeach()
  if(TARGET torch_cuda)
    # Match the private cache header layout even when CUDA test registration is
    # disabled: CPU fixtures can still link against a CUDA-enabled library.
    target_compile_definitions(gemma_models_test PRIVATE TEI_TORCH_CUDA_KERNELS)
    target_compile_definitions(decoder_models_test PRIVATE TEI_TORCH_CUDA_KERNELS)
    target_link_libraries(decoder_models_test PRIVATE CUDA::cudart)
    target_compile_definitions(qwen3_vl_models_test PRIVATE TEI_TORCH_CUDA_KERNELS)
    target_compile_definitions(qwen35_models_test PRIVATE TEI_TORCH_CUDA_KERNELS)
  endif()
  if(TARGET torch_cuda AND INFINITY_RUN_CUDA_TESTS)
    foreach(test output bert_private_flash qwen35_reference qwen35_chunk fp8_linear decoder_fp8 fast_kernels encoder_activation encoder_norm decoder_norm cuda_graph gemma_vision_kernels gemma_rotary packed_pool packed_alibi packed_relative_bias moe_kernels moe_cuda_graph torch_flash_fma laya_features)
      add_executable(${test}_test tests/${test}_test.cpp)
      target_compile_features(${test}_test PRIVATE cxx_std_20)
      target_link_libraries(${test}_test PRIVATE tei_torch "${TORCH_LIBRARIES}")
      add_test(NAME ${test}_cuda COMMAND ${test}_test)
    endforeach()
    add_test(NAME decision_models_cuda COMMAND decision_models_test --cuda)
    target_compile_definitions(output_test PRIVATE TEI_TORCH_CUDA_KERNELS)
    target_compile_definitions(bert_private_flash_test PRIVATE TEI_TORCH_CUDA_KERNELS)
    target_compile_options(qwen35_reference_test PRIVATE -ffp-contract=off)
    target_link_libraries(fp8_linear_test PRIVATE CUDA::cublasLt)
    target_link_libraries(decoder_fp8_test PRIVATE CUDA::cublasLt)
    target_link_libraries(fast_kernels_test PRIVATE CUDA::cublasLt)
    set_tests_properties(cuda_graph_cuda PROPERTIES
      ENVIRONMENT "TORCH_CUBLAS_WORKSPACE_CACHE=1")
    set_tests_properties(fast_kernels_cuda PROPERTIES
      ENVIRONMENT "TORCH_BLAS_PREFER_CUBLASLT=1;CUBLASLT_WORKSPACE_SIZE=32768")
    target_link_libraries(decoder_norm_test PRIVATE CUDA::cublasLt)
    target_link_libraries(bert_private_flash_test PRIVATE CUDA::cublasLt)
    foreach(test decoder_models encoder_models gemma_models qwen35_models qwen3_vl_models)
      add_test(NAME ${test}_cuda COMMAND ${test}_test cuda:0)
    endforeach()
  endif()
endif()
