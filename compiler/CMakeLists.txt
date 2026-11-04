cmake_minimum_required(VERSION 3.20.0)
project(CausalInferenceCompiler LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH
  "${MLIR_CMAKE_DIR}"
  "${LLVM_CMAKE_DIR}"
)

include(AddLLVM)
include(AddMLIR)
include(TableGen)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

add_subdirectory(include/Causal)
add_subdirectory(lib/Causal)
add_subdirectory(tools/ci-opt)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
