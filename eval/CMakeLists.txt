# Copyright (c) zili zhang & fangyue liu @PKU.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(OpenMP REQUIRED)

find_package(CUDAToolkit REQUIRED COMPONENTS nvtx)

add_executable(overall EXCLUDE_FROM_ALL overall.cu)
target_link_libraries(overall PRIVATE faiss OpenMP::OpenMP_CXX)

add_executable(overall_billion EXCLUDE_FROM_ALL overall_billion.cpp)
target_link_libraries(overall_billion PUBLIC faiss OpenMP::OpenMP_CXX)

add_executable(pcie_transfer_profile EXCLUDE_FROM_ALL pcie_transfer_profile.cpp)
target_include_directories(pcie_transfer_profile PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(pcie_transfer_profile PUBLIC faiss OpenMP::OpenMP_CXX)
target_link_libraries(pcie_transfer_profile PRIVATE CUDA::cudart)
# NVML 찾기
find_library(NVML_LIB
  NAMES nvidia-ml
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /usr/lib/wsl/lib
  NO_DEFAULT_PATH)
if(NOT NVML_LIB)
  find_library(NVML_LIB NAMES nvidia-ml)  # 시스템 기본 경로에서도 재탐색
endif()

if(NVML_LIB)
  message(STATUS "Found NVML: ${NVML_LIB}")
  target_link_libraries(pcie_transfer_profile PRIVATE ${NVML_LIB})
else()
  message(WARNING "NVML not found; build will fail unless NVML use is disabled")
endif()


