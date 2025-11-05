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
