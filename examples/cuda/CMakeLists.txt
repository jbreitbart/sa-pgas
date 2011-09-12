# Include the current directory
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )
INCLUDE_DIRECTORIES( /usr/local/cuda/include )

list(APPEND CUDA_NVCC_FLAGS -arch sm_20)
CUDA_ADD_LIBRARY(cuda_lib kernel.cu)

#list all source files here
ADD_EXECUTABLE(cuda_example main.cpp)

#need to link to some other libraries ? just add them here
TARGET_LINK_LIBRARIES(cuda_example ${GASLIBs} cuda_lib)

# set minimum cmake version
cmake_minimum_required(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
