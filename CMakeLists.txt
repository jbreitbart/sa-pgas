# General project settings
project(psam)
cmake_minimum_required(VERSION 2.8)

# Enable support for external projects
include(ExternalProject)

########
# Handle different build types
IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Debug
       CACHE STRING "Choose the type of build : Debug Release."
       FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
message("* Current build type is : ${CMAKE_BUILD_TYPE}")
string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
########

########
# Compiler/linker options based on the different compilers
ADD_DEFINITIONS(-std=c++11 )
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Using Clang
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    ADD_DEFINITIONS(-g -Ofast -march=native)
  endif()

  if ("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    ADD_DEFINITIONS(-g3 -O0 -Weverything -Wnoc++98-compat)
  endif()

  ADD_DEFINITIONS (-Wno-c++98-compat -Wno-c++98-compat-pedantic)

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # Using GCC
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    ADD_DEFINITIONS(-g -O2 -march=native)
  endif()

  if ("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    ADD_DEFINITIONS(-g3 -O0 -Wall -Wextra -Wunused)
  endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # Using Intel C++
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    ADD_DEFINITIONS(-g -fast)
  endif()

  if ("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    ADD_DEFINITIONS(-g3 -O0 -Wall -Wextra)
  endif()

endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "release")
  # Strip the binary
  set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -s")
endif()
########

find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

########
# Includes
# psam include path
include_directories(include)
########

########
# Compiling and linking
add_executable(jacobi examples/jacobi.cpp)
#target_link_libraries(jacobi)
########

########
# Add git-cmake-format
option(CLANG_FORMAT "Call clang-format for every call to make" ON)
if(CLANG_FORMAT)
    set (GCF_IGNORE_LIST "vendor/")
    add_subdirectory(vendor/git-cmake-format)
endif(CLANG_FORMAT)
########
