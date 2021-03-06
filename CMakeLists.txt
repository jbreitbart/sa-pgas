PROJECT(adabs)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})

find_package(CUDA QUIET REQUIRED)

# GNU Toolchain compile options
#ADD_DEFINITIONS(-g -O2 -fopenmp)

#####################
# setup intel environment
#SET(CMAKE_AR /opt/intel/bin/xiar)

# compiler: vanila
#SET(CMAKE_CXX_COMPILER /opt/intel/bin/icpc)
#SET(CMAKE_C_COMPILER /opt/intel/bin/icc)

# compiler: mpi
SET(CMAKE_CXX_COMPILER /opt/openmpi/bin/mpiCC)
SET(CMAKE_C_COMPILER /opt/openmpi/bin/mpicc)
#ADD_DEFINITIONS(-DUSE_MPI_COLLETIVES)

# Compiler options
ADD_DEFINITIONS(-Wall -O0 -g3 -Wno-unused-function -openmp)
#ADD_DEFINITIONS(-Wall -g0 -fast -Wno-unused-function -openmp)
#####################

SET(LIBs -lrt -ladabs -openmp)

# set the include path
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include)

# set link path
LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/lib ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# SMP conduit
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include/smp-conduit)
#SET(GASLIBs -lgasnet-smp-par -lgasnet_tools-par -lamudp ${LIBs})

# UDP conduit
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include/udp-conduit)
#SET(GASLIBs -lgasnet-udp-par -lgasnet_tools-par -lamudp ${LIBs})

# MPI conduit
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include/mpi-conduit)
SET(GASLIBs -lgasnet-mpi-par -lgasnet_tools-par -lammpi ${LIBs})


# if you don want the full compiler output, uncomment the following line
#SET(CMAKE_VERBOSE_MAKEFILE ON)


# build the library
SUBDIRS(src)

# build the examples
SUBDIRS(examples)

# set minimum cmake version
cmake_minimum_required(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


## UNTESTED UNTESTED UNTESTED UNTESTED UNTESTED
# IBV conduit
#SET(CMAKE_CXX_COMPILER /home/jbreitbart/temp/openmpiInstall/bin/mpic++)
#SET(CMAKE_C_COMPILER /home/jbreitbart/temp/openmpiInstall/bin/mpicc)
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include/ibv-conduit)
#SET(GASLIBs -lgasnet-ibv-par -lgasnet_tools-par -lamudp -lrt -ladabs -openmp -libverbs )


# OLD OLD OLD OLD OLD OLD OLD OLD OLD
# MPI conduit
#SET(CMAKE_CXX_COMPILER /usr/bin/mpiCC)
#SET(CMAKE_C_COMPILER /usr/bin/mpicc)
#ADD_DEFINITIONS(-g -O2 -fopenmp)

