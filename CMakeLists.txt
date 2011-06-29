PROJECT(adabs)

# set the intel compiler
SET(CMAKE_CXX_COMPILER /opt/intel/bin/icpc)
SET(CMAKE_C_COMPILER /opt/intel/bin/icc)
SET(CMAKE_AR /opt/intel/bin/xiar)

# set the include path
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include)

# set link path
LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/lib ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# SMP conduit
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include/smp-conduit)
#SET(GASLIBs -lgasnet-smp-seq -lgasnet_tools-seq -lamudp -lrt -ladabs -openmp)

# UDP conduit
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include/udp-conduit)
SET(GASLIBs -lgasnet-udp-par -lgasnet_tools-par -lamudp -lrt -ladabs -openmp)
#SET(GASLIBs -lgasnet-udp-parsync -lgasnet_tools-par -lamudp -lrt -ladabs -openmp)


# IBV conduit
#SET(CMAKE_CXX_COMPILER /home/jbreitbart/temp/openmpiInstall/bin/mpic++)
#SET(CMAKE_C_COMPILER /home/jbreitbart/temp/openmpiInstall/bin/mpicc)
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include/ibv-conduit)
#SET(GASLIBs -lgasnet-ibv-par -lgasnet_tools-par -lamudp -lrt -ladabs -openmp -libverbs )


# MPI conduit
#SET(CMAKE_CXX_COMPILER /usr/bin/mpiCC)
#SET(CMAKE_C_COMPILER /usr/bin/mpicc)
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/vendor/install/include/mpi-conduit)
#SET(GASLIBs -lgasnet-mpi-par -lgasnet_tools-par -lammpi -lrt -ladabs -fopenmp)

# if you don want the full compiler output, uncomment the following line
#SET(CMAKE_VERBOSE_MAKEFILE ON)

# add definitions, compiler switches, etc.
ADD_DEFINITIONS(-Wall -O0 -g3 -Wno-unused-function -openmp)
#ADD_DEFINITIONS(-Wall -g -fast -Wno-unused-function -openmp)
#ADD_DEFINITIONS(-Wall -g -O2 -fopenmp)


# build the library
SUBDIRS(src)

# build the examples
SUBDIRS(examples)

# set minimum cmake version
cmake_minimum_required(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
