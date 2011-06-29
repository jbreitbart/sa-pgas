SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib/)

# add device to library
ADD_LIBRARY(adabs STATIC adabs.cpp impl/callbacks.cpp collective/vector_base.cpp)


# set minimum cmake version
cmake_minimum_required(VERSION 2.4)

