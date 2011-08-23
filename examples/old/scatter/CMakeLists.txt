# Include the current directory
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )

#list all source files here
ADD_EXECUTABLE(scatter scatter.cpp)

#need to link to some other libraries ? just add them here
TARGET_LINK_LIBRARIES(scatter ${GASLIBs})

# set minimum cmake version
cmake_minimum_required(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
