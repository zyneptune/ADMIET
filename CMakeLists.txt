cmake_minimum_required(VERSION 3.10.0)
project(main VERSION 0.3.0)

SET(CMAKE_CXX_STANDARD 20)

# Debug
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

#SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE_RELEASE "$ENV{CXXFLAGS} -O2 -DNDEBUG")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

set(WET_INCLUDE_DIR include)
aux_source_directory(src DIR_SRCS)
add_subdirectory(src)
include_directories(include)

# function fitting program
add_executable(functionfitting src/functionfitting.cpp)
target_link_libraries(functionfitting wet)
target_link_libraries(functionfitting read)

# function fitting program 2
add_executable(func_treefile src/func_treefile.cpp)
target_link_libraries(func_treefile wet)
target_link_libraries(func_treefile read)

# ode fitting program
add_executable(odefitting src/odefitting.cpp)
target_link_libraries(odefitting wet)
target_link_libraries(odefitting ode)
target_link_libraries(odefitting read)
target_link_libraries(odefitting interpolation)

# ode fitting with file program
add_executable(ode_treefile src/ode_treefile.cpp)
target_link_libraries(ode_treefile wet)
target_link_libraries(ode_treefile ode)
target_link_libraries(ode_treefile read)
target_link_libraries(ode_treefile interpolation)

# test program
add_executable(wet_test src/test.cpp)
target_link_libraries(wet_test wet)
target_link_libraries(wet_test ode)
target_link_libraries(wet_test read)
target_link_libraries(wet_test interpolation)
