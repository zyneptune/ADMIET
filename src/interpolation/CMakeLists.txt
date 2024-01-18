message("interpolation lib--${PROJECT_SOURCE_DIR}")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

aux_source_directory(./ INTERP_SRC)
include_directories(../../include)
add_library(interpolation ${INTERP_SRC})

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(interpolation PRIVATE Eigen3::Eigen)