message("ode lib--${PROJECT_SOURCE_DIR}")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

aux_source_directory(./ ODE_SRC)
include_directories(../../include)
add_library(ode ${ODE_SRC})
target_link_libraries(ode wet)