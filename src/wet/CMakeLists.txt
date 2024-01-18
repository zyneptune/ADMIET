message("wet lib--${PROJECT_SOURCE_DIR}")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

aux_source_directory(./ WET_SRC)
include_directories(../../include)
add_library(wet ${WET_SRC})