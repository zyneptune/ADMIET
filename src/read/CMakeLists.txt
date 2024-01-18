message("read lib--${PROJECT_SOURCE_DIR}")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

aux_source_directory(./ READ_SRC)
include_directories(../../include)
add_library(read ${READ_SRC})

find_package(Armadillo REQUIRED)
# target_include_directories(read PRIVATE ${ARMADILLO_INCLUDE_DIRS})
# target_link_libraries(read PRIVATE ${ARMADILLO_LIBRARIES})
target_include_directories(read PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(read ${ARMADILLO_LIBRARIES})
# target_link_libraries(read tinyxml2)