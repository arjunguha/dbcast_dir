cmake_minimum_required(VERSION 3.10)
project(dbcast_dir C)

find_package(MPI REQUIRED)
find_path(MFU_INCLUDE_DIR mfu.h)
find_library(MFU_LIBRARY mfu)

if(NOT MFU_INCLUDE_DIR OR NOT MFU_LIBRARY)
    message(FATAL_ERROR
        "Could not find mpifileutils (mfu.h / libmfu). "
        "Load the mpifileutils module or set CMAKE_PREFIX_PATH.")
endif()

add_executable(dbcast_dir dbcast_dir.c)
target_include_directories(dbcast_dir PRIVATE ${MFU_INCLUDE_DIR})
target_link_libraries(dbcast_dir PRIVATE ${MFU_LIBRARY} m MPI::MPI_C)
set_target_properties(dbcast_dir PROPERTIES C_STANDARD 99)
install(TARGETS dbcast_dir DESTINATION bin)
