cmake_minimum_required (VERSION 2.6)
set (CMAKE_CXX_STANDARD 11)

project (BingoCpp)

# Set the version number.
set (Tutorial_VERSION_MAJOR 0)
set (Tutorial_VERSION_MINOR 1)
configure_file (
  "${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}/version.h.in"
  "${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}/version.h"
  )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ---------------------------    DEPENDENCIES    -------------------------------

# ----------GTEST------------ 
# Download and unpack googletest at configure time
configure_file(${CMAKE_SOURCE_DIR}/cmake/CMakeLists.txt.gtest googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build")

# The gtest/gmock targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if(CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include")
endif()

# -------------PYBIND----------------
# Download and unpack googletest at configure time
configure_file(${CMAKE_SOURCE_DIR}/cmake/CMakeLists.txt.pybind11 pybind11-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pybind11-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pybind11-download" )
add_subdirectory(${CMAKE_BINARY_DIR}/pybind11/src/pybind11)


# -------------EIGEN----------------
configure_file(${CMAKE_SOURCE_DIR}/cmake/CMakeLists.txt.eigen eigen-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
   WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/eigen-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
   WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/eigen-download" )
set(Eigen3_DIR ${CMAKE_BINARY_DIR}/eigen/src/eigen/cmake)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# add include directory
include_directories("${CMAKE_SOURCE_DIR}/include/")

# add libraries
add_library(bingo STATIC src/acyclic_graph.cc src/acyclic_graph_nodes.cc)
target_link_libraries(bingo Eigen3::Eigen)
set_target_properties(bingo PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
 
 
# ---------- main executable ----------
add_executable(runBingoCpp src/driver.cc)
target_link_libraries(runBingoCpp bingo Eigen3::Eigen)


# ---------- python binding ----------
pybind11_add_module(bingocpp src/bingocpp_pymodule.cc)
target_link_libraries(bingocpp bingo)


# ---------Testing executable-------------
add_executable(run_tests test/test_main.cc test/acyclic_graph_tests.cc)
target_link_libraries(run_tests gtest bingo)


