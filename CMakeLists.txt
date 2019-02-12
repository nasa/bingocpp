cmake_minimum_required(VERSION 3.1.3)

# Set project name, version and laguages here.
# Version numbers are available by including "BingoCpp/version.h" in 
# the source. See version.h.in for some more details.
project(BingoCpp VERSION 0.1.1.0 LANGUAGES C CXX)


# ---------- version ----------
configure_file (
  "${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/version.h.in"
  "${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/version.h")

# Include stuff. 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(ConfigSafeGuards)
include(Colors)


# ------------------------------------------------------------------------------
#                          Compile flags
# ------------------------------------------------------------------------------
# Set the C++ standard you wish to use (will apply to all files).
# If you do not use any features that limits the standard required,
# you could omit this line.
set(CMAKE_CXX_STANDARD 11)

# Things to always include as flags. Change as needed.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")  

# Build-type specific flags. Change as needed.
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

message(STATUS "Building with the following extra flags: ${CMAKE_CXX_FLAGS}")


# ------------------------------------------------------------------------------
#                         Locate files (no change needed).
# ------------------------------------------------------------------------------
# We make sure that CMake sees all the files.
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/external/googletest
    ${PROJECT_SOURCE_DIR}/external/googletest/include)

# Make variables referring to all the sources and test files.
file(GLOB SOURCES "src/*.cpp")
file(GLOB TESTFILES "tests/*.cpp")
set(TEST_MAIN unit_tests.x)  # Default name for test executable.


# ------------------------------------------------------------------------------
#                                 Pybind
# ------------------------------------------------------------------------------




# ------------------------------------------------------------------------------
#                            Build!
# ------------------------------------------------------------------------------
# Compile all sources into a library.
add_library( bingo STATIC ${SOURCES} )
add_dependencies(bingo eigen)
target_link_libraries(bingo eigen)
set_target_properties(bingo PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
# target_link_libraries(bingo profiler)

# ---------- main executable ----------
add_executable(runBingoCpp app/driver.cpp)
add_dependencies(runBingoCpp bingo)  
target_link_libraries(runBingoCpp bingo)

# ---------- benchmarking executable ----------
add_executable(runBenchmark app/step_profile.cpp)
add_dependencies(runBenchmark bingo)  
target_link_libraries(runBenchmark bingo)

# ---------- performance benchmark executable ----------
add_executable(performanceBenchmark app/performance_benchmarks.cpp)
add_dependencies(performanceBenchmark bingo)  
target_link_libraries(performanceBenchmark bingo)
configure_file(app/test-agraph-stacks.csv test-agraph-stacks.csv COPYONLY)
configure_file(app/test-agraph-consts.csv test-agraph-consts.csv COPYONLY)
configure_file(app/test-agraph-x-vals.csv test-agraph-x-vals.csv COPYONLY)
# ------------------------------------------------------------------------------
#                         Make Tests
# ------------------------------------------------------------------------------
# Add a make target 'gtest', that runs the tests (and builds all dependencies).
# The setup of Google Test is done at the very end of this file.
add_executable(${TEST_MAIN} ${TESTFILES})
add_dependencies(${TEST_MAIN} googletest bingo)
target_link_libraries(${TEST_MAIN} googletest bingo pthread)
add_custom_target(gtest 
    COMMAND "${PROJECT_BINARY_DIR}/${TEST_MAIN}"
    DEPENDS bingo ${TEST_MAIN})

# Add a standard make target 'test' that runs the tests under CTest (only as an alt. to gtest).
include(CTest)
enable_testing()
add_test(unit_tests ${PROJECT_BINARY_DIR}/${TEST_MAIN})


# ------------------------------------------------------------------------------
#                         Make bingocpp (python binding)
# ------------------------------------------------------------------------------
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/pybind/CMakeLists.txt")
    message("Unable to find pybind")
    execute_process(COMMAND git submodule update --init -- external/pybind
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory( external/pybind EXCLUDE_FROM_ALL)
else()
    add_subdirectory( external/pybind EXCLUDE_FROM_ALL)
endif()
pybind11_add_module(bingocpp app/bingocpp_pymodule.cpp EXCLUDE_FROM_ALL)
target_link_libraries(bingocpp PUBLIC bingo)


# ------------------------------------------------------------------------------
#                         Code Coverage
# ------------------------------------------------------------------------------
# If CMake aborts due to missing dependencies for code coverage 
# (gcov, lcov, genhtml and supported compiler), comment this line.
include(CodeCoverage)


# ------------------------------------------------------------------------------
#                         Documentation
# ------------------------------------------------------------------------------
# Add a make target 'doc' to generate API documentation with Doxygen.
# You should set options to your liking in the file 'Doxyfile.in'.
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc 
        ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/doxy.config &> doxygen.log
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/doc
        COMMENT "${BoldMagenta}Generating API documentation with Doxygen (open ${PROJECT_SOURCE_DIR}/doc/html/index.html to view).${ColourReset}" VERBATIM
        )
endif(DOXYGEN_FOUND)


# ------------------------------------------------------------------------------
#                         Git Submodule Updating
# ------------------------------------------------------------------------------
# This is a target which can be used to ensure the most up to date versions of 
# the submodules are used
add_custom_target( git_update
    COMMAND git submodule init
    COMMAND git submodule update
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} )


# ------------------------------------------------------------------------------
#                         Google Test
# ------------------------------------------------------------------------------
# The following makes sure that an up-to-date version of googletest is available,
# and built so that it may be used by your tests. 
add_library( googletest
    ${PROJECT_SOURCE_DIR}/external/googletest/src/gtest-all.cc
    ${PROJECT_SOURCE_DIR}/external/googletest/src/gtest_main.cc )
add_dependencies(googletest git_update)
set_source_files_properties(${PROJECT_SOURCE_DIR}/external/googletest/src/gtest-all.cc  PROPERTIES GENERATED 1)
set_source_files_properties(${PROJECT_SOURCE_DIR}/external/googletest/src/gtest_main.cc PROPERTIES GENERATED 1)


# ------------------------------------------------------------------------------
#                         Eigen
# ------------------------------------------------------------------------------
# The following makes sure that an up-to-date version of eigen is available
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE 
                           ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)
add_dependencies(eigen git_update)

