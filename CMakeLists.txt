cmake_minimum_required(VERSION 3.18)

# Set project name, version and laguages here.
# Version numbers are available by including "BingoCpp/version.h" in 
# the source. See version.h.in for some more details.
project(bingocpp VERSION 0.4.0.0 LANGUAGES C CXX)


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
set(CMAKE_CXX_STANDARD 14)

# Things to always include as flags. Change as needed.
IF (NOT CMAKE_CXX_FLAGS MATCHES "-Wall -Wextra$")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra" CACHE STRING "" FORCE)
ENDIF (NOT CMAKE_CXX_FLAGS MATCHES "-Wall -Wextra$")
# Build-type specific flags. Change as needed.

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
    ${PROJECT_SOURCE_DIR}/external/googletest/googletest/include)

# Make variables referring to all the sources and test files.
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTFILES "tests/*.cpp")
set(TEST_MAIN unit_tests)  # Default name for test executable.

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# ------------------------------------------------------------------------------
#                            Build!
# ------------------------------------------------------------------------------
# Compile all sources into a library.
add_library( bingo STATIC ${SOURCES} )
add_dependencies(bingo eigen)
target_link_libraries(bingo eigen pybind11::module pybind11::headers)
set_target_properties(bingo PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
pybind11_extension(bingo)

# ---------- benchmarking library ----------
file (GLOB BENCHMARK_SRC "include/benchmarking/*.cpp")
add_library( benchmarking STATIC ${BENCHMARK_SRC} )
add_dependencies( benchmarking bingo eigen)
target_link_libraries( benchmarking bingo eigen)
set_target_properties(benchmarking PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# ---------- performance benchmark executable ----------
add_executable(performanceBenchmark app/performance_benchmarks.cpp)
add_dependencies(performanceBenchmark bingo benchmarking )  
target_link_libraries(performanceBenchmark bingo benchmarking pybind11::embed)
get_target_property(INCLUDE_DIRS performanceBenchmark INCLUDE_DIRECTORIES)

#----------- fitness benchmark executable ---------------
add_executable(fitnessBenchmark app/fitness_benchmarks.cpp)
add_dependencies(fitnessBenchmark bingo benchmarking)
target_link_libraries(fitnessBenchmark bingo benchmarking pybind11::embed)


configure_file(app/test-agraph-stacks.csv test-agraph-stacks.csv COPYONLY)
configure_file(app/test-agraph-consts.csv test-agraph-consts.csv COPYONLY)
configure_file(app/test-agraph-x-vals.csv test-agraph-x-vals.csv COPYONLY)

# ------------------------------------------------------------------------------
#                         Make Tests
# ------------------------------------------------------------------------------
# The setup of Google Test
add_subdirectory(${PROJECT_SOURCE_DIR}/external/googletest)
enable_testing()
include(GoogleTest)
# Build executable that runs the tests (and builds all dependencies).
add_executable(${TEST_MAIN} ${TESTFILES})
add_dependencies(${TEST_MAIN} bingo)
target_link_libraries(${TEST_MAIN} GTest::gtest_main bingo eigen pthread pybind11::embed)


# ------------------------------------------------------------------------------
#                         Make bingocpp (python binding)
# ------------------------------------------------------------------------------
set( MODULE_LIST
    bingocpp
)
foreach(pymodule ${MODULE_LIST})
    pybind11_add_module(${pymodule} "app/${pymodule}_pymodule.cpp") # EXCLUDE_FROM_ALL)
    target_link_libraries(${pymodule} PUBLIC bingo)
endforeach(pymodule ${MODULE_LIST})


# ------------------------------------------------------------------------------
#                         Code Coverage
# ------------------------------------------------------------------------------
# If CMake aborts due to missing dependencies for code coverage 
# (gcov, lcov, genhtml and supported compiler), comment this line.
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if (cmake_build_type_tolower STREQUAL "coverage")
    include(CodeCoverage)
endif()


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
#                         Eigen
# ------------------------------------------------------------------------------
# The following makes sure that an up-to-date version of eigen is available
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE 
                           ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)

