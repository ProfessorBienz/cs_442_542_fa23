cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(cannon LANGUAGES CUDA CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(cannon_INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

cmake_policy(SET CMP0104 OLD)

option(BUILD_SHARED_LIBS "Build the shared library" ON)
option(BUILD_EXAMPLES "Build the examples" ON)
option(ENABLE_UNIT_TESTS "Enable unit testing" ON)
option(ENABLE_JENKINS_OUTPUT "Enable xml output for unit tests" OFF)

function(add_src_promisc dir)
  file(GLOB_RECURSE ${dir}_sources_ ${dir}/*.cpp)

  # remove tests
  set(EXCLUDE_DIR "/tests/")
  foreach (TMP_PATH ${${dir}_sources_})
    string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
    if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
      list (REMOVE_ITEM ${dir}_sources_ ${TMP_PATH})
    endif ()
  endforeach(TMP_PATH)

  set(${dir}_SOURCES ${${dir}_sources_} PARENT_SCOPE)
  file(GLOB_RECURSE ${dir}_includes_ ${dir}/*.hpp)
  set(${dir}_INCLUDES ${${dir}_includes_} PARENT_SCOPE)
endfunction(add_src_promisc)

if (ENABLE_UNIT_TESTS)
  enable_testing()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "-ccbin=mpicxx -arch=compute_35")

##################### 
## GOOGLETEST      ##
#####################
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  # Specify the commit you depend on and update it regularly.
  GIT_TAG 5376968f6948923e2411081fd9372e71a59d8e77
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
#####################

add_subdirectory(src)

target_compile_options(cannon PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
    -lineinfo
    -Xcompiler -Wall
    >
    $<$<COMPILE_LANGUAGE:CXX>: # Flags for C++ Objects
    -fno-omit-frame-pointer
    >
    )

set_property(TARGET cannon PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if (BUILD_EXAMPLES)
      add_subdirectory(examples)
endif()
