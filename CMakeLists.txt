cmake_minimum_required(VERSION 3.18)
project(pymeshray)

set(CMAKE_CXX_STANDARD 20)
set(BINDINGS _bvh_bind_ext)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)

message(STATUS "nanobind_DIR: ${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src/cpp/include)
include_directories(${CMAKE_SOURCE_DIR}/src/cpp/contrib)

nanobind_add_module(${BINDINGS} ${CMAKE_SOURCE_DIR}/src/cpp/bvh_bind_ext.cpp)

install(TARGETS ${BINDINGS} LIBRARY DESTINATION pyraymesh)
