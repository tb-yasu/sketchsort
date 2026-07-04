# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
cmake_minimum_required(VERSION 3.20)
project(sketchsort CXX)

# Default to Release for single-config generators. Without this, a plain
# `cmake -B build` produces an unoptimized binary whose float-ops can reorder
# differently from the wheel build, breaking byte-exact golden comparisons.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Release/Debug/RelWithDebInfo/MinSizeRel)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(sketchsort_core STATIC
    src/sketch_sort.cpp
)

target_include_directories(sketchsort_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Vendored Boost (2011) ships with deprecated constructs; silence the noise.
# This is project-wide acceptable here because the only non-Boost translation
# unit is sketch_sort.cpp itself, which we keep clean.
target_compile_options(sketchsort_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-deprecated-declarations>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-deprecated>
)

option(SKETCHSORT_BUILD_PYTHON "Build the Python extension module (_core)" ON)
option(SKETCHSORT_BUILD_CLI    "Build the standalone C++ CLI binary"        OFF)

if(SKETCHSORT_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_core
        bindings/python/sketchsort_ext.cpp
    )

    target_link_libraries(_core PRIVATE sketchsort_core)

    install(TARGETS _core
        LIBRARY DESTINATION sketchsort
        RUNTIME DESTINATION sketchsort
        ARCHIVE DESTINATION sketchsort
    )
endif()

if(SKETCHSORT_BUILD_CLI)
    add_executable(sketchsort_cli src/main.cpp)
    target_link_libraries(sketchsort_cli PRIVATE sketchsort_core)
    set_target_properties(sketchsort_cli PROPERTIES OUTPUT_NAME sketchsort)
endif()
