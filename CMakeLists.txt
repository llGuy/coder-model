cmake_minimum_required(VERSION "3.2")

project(coder_model)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Python
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind)

nanobind_add_module(coder_model_sim
    src/entry.cc
    src/sim.cc 
)

target_compile_definitions(coder_model_sim PRIVATE
    PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)

add_executable(gen
    src/gen.cc 
    src/prog.cc
)

target_compile_definitions(gen PRIVATE
    PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)
