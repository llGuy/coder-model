cmake_minimum_required(VERSION "3.2")

project(coder-model)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(common STATIC
    src/common/datalib.cc
    src/common/datalib.h
)

add_executable(gen
    src/gen/main.cc
)

add_executable(inspect
    src/inspect/main.cc
)

target_link_libraries(gen PRIVATE 
    common
)

target_include_directories(gen PRIVATE 
    src/common
)

target_link_libraries(inspect PRIVATE 
    common
)

target_include_directories(inspect PRIVATE 
    src/common
)



target_compile_definitions(gen PRIVATE
    PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)

target_compile_definitions(inspect PRIVATE
    PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)
