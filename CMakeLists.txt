cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(MpichOfi LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)

add_executable(main main.cu)

target_link_libraries(main MPI::MPI_CXX)

set_target_properties(main PROPERTIES
    CUDA_ARCHITECTURES "86"
)
