cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

# Set the project name and specify that it uses CUDA and C++
project(MpichOfi LANGUAGES CXX CUDA)

# Find the MPI package
find_package(MPI REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA support
enable_language(CUDA)

# Add the executable
add_executable(main main.cu)

# Link MPI libraries
target_link_libraries(main MPI::MPI_CXX)

# Set CUDA properties
set_target_properties(main PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# Optionally, you can add include directories if needed
# target_include_directories(my_cuda_mpi_executable PRIVATE ${MPI_INCLUDE_PATH})
