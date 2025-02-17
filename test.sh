#! /bin/bash

spack load mpich@4.2.3 +cuda cuda_arch=86 netmod=ofi cmake@3.23.5
rm -rf build
cmake -S . -B build
make -C build
mpirun -n 2 compute-sanitizer build/main

# spack load mpich@4.2.3 +cuda cuda_arch=86 netmod=ucx cmake@3.23.5
# rm -rf build
# cmake -S . -B build
# make -C build
# mpirun -n 2 build/main

# spack load openmpi@5.0.5 +cuda cuda_arch=86 cmake@3.23.5
# rm -rf build
# cmake -S . -B build
# make -C build
# mpirun -n 2 build/main