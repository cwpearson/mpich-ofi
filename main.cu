#include <mpi.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

/*
Author: Carl Pearson (cwpears@sandia.gov)

Basically a send/recv test with two ranks.
Send-side sets buf(i) = i
Recv side checks that that's what it gets.

Optionally, the buffers can be offset from the start of the allocation by a configurable
alignment amount, i.e. the allocated buffer is larger, and we don't use offset 0 as the
beginning of the send/recv buffer

There's a bit of asymmetry that shouldn't matter: both ranks allocate the recv buffer,
only the send side allocates the send buffer.
The reason this is done is because the MPICH errors seem sensitive to how many
CUDA allocations are in the code, and this triggers the error case.

In MPICH 4.2.3 + ofi, we get some errors:
* With a 0 alignment offset, eventually the recv side gets garbage.
* With a 128 alignment offset, we get IPC handle mapping errors.
This works in Open MPI 5.0.5
*/

// Macro to check for CUDA errors
#define CUDA(call)                                           \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error in file '" << __FILE__              \
                      << "' in line " << __LINE__ << ": "                \
                      << cudaGetErrorString(err) << " (" << err << ")"   \
                      << std::endl;                                      \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// a(i) = i
template <typename Scalar>
__global__ void init_array(Scalar* a, int sz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < sz) {
    a[i] = Scalar(i);
  }
}

// check a(i) = i
template <typename Scalar>
__global__ void check_array(const Scalar* a, int sz, int* errs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < sz && a[i] != Scalar(i)) {
    atomicAdd(errs, 1);
  }
}

// get a builting MPI_Datatype for int32_t, int64_t, float
template <typename Scalar>
MPI_Datatype mpi_type() {
  if constexpr (std::is_same_v<Scalar, int32_t>) {
    return MPI_INT;
  } else if constexpr (std::is_same_v<Scalar, int64_t>) {
    return MPI_LONG_LONG;
  } else if constexpr (std::is_same_v<Scalar, float>) {
    return MPI_FLOAT; 
  } else {
    static_assert(std::is_void_v<Scalar>, "unsupported type");
  }
}

// if alignment is 0, return ptr
// else, return the next aligned version of ptr+1
void* align_next(void* ptr, std::size_t alignment) {
  if (0 == alignment) return ptr;
  std::uintptr_t p = reinterpret_cast<std::uintptr_t>(ptr);
  // would be p + alignment - 1 if we weren't getting the next one
  std::uintptr_t aligned_p = (p + alignment) & ~(alignment - 1); 
  return reinterpret_cast<void*>(aligned_p);
}

template <typename Scalar>
void run_test(int num_elements, int alignment, bool use_ssend) {

  // get a string name of the Scalar type
  const char *name;
  if constexpr (std::is_same_v<Scalar, int32_t>) {
    name = "int32_t";
  } else if constexpr (std::is_same_v<Scalar, float>) {
    name = "float";
  } else if constexpr (std::is_same_v<Scalar, int64_t>) {
    name = "int64_t";
  } else {
    static_assert(std::is_void_v<Scalar>, "unsupported type");
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (2 != size) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  
  if (0 == rank) {
    std::cerr << __FILE__ << ":" << __LINE__ << " test: " << num_elements << " " << name << "\n";
  }

  Scalar* d_recv_buf;
  int* d_errs;

  size_t buffer_size = num_elements * sizeof(Scalar) + alignment;
  
  CUDA(cudaMalloc(&d_recv_buf, buffer_size));
  CUDA(cudaMalloc(&d_errs, sizeof(int)));
  CUDA(cudaMemset(d_errs, 0, sizeof(int)));
  Scalar* recv_buf = reinterpret_cast<Scalar*>(align_next(d_recv_buf, alignment));
    
  if (rank == 0) {
    Scalar* d_send_buf;
    CUDA(cudaMalloc(&d_send_buf, buffer_size));
    Scalar* send_buf = reinterpret_cast<Scalar*>(align_next(d_send_buf, alignment));
    init_array<<<(num_elements + 255) / 256, 256>>>(send_buf, num_elements);
    CUDA(cudaDeviceSynchronize());

    std::cerr << __FILE__ << ":" << __LINE__ << " send: " << d_send_buf << " " << send_buf << "\n";
    if (use_ssend) {
      MPI_Ssend(send_buf, num_elements, mpi_type<Scalar>(), 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Send(send_buf, num_elements, mpi_type<Scalar>(), 1, 0, MPI_COMM_WORLD);
    }

    CUDA(cudaFree(d_send_buf));
  } else if (rank == 1) {

    int h_errs = 0;

    std::cerr << __FILE__ << ":" << __LINE__ << " recv: " << d_recv_buf << " " << recv_buf << "\n";
    MPI_Recv(recv_buf, num_elements, mpi_type<Scalar>(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    check_array<<<(num_elements + 255) / 256, 256>>>(recv_buf, num_elements, d_errs);
    CUDA(cudaMemcpy(&h_errs, d_errs, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_errs) {
      std::cerr << __FILE__ << ":" << __LINE__ << " h_errs=" << h_errs << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  CUDA(cudaFree(d_recv_buf));
  CUDA(cudaFree(d_errs));
}

template<typename Scalar>
void run_test() {
  const int alignment = 0;
  
  for (size_t _ : {0,1,2}) {
    for (size_t n : {113, 16, 8, 4, 2, 1}) {
        MPI_Barrier(MPI_COMM_WORLD);
        run_test<Scalar>(n, alignment, false /* MPI_Send */);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // for (size_t n : {113, 16, 8, 4, 2, 1}) {
    //   run_test<Scalar>(n, alignment, true /*MPI_Ssend*/);
    // }
  }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    run_test<int>();
    run_test<int64_t>();
    run_test<float>();
    MPI_Finalize();
    return 0;
}