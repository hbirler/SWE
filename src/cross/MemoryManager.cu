#include "MemoryManager.hh"

#include <cuda_runtime.h>

static void checkCUDAErrorX(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "\nCuda error (%s): %s.\n", msg, cudaGetErrorString( err) );
    exit(-1);
  }
}

void* swe_alloc(size_t alignment, size_t elem, size_t count) {
  void *ptr;
  cudaMallocManaged(&ptr, elem * count, cudaMemAttachGlobal);
  checkCUDAErrorX("allocate managed memory");
  return ptr;
}

void swe_free(void* ptr) {
  cudaFree(ptr);
  checkCUDAErrorX("free managed memory");
}