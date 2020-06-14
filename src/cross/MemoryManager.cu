#include "MemoryManager.hh"

#include <cassert>
#include <cuda_runtime.h>

static void checkCUDAErrorX(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "\nCuda error (%s): %s.\n", msg, cudaGetErrorString( err) );
    exit(-1);
  }
}

void* swe_alloc_void(size_t alignment, size_t elem, size_t count) {
  void *ptr;
  //cudaMallocManaged(&ptr, elem * count, cudaMemAttachGlobal);
  cudaMalloc(&ptr, elem * count);
  checkCUDAErrorX("allocate managed memory");
  return ptr;
}

void swe_free(void* ptr) {
  cudaFree(ptr);
  checkCUDAErrorX("free managed memory");
}

/// Syncs memory from device to cpu
/// If device is cpu: cpuPtr is simply set to devicePtr
/// If device is not cpu: If cpuPtr is nullptr, new memory allocation is made for cpuPtr, if not, the data contained is overwritten
/// Beware! No size checks are made
void swe_sync_from_device_void(void* devicePtr, void*& cpuPtr, size_t size, size_t alignment) {
  if (cpuPtr == nullptr) {
    cpuPtr = aligned_alloc(alignment, size);
  }
  cudaMemcpy(cpuPtr, devicePtr, size, cudaMemcpyDeviceToHost);
}

/// Syncs memory from cpu to device
/// If device is cpu: If devicePtr and cpuPtr are the same, nothing is done, else memcpy
/// If device is not cpu: Memory is copied from cpuPtr to devicePtr
void swe_sync_to_device_void(void* devicePtr, void* cpuPtr, size_t size) {
  if (size == 0) return;
  assert(cpuPtr != nullptr);
  cudaMemcpy(devicePtr, cpuPtr, size, cudaMemcpyHostToDevice);
}

/// Frees cpuPtr if necessary
void swe_cleanup_cpuPtr_void(void* devicePtr, void*& cpuPtr) {
  if (cpuPtr != nullptr) { free(cpuPtr); }
  cpuPtr = nullptr;
}