#ifndef TOOLS_MEMORY_MANAGER_HH
#define TOOLS_MEMORY_MANAGER_HH

#include <memory>
#include <type_traits>


void* swe_alloc_void(size_t alignment, size_t elem, size_t count);
template <typename T>
T* swe_alloc(size_t count) {
  return reinterpret_cast<T*>(swe_alloc_void(alignof(T), sizeof(T), count));
}
void swe_free(void* ptr);

/// Syncs memory from device to cpu
/// If device is cpu: cpuPtr is simply set to devicePtr
/// If device is not cpu: If cpuPtr is nullptr, new memory allocation is made for cpuPtr, if not, the data contained is overwritten
/// Beware! No size checks are made
void swe_sync_from_device_void(void* devicePtr, void*& cpuPtr, size_t size, size_t alignment);

/// Syncs memory from cpu to device
/// If device is cpu: If devicePtr and cpuPtr are the same, nothing is done, else memcpy
/// If device is not cpu: Memory is copied from cpuPtr to devicePtr
void swe_sync_to_device_void(void* devicePtr, void* cpuPtr, size_t size);

/// Frees cpuPtr if necessary
void swe_cleanup_cpuPtr_void(void* devicePtr, void*& cpuPtr);

template <typename T>
void swe_sync_from_device(T* devicePtr, T*& cpuPtr, size_t count) {
  void* cpuPtrVoid = cpuPtr;
  swe_sync_from_device_void(devicePtr, cpuPtrVoid, count * sizeof(T), alignof(T));
  cpuPtr = reinterpret_cast<T*>(cpuPtrVoid);
}

template <typename T>
void swe_sync_to_device(T* devicePtr, T* cpuPtr, size_t count) {
  swe_sync_to_device_void(devicePtr, cpuPtr, count * sizeof(T));
}

template <typename T>
void swe_cleanup_cpuPtr(T* devicePtr, T*& cpuPtr) {
  void* cpuPtrVoid = cpuPtr;
  swe_cleanup_cpuPtr_void(devicePtr, cpuPtrVoid);
  cpuPtr = reinterpret_cast<T*>(cpuPtrVoid);
}

struct SWEFree {
  inline void operator()(void* ptr) { swe_free(ptr); }
};

template <typename T>
using managed_uptr = std::unique_ptr<T[], SWEFree>;

template <typename T> managed_uptr<T> make_managed_uptr(size_t count) {
  return managed_uptr<T>(swe_alloc<T>(count));
}

/*
#if defined(ENABLE_CUDA)
template <typename T> auto make_device_uptr(size_t size) {
  T *ptr;

  cudaMalloc((void **)&ptr, sizeof(T) * size);
  checkCUDAError("allocate device memory");

  struct Deleter {
    void operator()(T *ptr) const {
      cudaFree(ptr);
      checkCUDAError("free device memory");
    }
  };

  return std::unique_ptr<T[], Deleter>(ptr);
}
#endif
*/
#endif
