#ifndef TOOLS_MEMORY_MANAGER_HH
#define TOOLS_MEMORY_MANAGER_HH

#include <memory>
#include <type_traits>

template <typename T> auto make_managed_uptr(size_t size) {
  T *ptr;
#if defined(ENABLE_CUDA)
  cudaMallocManaged((void **)&ptr, sizeof(T) * size, cudaMemAttachGlobal);
  checkCUDAError("allocate managed memory");
#else
  ptr = new T[size];
#endif

  struct Deleter {
    void operator()(T *ptr) const {
#if defined(ENABLE_CUDA)
      cudaFree(ptr);
      checkCUDAError("free managed memory");
#else
      delete[] ptr;
#endif
    }
  };

  return std::unique_ptr<T[], Deleter>(ptr);
}

template <typename T>
using managed_uptr = decltype(make_managed_uptr<T>(0));

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

#endif
