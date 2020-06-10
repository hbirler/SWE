#ifndef TOOLS_MEMORY_MANAGER_HH
#define TOOLS_MEMORY_MANAGER_HH

#include <memory>
#include <type_traits>


void* swe_alloc(size_t alignment, size_t elem, size_t count);
template <typename T>
T* swe_alloc(size_t count) {
  return reinterpret_cast<T*>(swe_alloc(alignof(T), sizeof(T), count));
}
void swe_free(void* ptr);

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
