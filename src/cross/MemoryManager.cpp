#include "MemoryManager.hh"

#include <cstdlib>
#include <cstring>
#include <cassert>

void* swe_alloc_void(size_t alignment, size_t elem, size_t count) {
  return aligned_alloc(alignment, elem * count);
}

void swe_free(void* ptr) {
  assert(ptr);
  free(ptr);
}

/// Syncs memory from device to cpu
/// If device is cpu: cpuPtr is simply set to devicePtr
/// If device is not cpu: If cpuPtr is nullptr, new memory allocation is made for cpuPtr, if not, the data contained is overwritten
/// Beware! No size checks are made
void swe_sync_from_device_void(void* devicePtr, void*& cpuPtr, size_t size, size_t alignment) {
  cpuPtr = devicePtr;
}

/// Syncs memory from cpu to device
/// If device is cpu: If devicePtr and cpuPtr are the same, nothing is done, else memcpy
/// If device is not cpu: Memory is copied from cpuPtr to devicePtr
void swe_sync_to_device_void(void* devicePtr, void* cpuPtr, size_t size) {
  if (devicePtr == cpuPtr) return;
  memcpy(devicePtr, cpuPtr, size);
}

/// Frees cpuPtr if necessary
void swe_cleanup_cpuPtr_void(void* devicePtr, void*& cpuPtr) {
  if (cpuPtr != devicePtr && cpuPtr != nullptr) { free(cpuPtr); }
  cpuPtr = nullptr;
}