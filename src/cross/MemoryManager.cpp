#include "MemoryManager.hh"

#include <cstdlib>

void* swe_alloc(size_t alignment, size_t elem, size_t count) {
  aligned_alloc(alignment, elem * count);
}

void swe_free(void* ptr) {
  free(ptr);
}