if (NOT ENABLE_CUDA)
    set(SELECTED_SWE_MEMORY_MANAGER MemoryManager.cpp)
else()
    set(SELECTED_SWE_MEMORY_MANAGER MemoryManager.cu)
endif()

target_sources(swe
        PRIVATE
        ${SELECTED_SWE_MEMORY_MANAGER})
