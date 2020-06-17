#ifndef TOOLS_POLICIES_HH
#define TOOLS_POLICIES_HH

#include <RAJA/RAJA.hpp>
#include <RAJA/pattern/kernel.hpp>

namespace policies {
#if defined(LOOP_OPENMP) && !defined(CUDA)
using region = RAJA::omp_parallel_region;
#else
using region = RAJA::seq_region;
#endif

#ifdef CUDA
using reduce = RAJA::cuda_reduce;
using reduce_seq = RAJA::cuda_reduce;
#else
#ifdef LOOP_OPENMP
using reduce = RAJA::omp_reduce;
using reduce_seq = RAJA::seq_reduce;
#else
using reduce = RAJA::seq_reduce;
using reduce_seq = RAJA::seq_reduce;
#endif
#endif

#ifdef CUDA
template <bool ExistingRegion, bool NoWait> struct OuterExecPolicy { using type = RAJA::cuda_exec<256>; };
#else
#ifdef LOOP_OPENMP
template <bool ExistingRegion, bool NoWait> struct OuterExecPolicy;
template <> struct OuterExecPolicy<true, true> { using type = RAJA::omp_for_nowait_exec; };
template <> struct OuterExecPolicy<true, false> { using type = RAJA::omp_for_exec; };
template <> struct OuterExecPolicy<false, true> { using type = RAJA::omp_parallel_for_exec; };
template <> struct OuterExecPolicy<false, false> { using type = RAJA::omp_parallel_for_exec; };
#else
#ifdef VECTORIZE
// Cannot use SIMD safely with current version of RAJA since:
// "RAJA reductions used with SIMD execution policies are not guaranteed to generate correct results at present."
template <bool ExistingRegion, bool NoWait>
template <bool ExistingRegion, bool NoWait> struct OuterExecPolicy { using type = RAJA::loop_exec; };
#else
template <bool ExistingRegion, bool NoWait>
template <bool ExistingRegion, bool NoWait> struct OuterExecPolicy { using type = RAJA::loop_exec; };
#endif
#endif
#endif
template <bool ExistingRegion, bool NoWait = false>
using outer_exec_policy = typename OuterExecPolicy<ExistingRegion, NoWait>::type;

#ifdef VECTORIZE
// Cannot use SIMD safely with current version of RAJA since:
// "RAJA reductions used with SIMD execution policies are not guaranteed to generate correct results at present."
using inner_exec_policy = RAJA::loop_exec;
#else
using inner_exec_policy = RAJA::loop_exec;
#endif

template <bool ExistingRegion = false, bool NoWait = false>
using loop_1d = RAJA::KernelPolicy<
    RAJA::statement::For<0, outer_exec_policy<ExistingRegion, NoWait>,
        RAJA::statement::Lambda<0>
    >>;

#ifdef CUDA
template <bool ExistingRegion = false, bool NoWait = false>
using loop_2d =
RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_block_x_loop,    // row
            RAJA::statement::For<0, RAJA::cuda_thread_x_loop, // col
                RAJA::statement::Lambda<0>
            >>>>;
#else
template <bool ExistingRegion = false, bool NoWait = false>
using loop_2d = RAJA::KernelPolicy<
    RAJA::statement::For<1, outer_exec_policy<ExistingRegion, NoWait>,
    RAJA::statement::For<0, inner_exec_policy,
        RAJA::statement::Lambda<0>
    >>>;
#endif


}

#endif