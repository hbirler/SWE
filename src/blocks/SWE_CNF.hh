#include "SWE_WaveAccumulationBlock.hh"

#include "cross/policies.hh"

/**
 * Compute net updates for the block.
 * The member variable #maxTimestep will be updated with the
 * maximum allowed time step size
 */
void SWE_WaveAccumulationBlock::computeNumericalFluxes() {
  float dx_inv = 1.0f / dx;
  float dy_inv = 1.0f / dy;

  RAJA::ReduceMax<policies::reduce, float> maxWaveSpeed(0.0f);

  RAJA::region<policies::region>([=]() {
    RAJA::kernel<policies::loop_2d<true>>(
        RAJA::make_tuple(RAJA::RangeSegment(1, nx + 2), RAJA::RangeSegment(1, ny + 1)), [=] RAJA_HOST_DEVICE (size_t i, size_t j) {
          float maxEdgeSpeed;
          float hNetUpLeft, hNetUpRight;
          float huNetUpLeft, huNetUpRight;

          wavePropagationSolver.computeNetUpdates(h[i - 1][j], h[i][j], hu[i - 1][j], hu[i][j], b[i - 1][j], b[i][j],
                                                  hNetUpLeft, hNetUpRight, huNetUpLeft, huNetUpRight, maxEdgeSpeed);

          // accumulate net updates to cell-wise net updates for h and hu
          hNetUpdates[i - 1][j] += dx_inv * hNetUpLeft;
          huNetUpdates[i - 1][j] += dx_inv * huNetUpLeft;
          hNetUpdates[i][j] += dx_inv * hNetUpRight;
          huNetUpdates[i][j] += dx_inv * huNetUpRight;

          maxWaveSpeed.combine(maxEdgeSpeed);
        });

    RAJA::kernel<policies::loop_2d<true>>(
        RAJA::make_tuple(RAJA::RangeSegment(1, nx + 1), RAJA::RangeSegment(1, ny + 2)), [=] RAJA_HOST_DEVICE (size_t i, size_t j) {
          float maxEdgeSpeed;
          float hNetUpDow, hNetUpUpw;
          float hvNetUpDow, hvNetUpUpw;

          wavePropagationSolver.computeNetUpdates(
              h[i][j - 1], h[i][j], hv[i][j - 1], hv[i][j], b[i][j - 1], b[i][j],
              hNetUpDow, hNetUpUpw, hvNetUpDow, hvNetUpUpw, maxEdgeSpeed);

          // accumulate net updates to cell-wise net updates for h and hu
          hNetUpdates[i][j - 1] += dy_inv * hNetUpDow;
          hvNetUpdates[i][j - 1] += dy_inv * hvNetUpDow;
          hNetUpdates[i][j] += dy_inv * hNetUpUpw;
          hvNetUpdates[i][j] += dy_inv * hvNetUpUpw;

          maxWaveSpeed.combine(maxEdgeSpeed);
        });
  });

  if (maxWaveSpeed.get() > 0.00001) {
    // TODO zeroTol

    // compute the time step width
    // CFL-Codition
    //(max. wave speed) * dt / dx < .5
    // => dt = .5 * dx/(max wave speed)

    maxTimestep = std::min(dx / maxWaveSpeed.get(), dy / maxWaveSpeed.get());

    // reduce maximum time step size by "safety factor"
    maxTimestep *= (float).4; // CFL-number = .5
  } else {
    // might happen in dry cells
    maxTimestep = std::numeric_limits<float>::max();
  }
}