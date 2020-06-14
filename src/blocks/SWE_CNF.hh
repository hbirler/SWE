#include "SWE_WaveAccumulationBlock.hh"

#include "cross/policies.hh"
#include "solvers/AugRieRaja.hpp"

/**
 * Compute net updates for the block.
 * The member variable #maxTimestep will be updated with the
 * maximum allowed time step size
 */
void SWE_WaveAccumulationBlock::computeNumericalFluxes() {
  float dx_inv = 1.0f / dx;
  float dy_inv = 1.0f / dy;

  RAJA::ReduceMax<policies::reduce, float> maxWaveSpeed(0.0f);

  // We assume all the sizes are the same

   auto hNetUpdatesL =  this->hNetUpdatesL.getDeviceView().asLight();
  auto huNetUpdatesL = this->huNetUpdatesL.getDeviceView().asLight();
  auto hvNetUpdatesL = this->hvNetUpdatesL.getDeviceView().asLight();
   auto hNetUpdatesR =  this->hNetUpdatesR.getDeviceView().asLight();
  auto huNetUpdatesR = this->huNetUpdatesR.getDeviceView().asLight();
  auto hvNetUpdatesR = this->hvNetUpdatesR.getDeviceView().asLight();
  auto h = this->h.getDeviceView().asLight();
  auto hu = this->hu.getDeviceView().asLight();
  auto hv = this->hv.getDeviceView().asLight();
  auto b = this->b.getDeviceView().asLight();

  auto rows = this->h.getRows();

  RAJA::region<policies::region>([=]() {
    RAJA::ReduceMax<policies::reduce, float> l_maxWaveSpeed(0.0f);

    RAJA::kernel<policies::loop_2d<true>>(
        RAJA::make_tuple(RAJA::RangeSegment(1, nx + 2), RAJA::RangeSegment(1, ny + 1)), [=] RAJA_HOST_DEVICE (size_t i, size_t j) {
          static constexpr size_t hNetUpLeft = 0, hNetUpRight = 1, huNetUpLeft = 2, huNetUpRight = 3, maxEdgeSpeed = 4;
          float l_netUpdates[5];

          augRieComputeNetUpdates (h(rows, i - 1, j), h(rows, i, j), hu(rows, i - 1, j), hu(rows, i, j), b(rows, i - 1, j), b(rows, i, j),
              static_cast<real>(9.81), static_cast<real>(0.01), static_cast<real>(0.000001), static_cast<real>(0.0001), 10,
              l_netUpdates);

          // accumulate net updates to cell-wise net updates for h and hu
          hNetUpdatesL(rows, i - 1, j) += dx_inv * l_netUpdates[hNetUpLeft];
          huNetUpdatesL(rows, i - 1, j) += dx_inv * l_netUpdates[huNetUpLeft];
          hNetUpdatesR(rows, i, j) += dx_inv * l_netUpdates[hNetUpRight];
          huNetUpdatesR(rows, i, j) += dx_inv * l_netUpdates[huNetUpRight];

          l_maxWaveSpeed.combine(l_netUpdates[maxEdgeSpeed]);
        });

    RAJA::kernel<policies::loop_2d<true>>(
        RAJA::make_tuple(RAJA::RangeSegment(1, nx + 1), RAJA::RangeSegment(1, ny + 2)), [=] RAJA_HOST_DEVICE (size_t i, size_t j) {
          static constexpr size_t hNetUpDow = 0, hNetUpUpw = 1, hvNetUpDow = 2, hvNetUpUpw = 3, maxEdgeSpeed = 4;
          float l_netUpdates[5];

          augRieComputeNetUpdates (h(rows, i, j - 1), h(rows, i, j), hv(rows, i, j - 1), hv(rows, i, j), b(rows, i, j - 1), b(rows, i, j),
                             static_cast<real>(9.81), static_cast<real>(0.01), static_cast<real>(0.000001), static_cast<real>(0.0001), 10,
                             l_netUpdates);

          // accumulate net updates to cell-wise net updates for h and hu
          hNetUpdatesL(rows, i, j - 1) += dy_inv * l_netUpdates[hNetUpDow];
          hvNetUpdatesL(rows, i, j - 1) += dy_inv * l_netUpdates[hvNetUpDow];
          hNetUpdatesR(rows, i, j) += dy_inv * l_netUpdates[hNetUpUpw];
          hvNetUpdatesR(rows, i, j) += dy_inv * l_netUpdates[hvNetUpUpw];

          l_maxWaveSpeed.combine(l_netUpdates[maxEdgeSpeed]);
        });

    maxWaveSpeed.combine(l_maxWaveSpeed.get());
  });

  auto mws = maxWaveSpeed.get();
  if (mws > 0.00001) {
    // TODO zeroTol

    // compute the time step width
    // CFL-Codition
    //(max. wave speed) * dt / dx < .5
    // => dt = .5 * dx/(max wave speed)

    maxTimestep = std::min(dx / mws, dy / mws);

    // reduce maximum time step size by "safety factor"
    maxTimestep *= (float).4; // CFL-number = .5
  } else {
    // might happen in dry cells
    maxTimestep = std::numeric_limits<float>::max();
  }
}


/**
 * Updates the unknowns with the already computed net-updates.
 *
 * @param dt time step width used in the update.
 */
void SWE_WaveAccumulationBlock::updateUnknowns(float dt) {
  auto hNetUpdatesL =  this->hNetUpdatesL.getDeviceView().asLight();
  auto huNetUpdatesL = this->huNetUpdatesL.getDeviceView().asLight();
  auto hvNetUpdatesL = this->hvNetUpdatesL.getDeviceView().asLight();
  auto hNetUpdatesR =  this->hNetUpdatesR.getDeviceView().asLight();
  auto huNetUpdatesR = this->huNetUpdatesR.getDeviceView().asLight();
  auto hvNetUpdatesR = this->hvNetUpdatesR.getDeviceView().asLight();
  auto h = this->h.getDeviceView().asLight();
  auto hu = this->hu.getDeviceView().asLight();
  auto hv = this->hv.getDeviceView().asLight();

  auto rows = this->h.getRows();

  RAJA::kernel<policies::loop_2d<>>(
      RAJA::make_tuple(RAJA::RangeSegment(1, nx + 1), RAJA::RangeSegment(1, ny + 1)), [=] RAJA_HOST_DEVICE (size_t i, size_t j) {
        h(rows, i, j)  -= dt * (hNetUpdatesL(rows, i, j) + hNetUpdatesR(rows, i, j));
        hu(rows, i, j) -= dt * (huNetUpdatesL(rows, i, j) + huNetUpdatesR(rows, i, j));
        hv(rows, i, j) -= dt * (hvNetUpdatesL(rows, i, j) + hvNetUpdatesR(rows, i, j));

         hNetUpdatesL(rows, i, j) = (float) 0;
        huNetUpdatesL(rows, i, j) = (float) 0;
        hvNetUpdatesL(rows, i, j) = (float) 0;
         hNetUpdatesR(rows, i, j) = (float) 0;
        huNetUpdatesR(rows, i, j) = (float) 0;
        hvNetUpdatesR(rows, i, j) = (float) 0;

        //TODO: proper dryTol
        if (h(rows, i, j) < 0.1)
          hu(rows, i, j) = hv(rows, i, j) = 0.; //no water, no speed!

        if (h(rows, i, j) < 0) {
          //zero (small) negative depths
          h(rows, i, j) = (float) 0;
        }
      });
}