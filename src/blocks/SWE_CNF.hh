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

   auto hNetUpdatesL =  this->hNetUpdatesL.getDeviceView();
  auto huNetUpdatesL = this->huNetUpdatesL.getDeviceView();
  auto hvNetUpdatesL = this->hvNetUpdatesL.getDeviceView();
   auto hNetUpdatesR =  this->hNetUpdatesR.getDeviceView();
  auto huNetUpdatesR = this->huNetUpdatesR.getDeviceView();
  auto hvNetUpdatesR = this->hvNetUpdatesR.getDeviceView();
  auto h = this->h.getDeviceView();
  auto hu = this->hu.getDeviceView();
  auto hv = this->hv.getDeviceView();
  auto b = this->b.getDeviceView();

  RAJA::region<policies::region>([=]() {
    RAJA::ReduceMax<policies::reduce, float> l_maxWaveSpeed(0.0f);

    RAJA::kernel<policies::loop_2d<true>>(
        RAJA::make_tuple(RAJA::RangeSegment(1, nx + 2), RAJA::RangeSegment(1, ny + 1)), [=] RAJA_HOST_DEVICE (size_t i, size_t j) {
          constexpr unsigned hNetUpLeft = 0, hNetUpRight = 1, huNetUpLeft = 2, huNetUpRight = 3, maxEdgeSpeed = 4;
          float l_netUpdates[5];

          augRieComputeNetUpdates (h(i - 1, j), h(i, j), hu(i - 1, j), hu(i, j), b(i - 1, j), b(i, j),
              static_cast<real>(9.81), static_cast<real>(0.01), static_cast<real>(0.000001), static_cast<real>(0.0001), 10,
              l_netUpdates);

          // accumulate net updates to cell-wise net updates for h and hu
          hNetUpdatesL(i - 1, j) += dx_inv * l_netUpdates[hNetUpLeft];
          huNetUpdatesL(i - 1, j) += dx_inv * l_netUpdates[huNetUpLeft];
          hNetUpdatesR(i, j) += dx_inv * l_netUpdates[hNetUpRight];
          huNetUpdatesR(i, j) += dx_inv * l_netUpdates[huNetUpRight];

          l_maxWaveSpeed.combine(l_netUpdates[maxEdgeSpeed]);
        });

    RAJA::kernel<policies::loop_2d<true>>(
        RAJA::make_tuple(RAJA::RangeSegment(1, nx + 1), RAJA::RangeSegment(1, ny + 2)), [=] RAJA_HOST_DEVICE (size_t i, size_t j) {
          constexpr unsigned hNetUpDow = 0, hNetUpUpw = 1, hvNetUpDow = 2, hvNetUpUpw = 3, maxEdgeSpeed = 4;
          float l_netUpdates[5];

          augRieComputeNetUpdates (h(i, j - 1), h(i, j), hv(i, j - 1), hv(i, j), b(i, j - 1), b(i, j),
                             static_cast<real>(9.81), static_cast<real>(0.01), static_cast<real>(0.000001), static_cast<real>(0.0001), 10,
                             l_netUpdates);

          // accumulate net updates to cell-wise net updates for h and hu
          hNetUpdatesL(i, j - 1) += dy_inv * l_netUpdates[hNetUpDow];
          hvNetUpdatesL(i, j - 1) += dy_inv * l_netUpdates[hvNetUpDow];
          hNetUpdatesR(i, j) += dy_inv * l_netUpdates[hNetUpUpw];
          hvNetUpdatesR(i, j) += dy_inv * l_netUpdates[hvNetUpUpw];

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
  auto hNetUpdatesL =  this->hNetUpdatesL.getDeviceView();
  auto huNetUpdatesL = this->huNetUpdatesL.getDeviceView();
  auto hvNetUpdatesL = this->hvNetUpdatesL.getDeviceView();
  auto hNetUpdatesR =  this->hNetUpdatesR.getDeviceView();
  auto huNetUpdatesR = this->huNetUpdatesR.getDeviceView();
  auto hvNetUpdatesR = this->hvNetUpdatesR.getDeviceView();
  auto h = this->h.getDeviceView();
  auto hu = this->hu.getDeviceView();
  auto hv = this->hv.getDeviceView();

  RAJA::kernel<policies::loop_2d<>>(
      RAJA::make_tuple(RAJA::RangeSegment(1, nx + 1), RAJA::RangeSegment(1, ny + 1)), [=] RAJA_HOST_DEVICE (size_t i, size_t j) {
        h(i, j)  -= dt * (hNetUpdatesL(i, j) + hNetUpdatesR(i, j));
        hu(i, j) -= dt * (huNetUpdatesL(i, j) + huNetUpdatesR(i, j));
        hv(i, j) -= dt * (hvNetUpdatesL(i, j) + hvNetUpdatesR(i, j));

         hNetUpdatesL(i, j) = (float) 0;
        huNetUpdatesL(i, j) = (float) 0;
        hvNetUpdatesL(i, j) = (float) 0;
         hNetUpdatesR(i, j) = (float) 0;
        huNetUpdatesR(i, j) = (float) 0;
        hvNetUpdatesR(i, j) = (float) 0;

        //TODO: proper dryTol
        if (h(i, j) < 0.1)
          hu(i, j) = hv(i, j) = 0.; //no water, no speed!

        if (h(i, j) < 0) {
          //zero (small) negative depths
          h(i, j) = (float) 0;
        }
      });
}