#include "DeviceFloat2D.hh"

DeviceFloat2D::ViewBase::ViewBase(size_t rows, float *data)
  :data(data), rows(rows) {
}

DeviceFloat2D::DeviceFloat2D(size_t cols, size_t rows) : cols(cols), rows(rows), data(swe_alloc<float>(cols * rows)) {}

DeviceFloat2D::~DeviceFloat2D() {
  swe_cleanup_cpuPtr<float>(data, cpuData);
  swe_free(data);
}

void DeviceFloat2D::updateCpuData() const {
  if (cpuVersion < deviceVersion) {
    swe_sync_from_device<float>(data, cpuData, cols * rows);
    cpuVersion = deviceVersion;
  }
}

void DeviceFloat2D::updateDevice() {
  if (deviceVersion < cpuVersion) {
    swe_sync_to_device<float>(data, cpuData, cols * rows);
    deviceVersion = cpuVersion;
  }
}

DeviceFloat2D::View<false> DeviceFloat2D::getReadonlyCPUView() const {
  updateCpuData();
  return DeviceFloat2D::View<false>(rows, cpuData);
}

DeviceFloat2D::View<true> DeviceFloat2D::getMutableCPUView() {
  updateCpuData();
  cpuVersion++;
  return DeviceFloat2D::View<true>(rows, cpuData);
}
DeviceFloat2D::View<true> DeviceFloat2D::getDeviceView() {
  updateDevice();
  deviceVersion++;
  return DeviceFloat2D::View<true>(rows, data);
}

/*BackedFloat1D getColProxy(int i) const {
    // subarray elem[i][*]:
    // starting at elem[i][0] with rows elements and unit stride
    return getProxyUnsafe(rows * i, rows);
  };

  BackedFloat1D getRowProxy(int j) const {
    // subarray elem[*][j]
    // starting at elem[0][j] with cols elements and stride rows
    return getProxyUnsafe(j, cols, rows);
  };*/

BackedFloat1D DeviceFloat2D::getColProxy(size_t i) const { return BackedFloat1D(*this, rows * i, rows); }
BackedFloat1D DeviceFloat2D::getRowProxy(size_t j) const { return BackedFloat1D(*this, j, cols, rows); }

const Float1D BackedFloat1D::getReadonly() const {
  return back.getReadonlyCPUView().getProxy(offset, rows, stride);
}
