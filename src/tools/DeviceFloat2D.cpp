#include "DeviceFloat2D.hh"

DeviceFloat2D::CPUViewBase::CPUViewBase(int cols, int rows, float *cpuData)
  :cols(cols), rows(rows), cpuData(cpuData) {
}

DeviceFloat2D::CPUViewBase::~CPUViewBase() = default;

DeviceFloat2D::DeviceFloat2D(int cols, int rows) : cols(cols), rows(rows), data(swe_alloc<float>(cols * rows)) {}

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

DeviceFloat2D::CPUView<false> DeviceFloat2D::getReadonlyCPUView() const {
  updateCpuData();
  return DeviceFloat2D::CPUView<false>(cols, rows, cpuData);
}

DeviceFloat2D::CPUView<true> DeviceFloat2D::getMutableCPUView() {
  updateCpuData();
  cpuVersion++;
  return DeviceFloat2D::CPUView<true>(cols, rows, cpuData);
}
RAJA::View<float, RAJA::Layout<2>> DeviceFloat2D::getDeviceView() {
  updateDevice();
  deviceVersion++;
  return RAJA::View<float, RAJA::Layout<2>>(data, cols, rows);
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

BackedFloat1D DeviceFloat2D::getColProxy(int i) const { return BackedFloat1D(*this, rows * i, rows); }
BackedFloat1D DeviceFloat2D::getRowProxy(int j) const { return BackedFloat1D(*this, j, cols, rows); }

const Float1D BackedFloat1D::getReadonly() const {
  return back.getReadonlyCPUView().getProxy(offset, rows, stride);
}
