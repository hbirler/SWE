#ifndef _TOOLS_SYSTEMFLOAT2D_HH
#define _TOOLS_SYSTEMFLOAT2D_HH

#include <memory>
#include <cstdint>
#include <RAJA/RAJA.hpp>
#include "cross/MemoryManager.hh"
#include "tools/help.hh"

template <typename T, bool IsConst> struct MaybeConstRef;
template <typename T> struct MaybeConstRef<T, true> { using type = const T&; };
template <typename T> struct MaybeConstRef<T, false> { using type = T&; };
template <typename T, bool IsConst> using maybe_const_ref_t = typename MaybeConstRef<T, IsConst>::type;


template <typename T, bool IsConst> struct MaybeConst;
template <typename T> struct MaybeConst<T, true> { using type = const T; };
template <typename T> struct MaybeConst<T, false> { using type = T; };
template <typename T, bool IsConst> using maybe_const_t = typename MaybeConst<T, IsConst>::type;

class BackedFloat1D;
class DeviceFloat2D;

/// Float1D backed by a DeviceFloat2D
class BackedFloat1D {
  const DeviceFloat2D& back;
  int offset;
  int rows;
  int stride;

public:
  BackedFloat1D(const DeviceFloat2D& back, int offset, int rows, int stride = 1) : back(back), offset(offset), rows(rows), stride(stride) {}
  const Float1D getReadonly() const;
};

/// Allocates memory on system and provides access.
/// No guarantees are made for simultaneous access to data from cpu and device
class DeviceFloat2D {
private:
  /// Number of columns
  int cols;
  /// Number of rows
  int rows;
  /// The data on device
  float* data;
  /// The data on CPU
  mutable float* cpuData = nullptr;
  /// The version of data on device
  uint64_t deviceVersion = 1;
  /// The version of data on cpu
  mutable uint64_t cpuVersion = 0;

  void updateCpuData() const;
  void updateDevice();
public:
  class CPUViewBase {
  protected:
    /// Number of columns
    int cols;
    /// Number of rows
    int rows;
    /// The data on cpu
    float* cpuData;
    /// Constructor
    CPUViewBase(int cols, int rows, float* cpuData);


    Float1D getProxy(int offset, int rows, int stride = 1) const {
      return Float1D(cpuData + offset, rows, stride);
    };

    friend class DeviceFloat2D;
    friend class BackedFloat1D;
  public:
    ~CPUViewBase();
  };

  template <bool Mutable>
  class CPUView : public CPUViewBase {
  public:
    maybe_const_t<float, !Mutable>* operator()(int i) const { return cpuData + i * rows; }
    maybe_const_ref_t<float, !Mutable> operator()(int i, int j) const { return *(cpuData + i * rows + j); }
    CPUView(int cols, int rows, float* cpuData) : CPUViewBase(cols, rows, cpuData) {}
  };

  /// Constructor
  DeviceFloat2D(int cols, int rows);
  /// Destructor
  ~DeviceFloat2D();

  DeviceFloat2D(const DeviceFloat2D&) = delete;
  DeviceFloat2D(DeviceFloat2D&&) = delete;

  /// Return a multi-use proxy that is able to access a certain view.
  BackedFloat1D getColProxy(int i) const;
  /// Return a multi-use proxy that is able to access a certain view.
  BackedFloat1D getRowProxy(int j) const;

  /// Get single-use CPU view that is readonly
  CPUView<false> getReadonlyCPUView() const;
  /// Get single-use CPU view that is mutable
  CPUView<true> getMutableCPUView();
  /// Get single-use device view
  RAJA::View<float, RAJA::Layout<2>> getDeviceView();

  constexpr int getRows() const { return rows; };
  constexpr int getCols() const { return cols; };
};



using ViewF2D = DeviceFloat2D::CPUView<false>;
using ViewF2DMutable = DeviceFloat2D::CPUView<true>;

#endif