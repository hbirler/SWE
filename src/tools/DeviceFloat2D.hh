#ifndef _TOOLS_SYSTEMFLOAT2D_HH
#define _TOOLS_SYSTEMFLOAT2D_HH

#include <memory>
#include <cstdint>
#include "cross/MemoryManager.hh"
#include "tools/help.hh"
#include <RAJA/util/macros.hpp>

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
  class ViewBase {
  protected:
    /// The data
    float* data;
    /// Number of rows
    size_t rows;
    /// Constructor
    ViewBase(size_t rows, float* data);

    Float1D getProxy(size_t offset, size_t rows, size_t stride = 1) const {
      return Float1D(data + offset, rows, stride);
    };

    friend class DeviceFloat2D;
    friend class BackedFloat1D;
  public:
    ViewBase(const ViewBase &) = default;
    ViewBase(ViewBase &&) = default;
    ~ViewBase() = default;
  };

  template <bool Mutable>
  class LightView {
    /// The data on cpu
    float* data;
  public:
    RAJA_HOST_DEVICE RAJA_INLINE maybe_const_t<float, !Mutable>* operator()(size_t rows, size_t i) const { return data + i * rows; }
    RAJA_HOST_DEVICE RAJA_INLINE maybe_const_ref_t<float, !Mutable> operator()(size_t rows, size_t i, size_t j) const { return *(data + i * rows + j); }
    LightView(float* data) : data(data) {}
  };

  template <bool Mutable>
  class View : public ViewBase {
  public:
    RAJA_HOST_DEVICE RAJA_INLINE maybe_const_t<float, !Mutable>* operator()(size_t i) const { return data + i * rows; }
    RAJA_HOST_DEVICE RAJA_INLINE maybe_const_ref_t<float, !Mutable> operator()(size_t i, size_t j) const { return *(data + i * rows + j); }
    View(size_t rows, float* cpuData) : ViewBase(rows, cpuData) {}

    LightView<Mutable> asLight() const { return {data}; }
  };


  /// Constructor
  DeviceFloat2D(size_t cols, size_t rows);
  /// Destructor
  ~DeviceFloat2D();

  DeviceFloat2D(const DeviceFloat2D&) = delete;
  DeviceFloat2D(DeviceFloat2D&&) = delete;

  /// Return a multi-use proxy that is able to access a certain view.
  BackedFloat1D getColProxy(size_t i) const;
  /// Return a multi-use proxy that is able to access a certain view.
  BackedFloat1D getRowProxy(size_t j) const;

  /// Get single-use CPU view that is readonly
  View<false> getReadonlyCPUView() const;
  /// Get single-use CPU view that is mutable
  View<true> getMutableCPUView();
  /// Get single-use device view
  View<true> getDeviceView();

  inline size_t getRows() const { return rows; };
  inline size_t getCols() const { return cols; };
};



using ViewF2D = DeviceFloat2D::View<false>;
using ViewF2DMutable = DeviceFloat2D::View<true>;

#endif