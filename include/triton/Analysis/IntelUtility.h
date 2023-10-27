//
// Created by chengjun on 8/7/23.
//

#ifndef TRITON_INTELUTILITY_H
#define TRITON_INTELUTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>
#include <string>

namespace mlir {
namespace triton {
namespace intel {

enum class DeviceArch {
  ATS = 0,
  PVC = 1,
  UNKNOWN,
};

struct IntelXMXCapability {
  uint32_t systolicDepth;
  uint32_t repeatCount;
  uint32_t executionSize;
  uint32_t opsChanBitWidths;
};

DeviceArch
computeCapabilityToXMXArch(const std::map<std::string, int> &computeCapability);

bool supportXMX(Value value, DeviceArch arch);

bool supportXMX(triton::DotOp op, DeviceArch arch);

IntelXMXCapability getXMXCapability(DeviceArch arch);

} // namespace intel
} // namespace triton
} // namespace mlir

#endif // TRITON_INTELUTILITY_H
