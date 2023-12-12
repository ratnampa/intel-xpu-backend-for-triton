//
// Created by chengjun on 8/7/23.
//
#include "triton/Analysis/IntelUtility.h"

namespace mlir {
namespace triton {
namespace intel {

static IntelXMXCapability caps[] = {
    [(uint32_t)DeviceArch::ATS] =
        {
            .systolicDepth = 8,
            .repeatCount = 8,
            .executionSize = 8,
            .opsChanBitWidths = 32,
        },

    [(uint32_t)DeviceArch::PVC] =
        {
            .systolicDepth = 8,
            .repeatCount = 8,
            .executionSize = 16,
            .opsChanBitWidths = 32,
        },
};

IntelXMXCapability getXMXCapability(DeviceArch arch) {
  assert(arch <= DeviceArch::UNKNOWN && "Unknown Intel GPU archs");
  return caps[(uint32_t)arch];
}

bool supportXMX(Value value, DeviceArch arch) {
  if (arch == DeviceArch::UNKNOWN)
    return false;
  assert((arch == DeviceArch::ATS || arch == DeviceArch::PVC) &&
         "Unexpected MMA layout version found");
  auto elemTy = value.getType().cast<RankedTensorType>().getElementType();
  return elemTy.isF16() || elemTy.isBF16(); /* ||
           (elemTy.isF32() && version >= 2) ||
           (elemTy.isInteger(8) && version >= 2);*/
}

bool supportXMX(triton::DotOp op, DeviceArch arch) {
  auto aElemTy = op.getA().getType().cast<RankedTensorType>().getElementType();
  auto bElemTy = op.getB().getType().cast<RankedTensorType>().getElementType();
  if (aElemTy.isF32() && bElemTy.isF32()) {
    return op.getAllowTF32() && arch == DeviceArch::PVC;
  }
  return supportXMX(op.getA(), arch) && supportXMX(op.getB(), arch);
}

} // namespace intel
} // namespace triton
} // namespace mlir
