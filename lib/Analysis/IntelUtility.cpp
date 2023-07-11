//
// Created by chengjun on 8/7/23.
//
#include "triton/Analysis/IntelUtility.h"

namespace mlir {

DeviceArch computeCapabilityToXMXArch(
    const std::map<std::string, int> &computeCapability) {
  if (computeCapability.find("ATS") != computeCapability.end()) {
    return DeviceArch::ATS;
  } else if (computeCapability.find("PVC") != computeCapability.end()) {
    return DeviceArch::PVC;
  } else {
    return DeviceArch::UNKNOWN;
    ;
  }
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

bool supportXMX(triton::DotOp op, mlir::DeviceArch arch) {
  auto aElemTy = op.getA().getType().cast<RankedTensorType>().getElementType();
  auto bElemTy = op.getB().getType().cast<RankedTensorType>().getElementType();
  if (aElemTy.isF32() && bElemTy.isF32()) {
    return op.getAllowTF32() && arch == DeviceArch::PVC;
  }
  return supportXMX(op.getA(), arch) && supportXMX(op.getB(), arch);
}

} // namespace mlir
