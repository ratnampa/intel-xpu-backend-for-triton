
#include "triton/Dialect/TritonIntelGPU/Transforms/Utility.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

bool supportDPAS(DotOp op, DeviceArch arch) {
  auto aElemTy = op.getA().getType().cast<RankedTensorType>().getElementType();
  auto bElemTy = op.getB().getType().cast<RankedTensorType>().getElementType();
  auto dElemTy =
      op.getResult().getType().cast<RankedTensorType>().getElementType();

  // Skip the dot op with known issue.
//  if (aElemTy.isF32() && bElemTy.isF32()) {
//    // The FP32-FP32-FP32 data type result
//    // incorrect:https://github.com/intel/intel-xpu-backend-for-triton/issues/402
//    return false;
//  }
  if (dElemTy.isF16()) {
    // The FP16-FP16-FP16 data type result
    // incorrect:https://github.com/intel/intel-xpu-backend-for-triton/issues/400
    return false;
  }

  if (arch == DeviceArch::UNKNOWN)
    return false;
  assert((arch == DeviceArch::ATS || arch == DeviceArch::PVC) &&
         "Unexpected MMA layout version found");

  if (getDPASType(op) != DPASEngineType::NOT_APPLICABLE)
    return true;

  return false;
}

DPASEngineType getDPASType(DotOp op) {
  // d = a*b + c
  auto aTy = op.getA().getType().cast<RankedTensorType>().getElementType();
  auto bTy = op.getB().getType().cast<RankedTensorType>().getElementType();
  auto cTy = op.getC().getType().cast<RankedTensorType>().getElementType();
  auto dTy = op.getD().getType().cast<RankedTensorType>().getElementType();

  // Overall check
  if (aTy != bTy || cTy != dTy)
    return DPASEngineType::NOT_APPLICABLE;

  // TODO: add more dpas supported data type.
  if (dTy.isa<FloatType>()) {
    // floating.
    if (dTy.isF32()) {

      if (aTy.isF16() && bTy.isF16())
        return DPASEngineType::FP32_FP32_FP16_FP16;
      if (aTy.isBF16() && bTy.isBF16())
        return DPASEngineType::FP32_FP32_BF16_BF16;
      if (aTy.isF32() && bTy.isF32() && op.getAllowTF32())
        return DPASEngineType::FP32_FP32_TF32_TF32;

    } else if (dTy.isF16()) {

      if (aTy.isF16() && bTy.isF16())
        return DPASEngineType::FP16_FP16_FP16_FP16;

    } else if (dTy.isBF16()) {

      if (aTy.isBF16() && bTy.isBF16())
        return DPASEngineType::BF16_BF16_BF16_BF16;
    }
  } else {
    // Integer
    if (dTy.getIntOrFloatBitWidth() == 32) {

      if (aTy.getIntOrFloatBitWidth() == 8 && bTy.getIntOrFloatBitWidth() == 8)
        return dTy.isSignedInteger() ? DPASEngineType::S32_S32_S8_S8
                                     : DPASEngineType::U32_U32_U8_U8;
    }
  }

  return DPASEngineType::NOT_APPLICABLE;
}

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
