#ifndef TRITON_TRITON_INTEL_GPU_TO_SPIRV_PASS_H
#define TRITON_TRITON_INTEL_GPU_TO_SPIRV_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonIntelGPUToSPIRVPass(
    std::map<std::string, int> computeCapability = {});

} // namespace triton
} // namespace mlir

#endif // TRITON_TRITON_INTEL_GPU_TO_SPIRV_PASS_H
