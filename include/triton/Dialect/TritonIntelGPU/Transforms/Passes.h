#ifndef TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createTritonIntelGPUAccelerateMatmulPass(
    std::map<std::string, int> computeCapability = {});

std::unique_ptr<Pass> createTritonIntelGPUDecomposeConversionsPass(
    std::map<std::string, int> computeCapability = {});

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif
