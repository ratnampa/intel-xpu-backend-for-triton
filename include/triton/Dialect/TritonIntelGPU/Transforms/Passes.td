#ifndef TRITON_INTEL_GPU_PASSES
#define TRITON_INTEL_GPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonIntelGPUAccelerateMatmul : Pass<"tritonintelgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "intel accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
  }];

  let constructor = "mlir::createTritonIntelGPUAccelerateMatmulPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"computeCapability", "compute-capability",
            "std::map<std::string, int>", /*default*/"std::map<std::string, int>{}",
            "device compute capability">
  ];
}

def TritonIntelGPUDecomposeConversions: Pass<"tritonintelgpu-decompose-conversions", "mlir::ModuleOp"> {
  let summary = "Decompose convert[mma -> blocked] into convert[mma -> shared -> blocked]";

  let description = "Because of the JointMatrix Type is opaque. store it to memory and load it back ";

  let constructor = "mlir::createTritonIntelGPUDecomposeConversionsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::triton::TritonDialect"];
}

#endif
