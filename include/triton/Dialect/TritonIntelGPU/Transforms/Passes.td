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
    Option<"deviceArch", "device architecture",
            "mlir::triton::gpu::intel::DeviceArch", /*default*/" mlir::triton::gpu::intel::DeviceArch::UNKNOWN",
            "device architecture">
  ];
}

def TritonIntelGPUPipeline : Pass<"tritonintelgpu-pipeline", "mlir::ModuleOp"> {
  let summary = "Intel GPU pipeline";

  let description = [{
    Use the cache prefetch to buffer the `LoadOp` in loops that needed at next iteration.
  }];

  let constructor = "mlir::createTritonIntelGPUPipelinePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">
  ];
}

def TritonIntelGPUMaterializeBlockPointer : Pass<"tritonintelgpu-materialize-block-pointer", "mlir::ModuleOp"> {
  let summary = "Intel GPU materialize block pointer";

  let description = [{
  }];

  let constructor = "mlir::createTritonIntelGPUMaterializeBlockPointerPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    //Option<"numStages", "num-stages",
    //       "int32_t", /*default*/"2",
    //       "number of pipeline stages">
  ];
}

#endif
