#ifndef TRITON_INTEL_GPU_CONVERSION_PASSES
#define TRITON_INTEL_GPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTritonIntelGPUToSPIRV : Pass<"convert-triton-intel-gpu-to-spriv", "mlir::ModuleOp"> {
    let summary = "Convert TritonIntelGPU to SPIRV";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonIntelGPUToSPIRVPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::spirv::SPIRVDialect",
                             "mlir::triton::gpu::intel::TritonIntelGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
                "std::map<std::string, int>", /*default*/"std::map<std::string, int>{}",
                "device compute capability">
    ];
}

#endif
