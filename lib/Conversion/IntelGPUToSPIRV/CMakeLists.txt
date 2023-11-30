add_mlir_conversion_library(TritonIntelGPUToSPIRV
    IntelGPUToSPIRVPass.cpp

    ADDITIONAL_HEADER_DIRS
#    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/IntelGPUToSPIRV
#    ${PROJECT_BINARY_DIR}/include/triton/Conversion/IntelGPUToSPIRV
    ${PROJECT_SOURCE_DIR}/third_party/intel_xpu_backend/lib/Conversion/TritonGPUToSPIRV

    DEPENDS
    TritonIntelGPUToSPIRVConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    LLVMGenXIntrinsics
    MLIRTargetLLVMIRImport
    GenISAIntrinsics
)
