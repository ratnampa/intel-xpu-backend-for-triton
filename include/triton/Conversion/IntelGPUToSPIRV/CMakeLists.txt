set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonIntelGPUToSPIRV)
add_public_tablegen_target(TritonIntelGPUToSPIRVConversionPassIncGen)
