add_mlir_library(TritonIntelAnalysis
  IntelUtility.cpp

  DEPENDS
  TritonTableGen
  TritonIntelGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
