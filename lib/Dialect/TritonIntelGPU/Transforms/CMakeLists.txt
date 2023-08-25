add_mlir_dialect_library(TritonIntelGPUTransforms
  AccelerateMatmul.cpp
  DecomposeConversions.cpp

  DEPENDS
  TritonIntelGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonIntelGPUIR
)
