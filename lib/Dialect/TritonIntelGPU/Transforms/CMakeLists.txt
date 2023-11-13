add_mlir_dialect_library(TritonIntelGPUTransforms
  AccelerateMatmul.cpp
  DecomposeConversions.cpp
  BufferDotOperandsInCache.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/PipelineExpander.cpp
  Pipeliner/SoftwarePipeliner.cpp

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
